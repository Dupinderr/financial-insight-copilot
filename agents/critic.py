"""Critic / Verifier Agent — checks every numeric claim against the evidence.

Verification runs in three stages, deliberately, because neither a regex nor an
LLM is trustworthy alone here:

1. Deterministic sweep. Pull every significant number out of the draft and try
   to match it against the numbers present in the evidence (allowing for
   rounding and the usual presentation rescalings). Numbers that match are
   almost certainly fine; numbers that don't are *candidates*, not verdicts —
   a legitimately derived figure won't appear verbatim either.

2. LLM adjudication. The model lists each numeric claim and, when it calls one
   supported, must name the exact figure from the evidence it relies on.

3. Citation check. That named figure is then verified against the evidence
   itself. An LLM claiming support while citing a number that isn't there is
   the exact failure mode this agent exists to catch, so the final gate is
   deterministic rather than trusting the adjudication.

A claim is flagged unless it survives all three.
"""

import re

from observability.tracing import llm_call

from .utils import (
    extract_json,
    extract_numbers,
    matches_any,
    matches_exact,
    significant_numbers,
    to_float,
)

UNVERIFIED_MARKER = "⚠️ unverified"

CRITIC_PROMPT = """You are a fact-checker verifying a financial research note. You are strict and literal.

THE EVIDENCE — the only facts that exist
=========================================
{evidence}

THE DRAFT NOTE TO CHECK
=======================
{draft}

{hint_block}

List every numeric claim in the draft note. For each one, decide whether it is
traceable to the evidence above.

What counts as a numeric claim:
- A figure the note asserts as a fact about the company or its market data.

What is NOT a numeric claim — do NOT list these:
- Sentences containing no number at all.
- Numbers that merely name the metric being discussed: the "14" in "14-day RSI",
  the "30" in "30-day moving average", or a threshold restated from the question
  such as the "70" in "RSI above 70". These are labels, not assertions.
- Citation markers like "[DOC-1]" or "[DOC-3]". The digit identifies a source
  passage; it asserts nothing.
- Dates. "on July 27, 2026" is a timestamp, not a quantity being claimed.

When you copy a claim verbatim, copy only the phrase containing the figure.
Never include a trailing citation marker or a date fragment in it.

Reply with ONLY a JSON array, no prose:
[
  {{
    "claim": "the exact phrase from the draft containing the number, copied verbatim",
    "value": "the number itself, e.g. 3996.42",
    "verdict": "supported" or "unsupported" or "contradicted",
    "supporting_value": "the exact figure from the evidence that backs this claim, or null",
    "source": "which evidence it came from, e.g. the SQL column name or DOC-2, or null",
    "note": "one short sentence"
  }}
]

Verdict rules:
- "supported"    — the figure appears in the evidence, allowing for rounding or
                   a restated unit. You MUST fill in "supporting_value" with the
                   exact figure as it appears in the evidence. A claim with a
                   null "supporting_value" is NOT supported.
- "contradicted" — the evidence contains this quantity but with a different value.
- "unsupported"  — the evidence does not contain this quantity at all.

Do not be generous. If you cannot point to the specific figure in the evidence,
the verdict is "unsupported". Copy "claim" verbatim from the draft — it is used
for exact string matching."""

HINT_TEMPLATE = """AUTOMATED PRE-CHECK
===================
A numeric scan found these figures in the draft that do NOT appear anywhere in
the evidence (after allowing for rounding and unit rescaling):

  {unmatched}

Scrutinise these especially closely. Each is either a figure the analyst derived
by calculation — in which case say so and name the inputs — or it is fabricated."""


def build_evidence_pool(
    sql_result: dict | None = None,
    doc_result: dict | None = None,
    fallback_text: str | None = None,
) -> list[float]:
    """Collect the numbers that genuinely count as evidence.

    Deliberately built from result *values* rather than by scanning the
    rendered evidence text: that text embeds the SQL query, and literals in it
    (`rsi_14`, `'2025'`, `LIMIT 1`) would otherwise be treated as facts and
    could accidentally vouch for a fabricated figure.
    """
    pool: list[float] = []

    if sql_result and sql_result.get("rows"):
        for row in sql_result["rows"]:
            for value in row.values():
                # Strings get every embedded number extracted, not squashed
                # through to_float: a Date of "2026-07-27 00:00:00" must
                # contribute 2026, 7 and 27 so a note saying "on July 27, 2026"
                # verifies, rather than one meaningless 20260727000000.
                if isinstance(value, str):
                    pool.extend(extract_numbers(value))
                else:
                    number = to_float(value)
                    if number is not None:
                        pool.append(number)

    if doc_result and doc_result.get("chunks"):
        for chunk in doc_result["chunks"]:
            pool.extend(extract_numbers(chunk.get("text", "")))

    # Only fall back to scraping the rendered text when no structured results
    # were handed over at all.
    if not pool and fallback_text:
        pool.extend(extract_numbers(fallback_text))

    return pool


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

# ISO dates (2026-07-27), and written dates in either order
# ("July 27, 2026" / "27 July 2026").
_DATE = re.compile(
    rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{0,4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s*\d{{0,4}}",
    re.IGNORECASE,
)


def build_context_pool(
    question: str | None,
    sql_result: dict | None = None,
    draft: str | None = None,
) -> list[float]:
    """Numbers that are part of the question's own vocabulary, not claims.

    "14-day RSI", "30-day moving average", "RSI above 70" — the 14, 30 and 70
    are metric labels and thresholds restated from the question or the column
    names, not assertions about the world. They will never appear in a result
    row, so without this they get flagged as fabricated and drown the real
    findings in noise.

    These numbers are neither evidence nor claims: they are simply ignored.
    """
    pool: list[float] = []

    if question:
        pool.extend(extract_numbers(question))

    # Date components are timestamps, not quantities being asserted. "on July
    # 27, 2026" should never be flagged as an unsupported figure.
    for text in (draft, question):
        for match in _DATE.finditer(text or ""):
            pool.extend(extract_numbers(match.group()))

    if sql_result:
        # Column names carry the metric parameters: rsi_14, ma_30, volatility_30d.
        for row in sql_result.get("rows") or []:
            for column in row.keys():
                pool.extend(extract_numbers(str(column)))
        # The query text carries thresholds and window sizes the user asked for.
        pool.extend(extract_numbers(sql_result.get("sql") or ""))

    return pool


# Citation markers the synthesizer emits, e.g. "[DOC-2]". The digit is a
# reference to a retrieved passage, not an assertion about the world.
_CITATION = re.compile(r"\[?\s*DOC[-–\s]?(\d+)\s*\]?", re.IGNORECASE)


def strip_citations(text: str) -> str:
    """Remove [DOC-n] markers so their digits aren't mistaken for claims."""
    return _CITATION.sub(" ", text or "")


def strip_dates(text: str) -> str:
    """Remove date expressions so their components aren't mistaken for claims."""
    return _DATE.sub(" ", text or "")


def date_only_numbers(draft: str) -> list[float]:
    """Numbers that appear in the draft *only* inside a date.

    A figure that also occurs outside a date is still a real claim, so this
    subtracts the non-date numbers rather than blanket-ignoring anything that
    looks like a day or month.
    """
    in_dates: list[float] = []
    for match in _DATE.finditer(draft or ""):
        in_dates.extend(extract_numbers(match.group()))

    elsewhere = significant_numbers(strip_citations(strip_dates(draft)))
    return [n for n in in_dates if not matches_exact(n, elsewhere)]


def _is_citation_claim(entry: dict) -> bool:
    """Did the model list a citation marker as though it were a claim?"""
    claim = str(entry.get("claim") or "")
    if "doc-" in claim.lower().replace(" ", "").replace("–", "-"):
        # A citation reference, unless there is a real figure alongside it.
        return not significant_numbers(strip_citations(claim))
    return False


def _is_numeric_claim(entry: dict) -> bool:
    """Filter out prose the model listed as a 'claim' with no number in it."""
    if _is_citation_claim(entry):
        return False
    if to_float(entry.get("value")) is not None:
        return True
    return bool(significant_numbers(strip_citations(str(entry.get("claim") or ""))))


def _build_evidence_text(sql_evidence: str | None, doc_evidence: str | None) -> str:
    parts = []
    if sql_evidence:
        parts.append("--- QUANTITATIVE (market database) ---\n" + sql_evidence)
    if doc_evidence:
        parts.append("--- QUALITATIVE (documents) ---\n" + doc_evidence)
    return "\n\n".join(parts) if parts else "(no evidence was gathered)"


def _annotate(draft: str, claims: list[dict], strip_unverified: bool) -> tuple[str, list[dict]]:
    """Mark unverified claims in the draft text.

    Default is flag-only: the claim stays and gets a marker. Stripping is
    opt-in because silently deleting a number is worse than showing a
    questionable one next to a warning.
    """
    lines = draft.split("\n")
    unplaced = []
    flagged_lines: set[int] = set()

    for claim in claims:
        if claim["verdict"] in {"supported", "contextual"}:
            continue

        phrase = (claim.get("claim") or "").strip()
        if not phrase:
            unplaced.append(claim)
            continue

        index = next((i for i, line in enumerate(lines) if phrase in line), None)
        if index is None:
            unplaced.append(claim)
            continue

        if strip_unverified:
            lines[index] = lines[index].replace(phrase, f"[{UNVERIFIED_MARKER} claim removed]", 1)
            continue

        # Mark at end of line rather than immediately after the phrase.
        # Inline insertion splits whatever the phrase happens to end inside —
        # "July 27 ⚠️, 2026" or "[DOC-1 ⚠️]" — which mangles the note. One
        # marker per line keeps it readable when a line carries two bad claims.
        if index not in flagged_lines:
            lines[index] = f"{lines[index]} {UNVERIFIED_MARKER}"
            flagged_lines.add(index)

    return "\n".join(lines), unplaced


def _material(claims: list[dict]) -> list[dict]:
    """Claims that actually bear on correctness — contextual labels excluded."""
    return [c for c in claims if c["verdict"] != "contextual"]


def _confidence(claims: list[dict], had_evidence: bool) -> str:
    if not had_evidence:
        return "low"

    claims = _material(claims)
    if not claims:
        return "medium"  # nothing numeric to verify either way

    supported = sum(1 for c in claims if c["verdict"] == "supported")
    total = len(claims)

    if supported == total:
        return "high"
    if supported >= total * 0.7:
        return "medium"
    return "low"


def criticize(
    draft: str,
    sql_evidence: str | None = None,
    doc_evidence: str | None = None,
    strip_unverified: bool = False,
    sql_result: dict | None = None,
    doc_result: dict | None = None,
    question: str | None = None,
) -> dict:
    """Verify the draft's numeric claims and return an annotated note.

    Pass `sql_result` / `doc_result` (the raw tool outputs) when available —
    they give a far more precise numeric pool than the rendered evidence text.
    Pass `question` so metric parameters restated from it aren't mistaken for
    fabricated figures.
    """
    evidence = _build_evidence_text(sql_evidence, doc_evidence)
    had_evidence = bool(sql_evidence or doc_evidence)
    evidence_pool = build_evidence_pool(
        sql_result=sql_result, doc_result=doc_result, fallback_text=evidence
    )
    context_pool = build_context_pool(question, sql_result, draft=draft)

    def is_contextual(value: float | None) -> bool:
        """A metric label restated from the question, not an assertion.

        Exact matching only — a rescaled match here would let a fabricated
        figure hide behind a metric label that happens to be 100x smaller.
        """
        return (
            value is not None
            and not matches_any(value, evidence_pool)
            and matches_exact(value, context_pool)
        )

    # Stage 1 — deterministic sweep, over the draft with citation markers and
    # dates removed: "[DOC-2]" is not a claim about 2, and "July 27, 2026" is
    # a timestamp rather than an asserted quantity.
    draft_numbers = significant_numbers(strip_dates(strip_citations(draft)))
    date_numbers = date_only_numbers(draft)
    unmatched = [
        n for n in draft_numbers
        if not matches_any(n, evidence_pool) and not is_contextual(n)
    ]

    hint_block = ""
    if unmatched:
        preview = ", ".join(f"{n:g}" for n in dict.fromkeys(unmatched))
        hint_block = HINT_TEMPLATE.format(unmatched=preview)

    # Stage 2 — LLM adjudication.
    raw = llm_call(
        CRITIC_PROMPT.format(evidence=evidence, draft=draft, hint_block=hint_block),
        name="critic",
        max_tokens=1200,
    )
    parsed = extract_json(raw, default=None)

    if not isinstance(parsed, list):
        # Without a usable verdict list, fall back to the deterministic sweep
        # alone rather than declaring the note clean.
        claims = [
            {
                "claim": f"{n:g}",
                "value": f"{n:g}",
                "verdict": "unsupported",
                "supporting_value": None,
                "source": None,
                "note": "critic output unparseable; figure not found in evidence by numeric scan",
                "checked_by": "deterministic",
            }
            for n in unmatched
        ]
        annotated, unplaced = _annotate(draft, claims, strip_unverified)
        return {
            "verified_answer": annotated,
            "claims": claims,
            "unverified_count": len(claims),
            "verified_count": 0,
            "confidence": "low",
            "unplaced_claims": unplaced,
            "critic_parse_failed": True,
            "raw": raw,
        }

    # Stage 3 — citation check on everything the model called supported.
    claims = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue

        # The model sometimes lists whole prose sentences as "claims" with a
        # null value; those default to unsupported and produce phantom flags.
        if not _is_numeric_claim(entry):
            continue

        # A date component the model mistook for a figure. Dropped outright
        # rather than marked contextual — it was never a claim to begin with.
        if matches_exact(to_float(entry.get("value")), date_numbers):
            continue

        verdict = str(entry.get("verdict", "unsupported")).lower().strip()
        if verdict not in {"supported", "unsupported", "contradicted"}:
            verdict = "unsupported"

        supporting_value = entry.get("supporting_value")
        checked_by = "llm"

        claim_value = to_float(entry.get("value"))
        if verdict != "supported" and is_contextual(claim_value):
            claims.append(
                {
                    "claim": entry.get("claim", ""),
                    "value": entry.get("value"),
                    "verdict": "contextual",
                    "supporting_value": None,
                    "source": None,
                    "note": "metric label or threshold restated from the question, not a claim",
                    "checked_by": "context_filter",
                }
            )
            continue

        if verdict == "supported":
            support_number = to_float(supporting_value)
            if support_number is None:
                verdict = "unsupported"
                entry["note"] = "claimed supported but cited no figure from the evidence"
                checked_by = "citation_check"
            elif not matches_any(support_number, evidence_pool):
                verdict = "unsupported"
                entry["note"] = (
                    f"claimed supported by '{supporting_value}', but that figure "
                    "does not appear in the evidence"
                )
                checked_by = "citation_check"

        claims.append(
            {
                "claim": entry.get("claim", ""),
                "value": entry.get("value"),
                "verdict": verdict,
                "supporting_value": supporting_value,
                "source": entry.get("source"),
                "note": entry.get("note", ""),
                "checked_by": checked_by,
            }
        )

    # Any number the numeric scan couldn't place and the critic never mentioned
    # gets flagged too — silence is not verification.
    mentioned = [to_float(c.get("value")) for c in claims]
    mentioned = [m for m in mentioned if m is not None]
    for number in dict.fromkeys(unmatched):
        if not matches_exact(number, mentioned):
            claims.append(
                {
                    "claim": f"{number:g}",
                    "value": f"{number:g}",
                    "verdict": "unsupported",
                    "supporting_value": None,
                    "source": None,
                    "note": "figure not found in evidence and not addressed by the critic",
                    "checked_by": "deterministic",
                }
            )

    annotated, unplaced = _annotate(draft, claims, strip_unverified)

    material = _material(claims)
    verified_count = sum(1 for c in material if c["verdict"] == "supported")
    unverified_count = len(material) - verified_count

    if unplaced and not strip_unverified:
        lines = "\n".join(
            f"- `{c.get('value')}` — {c.get('note') or c['verdict']}" for c in unplaced
        )
        annotated += f"\n\n> **{UNVERIFIED_MARKER}** — could not be traced to the evidence:\n{lines}"

    return {
        "verified_answer": annotated,
        "claims": claims,
        "unverified_count": unverified_count,
        "verified_count": verified_count,
        "confidence": _confidence(claims, had_evidence),
        "unplaced_claims": unplaced,
        "critic_parse_failed": False,
        "raw": raw,
    }

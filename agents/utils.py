"""Small shared helpers for the agents: JSON coaxing and numeric extraction."""

import json
import math
import re

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Matches 1234, 1,234.56, .5, 12% — with an optional leading sign.
# The lookbehind stops a hyphen between two numbers being read as a minus sign:
# "2026-07-27" must yield 2026, 7, 27 — not 2026, -7, -27 — or a date in the
# evidence will never match the same date written out in a draft.
_NUMBER = re.compile(r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?|(?<![\d.])\.\d+")

# Numbers that carry no analytic weight; flagging them is noise.
_TRIVIAL = {0.0, 1.0, 2.0, 3.0, 4.0, 100.0}


def extract_json(text: str, default=None):
    """Pull the first JSON object/array out of an LLM response.

    Models wrap JSON in prose or fences often enough that a bare json.loads is
    unreliable, so try progressively looser strategies.
    """
    if not text:
        return default

    candidates = []

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    candidates.append(text.strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return default


def extract_numbers(text: str) -> list[float]:
    """Every number appearing in a block of text, as floats."""
    values = []
    for match in _NUMBER.finditer(text or ""):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:
            continue
    return values


def to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[^\d.\-+]", "", value.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sig_figs(value: float) -> int:
    """How many significant digits a value was written to.

    Uses repr(), which gives the shortest string that round-trips. Fixed-point
    formatting would expose float representation error — 3996.42 renders as
    3996.420000000000073 and would be read as 19 significant digits.
    """
    number = abs(float(value))
    text = repr(number)
    if "e" in text or "E" in text:
        text = f"{number:.17g}"

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    digits = text.replace(".", "").lstrip("0")
    return max(1, len(digits))


def _round_sig(value: float, digits: int) -> float:
    if value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, -(exponent) + (digits - 1))


def rounds_to(value: float, candidate: float) -> bool:
    """Does `candidate` round to `value` at the precision `value` was written to?

    A flat relative tolerance mishandles rounded small numbers: writing
    0.0015649452 as "0.0016" is a 2.2% change and would fail a 1% check, even
    though it is exactly what an analyst should write. Comparing at the draft's
    own precision is the honest test.
    """
    try:
        return math.isclose(_round_sig(candidate, _sig_figs(value)), value, rel_tol=1e-9, abs_tol=1e-12)
    except (ValueError, OverflowError):
        return False


def matches_any(value: float, pool: list[float], rel_tol: float = 0.01) -> bool:
    """Is `value` present in `pool`, allowing for rounding and unit scaling?

    Tolerance is relative so that a draft saying "3,996.42" matches a stored
    3996.41845703125. Common presentation rescalings (percent<->fraction,
    thousands, crore/lakh) are also checked, since a synthesizer will often
    render 0.0234 as "2.34%".
    """
    if value is None:
        return False

    scalings = (1.0, 0.01, 100.0, 0.001, 1000.0, 1e5, 1e7)

    for candidate in pool:
        for scale in scalings:
            scaled = value * scale
            tolerance = max(abs(scaled), abs(candidate)) * rel_tol
            if abs(scaled - candidate) <= max(tolerance, 1e-9):
                return True
            # Also accept when the evidence rounds to the drafted figure at
            # the precision it was written to.
            if rounds_to(scaled, candidate):
                return True

    return False


def matches_exact(value: float | None, pool: list[float], rel_tol: float = 1e-6) -> bool:
    """Is `value` the same number as something in `pool`?

    Unlike matches_any, this does NOT try unit rescalings. Use it for identity
    questions ("is this the same figure?") — rescaling there is actively wrong:
    a fabricated 1400 would otherwise match a metric label of 14 via the 0.01
    scaling and escape being flagged.
    """
    if value is None:
        return False

    for candidate in pool:
        tolerance = max(abs(value), abs(candidate)) * rel_tol
        if abs(value - candidate) <= max(tolerance, 1e-9):
            return True

    return False


def significant_numbers(text: str) -> list[float]:
    """Numbers worth verifying — drops trivial small integers and years."""
    result = []
    for value in extract_numbers(text):
        if value in _TRIVIAL:
            continue
        if 1900 <= value <= 2100 and float(value).is_integer():
            continue  # a year, not a claim
        result.append(value)
    return result

"""Synthesizer Agent — drafts a structured research note from the evidence."""

from observability.tracing import llm_call

SYNTHESIZER_PROMPT = """You are a sell-side equity analyst writing a short research note.

QUESTION
{question}

QUANTITATIVE EVIDENCE (from the market database)
{sql_evidence}

QUALITATIVE EVIDENCE (from filings, presentations and news)
{doc_evidence}

Write the note in exactly this structure, using these headings:

## Thesis
Two or three sentences answering the question directly. Lead with the answer.

## Key Numbers
Bulleted. Every bullet must contain a figure that appears verbatim in the
quantitative evidence above. Reproduce figures at the precision they are given
(you may round to two decimals). Do not compute new derived figures.

## Risks
Two or three bullets. Ground them in the qualitative evidence where it exists;
otherwise state the risk in general terms without inventing specifics.

RULES — these matter more than style:
- Use ONLY the evidence above. Never introduce a number that is not in it.
- If the evidence does not support part of the question, say so plainly rather
  than filling the gap.
- Cite document passages inline as [DOC-n] where you rely on them.
- This is already-recorded historical market data (2022-2026). Treat every date
  as historical fact. Never refuse on the grounds of forecasting the future.
- No preamble, no sign-off. Start at "## Thesis"."""

NO_EVIDENCE_NOTE = "No quantitative evidence was gathered for this question."
NO_DOCS_NOTE = (
    "No document corpus is available for this question. "
    "Do not invent qualitative detail — say the narrative evidence is unavailable."
)


def synthesize(question: str, sql_evidence: str | None, doc_evidence: str | None) -> str:
    """Draft the research note. The Critic verifies it afterwards."""
    prompt = SYNTHESIZER_PROMPT.format(
        question=question,
        sql_evidence=sql_evidence or NO_EVIDENCE_NOTE,
        doc_evidence=doc_evidence or NO_DOCS_NOTE,
    )
    return llm_call(prompt, name="synthesizer", max_tokens=800)

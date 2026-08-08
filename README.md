# Financial Insight Copilot

Ask questions about NSE stock data in plain English. In v2 you get a short
research note where every number has been checked against its source before
it's shown to you.

Built as a final-year project. Stack: Groq (`llama-3.3-70b-versatile`),
LangGraph, ChromaDB, FastAPI, Streamlit, SQLite.

---

## About

The problem I wanted to solve is that an LLM sounds exactly the same whether a
number came from the data or it made the number up. "Reliance's revenue grew
14.8%" reads with the same confidence either way, and for anything financial
that's a real issue.

So this project answers questions from two sources — a SQLite database of daily
prices for six NSE stocks (2022–2026) and a set of company annual reports —
and then puts a separate agent in front of the answer whose only job is to
distrust it. That agent pulls out every number in the draft and refuses to pass
any figure it can't trace back to a specific database row or document passage.
Anything it can't verify gets marked `⚠️ unverified` rather than quietly
removed.

The interesting part isn't the RAG pipeline, it's whether that check actually
works. So I built an eval suite that feeds the Critic notes with deliberately
fabricated numbers in them and measures how many it catches — and, just as
importantly, how often it flags numbers that were fine all along.

---

## Results

Measured on 2026-08-05 with `python -m eval.run_eval`. Raw output in
[`eval/results.json`](eval/results.json).

| Metric | Result |
|---|---|
| Fabricated figures caught by the Critic | 9 / 9 (100%) |
| Legitimate figures wrongly flagged | 0 / 7 (0%) |
| Answer accuracy (22 questions) | 22 / 22 (100%) |
| Correct answers marked `high` confidence | 21 / 22 (95%) |

Without the Critic, all 9 fabricated numbers reach the user unflagged, because
the draft is what gets displayed. The false-positive column matters as much as
the catch rate: a verifier that flags everything would score 100% and be
useless.

The confidence figure started at 12/22 before I fixed the Critic's false
alarms (see below).

---

## v1 → v2

**v1** was a single-agent text-to-SQL tool:

```
Question → generate_sql → run_sql → explain_result → Answer
```

It worked, but nothing checked the final sentence against the data. The
explain step could round badly or add a number that was never in the result.

**v2** adds four more agents around it:

```
                 Question
                    │
                 Planner          picks which agents are needed
                ┌───┴───┐
                ▼       ▼
          Data Agent  Research Agent      (run in parallel)
          (v1 SQL)    (Chroma RAG)
                └───┬───┘
                    ▼
               Synthesizer        drafts the note
                    ▼
                 Critic           verifies every number
                    ▼
      Thesis + Key Numbers + Risks + confidence
```

- **Planner** — decides which evidence sources the question needs. Price-only
  questions skip retrieval.
- **Data Agent** — the v1 SQL pipeline, unchanged, wrapped as a tool.
- **Research Agent** — semantic search over company annual reports.
- **Synthesizer** — writes the structured note.
- **Critic** — the reason v2 exists.

The v1 functions were moved from `app.py` to [`data_agent.py`](data_agent.py)
without changing them. That was needed because `app.py` now serves the v2
endpoint, and having the graph import SQL functions back from `app.py` would
be a circular import. `/ask` behaves exactly as before.

---

## How the Critic works

Three stages, because a regex alone is too naive and an LLM alone can't be
trusted to check itself:

1. **Numeric sweep** — pull every number out of the draft and try to match it
   against numbers in the evidence, allowing for rounding. Unmatched numbers
   are suspects, not verdicts.
2. **LLM check** — the model reviews each claim, and to call one supported it
   has to name the exact figure from the evidence it used.
3. **Citation check** — that named figure is then verified against the evidence
   in code. An LLM claiming support while citing a number that isn't there is
   the main thing this catches, so the last gate isn't the LLM.

Unverified claims are flagged with `⚠️ unverified`, not deleted. Removing a
number silently seemed worse than showing it with a warning. Pass
`strip_unverified: true` to remove them instead.

**False alarms I had to fix.** Most of my debugging time went here:

- The evidence pool was being read from the rendered evidence text, which
  includes the SQL query. Literals like `rsi_14` and `'2025'` were counting as
  facts. Now it's built from the result values only.
- Metric labels were flagged as fabricated — the `14` in "14-day RSI", the `30`
  in "30-day moving average". They never appear in a result row.
- After adding the document corpus, citation markers (`[DOC-1]`) and dates
  ("on July 27, 2026") got flagged too.
- Flags were inserted inline and split things apart, e.g.
  `July 27 ⚠️ unverified, 2026`. They go at end of line now.
- A flat 1% tolerance rejected correct rounding: `0.0015649` written as
  `0.0016` is a 2.2% difference. Matching now compares at the precision the
  number was written to.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq key. Langfuse keys are
optional — tracing turns itself off without them.

The database is already built. To rebuild:

```bash
python data_pipeline.py && python features.py
```

### Document corpus

The Research Agent needs PDFs in `ingestion/corpus/<COMPANY>/`. I used the
FY2024-25 annual reports for Reliance (146 pages) and Sun Pharma (326 pages),
which come to 2,863 chunks. They aren't committed, so download them from the
companies' investor-relations pages and run:

```bash
python -m ingestion.ingest
```

TCS is missing because `tcs.com` blocks automated downloads with a 403. It
needs a manual download from a browser.

Without a corpus the app still runs — the Research Agent reports it as
unavailable and the note says so instead of making things up.

---

## Running

```bash
python app.py
```

```bash
streamlit run streamlit_app.py
```

The sidebar switches between **Quick Answer (v1)** and **Research Note (v2)**.
The v2 view shows the note, a confidence indicator, and a claim-by-claim
breakdown of what was verified.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | v1 text-to-SQL |
| `POST /research` | v2 multi-agent research note |
| `GET /health` | Corpus and tracing status, no LLM calls |

**Deploying to Streamlit Cloud:** leave `COPILOT_API_URL` unset and the agents
run inside the Streamlit process, so no separate backend is needed. Put the
keys in Settings → Secrets. Note the deployed app has no corpus, since the
index isn't committed — research questions fall back to SQL-only there.

---

## Evaluation

```bash
python -m eval.run_eval                    # both suites, ~250s, ~100k tokens
python -m eval.run_eval --suite injection  # critic only, ~10k tokens
python -m eval.test_retrieval              # retrieval only, free
```

**Factual suite** — 22 questions where the ground truth is stored as SQL rather
than as a fixed number, so it's recomputed from the database each run and can't
go stale.

**Injection suite** — fixed draft notes with deliberately fake figures, plus
clean controls to measure false positives. Drafts are fixed so the Critic is
tested on its own, without variation from the Synthesizer.

Retrieval scores 3/4 unfiltered and 4/4 when filtered by ticker. The one miss
is a query about Reliance's green hydrogen spending that returns Sun Pharma's
energy-conservation annexure instead — to the embedding model, a pharma
company's solar panels and an energy company's capex look very similar. The
graph filters by ticker, which avoids it.

---

## Known limitations

- Groq's free tier is 100k tokens/day and one full eval run uses most of it.
- The v1 SQL functions call Groq directly, so they skip the retry and tracing
  layers. Traced token counts are lower than actual usage.
- Langfuse shows cost as 0 — it has no pricing data for this Groq model.
- Confidence is a ratio of verified to total claims. Groq isn't fully
  deterministic even at temperature 0, so borderline notes can vary between
  runs. It's a hint to check the claim breakdown, not a score.
- `pypdf` can't OCR, so scanned PDFs ingest as almost nothing.
- Numeric matching allows unit rescaling (percent/fraction, lakh/crore), so a
  fake number that happens to be exactly 100× a real one would slip through.
- No Dockerfile — this runs locally.

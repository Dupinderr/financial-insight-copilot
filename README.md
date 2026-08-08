# Financial Insight Copilot

Ask questions about NSE stock data in plain English — and, in v2, get a
structured research note where **every number has been checked against its
source before you see it**.

---

## The headline result

The v2 Critic agent was measured on a suite of research notes containing
deliberately fabricated figures:

| Metric | Critic OFF | Critic ON |
|---|---|---|
| **Fabricated figures reaching the user unflagged** | 9 / 9 — **100%** | 0 / 9 — **0%** |
| **Legitimate figures wrongly flagged** | — | 0 / 7 — **0%** |
| **Answer accuracy** (22 questions, ground truth from the database) | 22 / 22 — **100%** | 22 / 22 — **100%** |

The critic-off baseline is 100% by construction: with no verifier in the loop,
the synthesizer's draft *is* what the user sees, so every fabricated number
reaches them unchallenged. The point of the number is the contrast — the same
drafts, run through the Critic, surfaced every invented figure without raising a
single false alarm on a real one.

Reproduce with `python -m eval.run_eval`. Full output in
[`eval/results.json`](eval/results.json).

Both suites were last measured on 2026-08-05, after every Critic fix described
below. Full per-case output in [`eval/results.json`](eval/results.json), which
records a `measured_at` per suite.

### Confidence calibration

A verifier that cries wolf is as useless as one that misses. Across the 22
factual questions — all of which the pipeline answers correctly — the share
reporting `high` confidence:

| | before the Critic fixes | after |
|---|---|---|
| Correct answers marked `high` confidence | 12 / 22 (55%) | **21 / 22 (95%)** |

The 10 spurious downgrades were all false alarms of the same few kinds: metric
labels, citation markers and date components being verified as though they were
factual claims. Each is described under [How the Critic works](#how-the-critic-works).

---

## v1 → v2: what changed and why

### What v1 was

A single-agent text-to-SQL tool. You asked a question, an LLM wrote SQL against
a SQLite table of NSE daily data, the SQL ran, and a second LLM call turned the
result into a sentence.

```
Question ──> generate_sql ──> run_sql ──> explain_result ──> Answer
```

It worked. It also had one structural weakness: **nothing checked the final
sentence against the data**. The explain step was free to round badly, mix up a
column, or add a plausible-sounding figure that was never in the result set —
and nothing in the pipeline would notice.

### What v2 adds

```
                    User Question
                          │
                          ▼
                   ┌─────────────┐
                   │   Planner   │  which sub-agents does this need?
                   └──────┬──────┘
                  ┌───────┴────────┐
                  ▼                ▼
          ┌───────────────┐  ┌──────────────┐
          │  Data Agent   │  │   Research   │   (run in parallel)
          │  (v1 SQL —    │  │    Agent     │
          │   unchanged)  │  │  Chroma RAG  │
          └───────┬───────┘  └──────┬───────┘
                  └───────┬─────────┘
                          ▼
                   ┌─────────────┐
                   │ Synthesizer │  drafts the research note
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │   Critic    │  verifies every numeric claim
                   └──────┬──────┘
                          ▼
            Thesis + Key Numbers + Risks + confidence
```

| Added | Why |
|---|---|
| **Planner agent** | Not every question needs both evidence sources. Pure price questions skip retrieval entirely. |
| **Research agent** (Chroma + `all-MiniLM-L6-v2`) | The database has prices but no *narrative*. Filings and news explain the numbers. |
| **Synthesizer agent** | Turns two heterogeneous evidence streams into one structured note. |
| **Critic agent** | The reason v2 exists. Nothing reaches the user without being traced to a source. |
| **LangGraph** | Conditional routing and parallel fan-out, with the v1 pipeline as one node. |
| **Langfuse** | Per-node latency, token cost and I/O for a 5-call pipeline that is otherwise a black box. |
| **Eval harness** | A claim like "the critic catches hallucinations" is worthless unmeasured. |

**The v1 pipeline was not rewritten.** Its four functions were moved verbatim
from `app.py` into [`data_agent.py`](data_agent.py) — same prompts, same retry
logic, same schema description — and wrapped in a single
`query_financial_data()` call so the graph can invoke them as a tool. The move
was necessary because `app.py` now serves the v2 endpoint too, and importing the
graph from `app.py` while the graph imported SQL functions *back* from `app.py`
would be circular. `/ask` calls exactly the functions it always did.

---

## How the Critic works

Neither a regex nor an LLM is trustworthy alone here, so verification runs in
three stages and a claim must survive all of them:

1. **Deterministic sweep.** Every significant number in the draft is matched
   against the numbers in the evidence, allowing for rounding and the usual
   presentation rescalings (`0.0234` → `"2.34%"`). Unmatched numbers become
   *candidates* — not verdicts, since a legitimately derived figure won't appear
   verbatim either.

2. **LLM adjudication.** The model lists each numeric claim and, to call one
   supported, **must name the exact figure from the evidence it relies on**.

3. **Citation check.** That named figure is then verified against the evidence
   itself, deterministically. An LLM asserting support while citing a number
   that isn't there is precisely the failure this agent exists to catch, so the
   final gate does not trust the adjudication.

Four details that mattered more than expected:

- **The evidence pool is built from result *values*, not the rendered evidence
  text.** That text embeds the SQL query, and literals inside it (`rsi_14`,
  `'2025'`, `LIMIT 1`) would otherwise count as facts and could vouch for a
  fabricated figure.

- **Metric labels are not claims.** The `14` in "14-day RSI", the `30` in
  "30-day moving average", a threshold like "RSI above 70" restated from the
  question — these never appear in a result row, so a naive verifier flags them
  as fabricated and buries the real findings in noise. They are recognised as
  *contextual* and ignored: neither evidence nor claim.

- **Citations and dates are not claims either.** Once the document corpus went
  in, the Synthesizer began citing `[DOC-1]` and writing "on July 27, 2026" —
  and the verifier dutifully flagged the `1` and the `27` as fabricated
  figures. Both are stripped before analysis. Dates are subtracted carefully:
  a number that also appears *outside* a date is still a real claim, so
  "on July 27 there were 27 outlier days" keeps the second `27` checkable.

- **Flags are placed at end of line, not inline.** Appending the marker
  directly after the offending phrase split whatever the phrase happened to end
  inside, producing `July 27 ⚠️ unverified, 2026` and `[DOC-1 ⚠️ unverified]`.
  Marking the line keeps the note readable and the citations intact.

- **Rounding is checked at the draft's own precision.** A flat 1% tolerance
  wrongly rejects small rounded numbers: writing `0.0015649452` as `0.0016` is
  a 2.2% change, and the Critic was flagging its own correctly-rounded figures.
  Matching now also asks "does the evidence round to this, at the number of
  significant digits it was written to?" Significant digits come from `repr()`,
  not fixed-point formatting — the latter surfaces float representation error
  and reads `3996.42` as 19 significant digits rather than 6.

Unverified claims are **flagged, not deleted** — the note keeps the number and
appends `⚠️ unverified`. Silently removing a figure is worse than showing a
questionable one next to a warning. Set `strip_unverified: true` to override.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```
GROQ_API_KEY=your_groq_key

# Optional — tracing is a no-op without these
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

The database (`financial_data.db`) is already built. To rebuild from scratch:

```bash
python data_pipeline.py && python features.py
```

### The research corpus

Currently indexed — **2,863 chunks** from two FY2024-25 integrated annual
reports, downloaded from the companies' own investor-relations sites:

| Company | Document | Pages | Chunks |
|---|---|---|---|
| Reliance | `reliance_annual_report_2024-25.pdf` | 146 | 1,322 |
| Sun Pharma | `sunpharma_annual_report_2024-25.pdf` | 326 | 1,541 |

**TCS is missing.** `tcs.com` sits behind Akamai bot protection that returns
`403` to every automated request — three header profiles, three document URLs,
a headless browser and a server-side fetcher all refused. It needs a manual
download from a normal browser session:
`tcs.com` → Investor Relations → Financial Statements → Annual Report, saved
into `ingestion/corpus/TCS/`.

To add documents (yours or TCS's), drop them in the matching folder and re-run:

```bash
python -m ingestion.ingest
```

Ingest upserts on a stable chunk id, so re-running won't duplicate existing
chunks. Embedding 2,863 chunks takes ~100s on CPU. See
[`ingestion/corpus/README.md`](ingestion/corpus/README.md) for filename
conventions and what else is worth adding.

With no corpus at all the graph still runs end-to-end: the Research Agent
reports it unavailable and the Synthesizer says so rather than inventing
narrative detail.

---

## Running it

```bash
python app.py
```

```bash
streamlit run streamlit_app.py
```

The Streamlit sidebar switches between **Quick Answer (v1)** — the original
single-agent mode, unchanged — and **Research Note (v2)**, which shows the
verified note, a confidence indicator, and a claim-by-claim verification
breakdown.

### Endpoints

| Endpoint | What it does |
|---|---|
| `POST /ask` | v1 text-to-SQL. Unchanged. |
| `POST /research` | v2 multi-agent verified research note. |
| `GET /health` | Reports whether the corpus and tracing are actually live. |

`/health` is the cheapest way to confirm setup — it makes no LLM calls:

```json
{"status":"ok",
 "corpus":{"available":false,"count":0,"reason":"collection 'financial_filings' not built yet"},
 "tracing":{"enabled":true,"reason":null,"host":"https://cloud.langfuse.com"}}
```

If `tracing.enabled` is false, `reason` distinguishes missing keys from a
rejected `auth_check` — the latter almost always means `LANGFUSE_HOST` points at
the wrong region (`cloud.langfuse.com` for EU, `us.cloud.langfuse.com` for US).

```bash
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question": "How volatile has TCS been, and what is driving it?"}'
```

---

## Deploying to Streamlit Community Cloud

The app runs in one of two topologies, chosen by a single environment variable:

| `COPILOT_API_URL` | Behaviour |
|---|---|
| unset (default) | Agents run **inside the Streamlit process**. One deploy, no backend. |
| set to an API URL | Streamlit calls that FastAPI instance over HTTP. |

Community Cloud runs only your Streamlit script — there is no second process —
so the default in-process mode is what it needs. Locally you can keep using the
two-terminal setup by exporting `COPILOT_API_URL=http://localhost:8000`, or just
run Streamlit alone and skip `app.py` entirely.

**Steps**

1. Push to GitHub. Confirm `.env` is **not** in the commit — it is gitignored,
   but check `git status` before the first push.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the
   repo with `streamlit_app.py` as the entry point.
3. Under **Settings → Secrets**, paste your credentials in TOML form:

   ```toml
   GROQ_API_KEY = "gsk_..."
   LANGFUSE_PUBLIC_KEY = "pk-lf-..."
   LANGFUSE_SECRET_KEY = "sk-lf-..."
   LANGFUSE_HOST = "https://cloud.langfuse.com"
   ```

   `streamlit_app.py` copies these into the environment at startup, before any
   module that reads them is imported.
4. Open the app and expand **System status** in the sidebar. It reports
   execution mode, whether the Groq key resolved, corpus size and tracing
   state — the hosted equivalent of `GET /health`.

**What ships and what doesn't.** `financial_data.db` is committed, so the SQL
side works immediately. The corpus PDFs (26 MB), the Chroma index (27 MB) and
the ONNX model (166 MB) are all gitignored — the repo deliberately carries no
text extracted from the source annual reports.

Two consequences for a hosted deploy:

- **There is no document corpus.** The Research Agent reports itself
  unavailable and research questions fall back to SQL-only evidence. The
  Synthesizer states the gap rather than inventing narrative. To demo
  retrieval, either run locally after `python -m ingestion.ingest`, or commit
  `chroma_db/` (drop it from `.gitignore`) and redeploy.
- **The first query downloads ~79 MB** of embedding model; later ones are warm.

## Evaluation

```bash
python -m eval.run_eval                    # both suites  (~250s, ~100k tokens)
python -m eval.run_eval --suite injection  # critic only  (~6s, ~10k tokens)
python -m eval.test_retrieval              # retrieval only, no LLM calls, free
```

**Factual suite** — 22 questions whose ground truth is stored as *SQL*, not as
literal values, so the expected answer is recomputed from the database at eval
time and can never drift out of sync with it.

**Injection suite** — fixed draft notes carrying deliberately fabricated
figures, paired with clean controls. Drafts are fixed rather than generated so
the Critic is measured in isolation, without synthesizer variance. The controls
are the important half: a verifier that flags everything would score a perfect
catch rate and be useless.

`eval/test_retrieval.py` grades retrieval independently, using only the local
embedding model — it costs nothing and works when the Groq quota is exhausted.
On the current corpus:

```
Unfiltered top-1 company accuracy: 3/4 = 75%    raw embedding discrimination
Ticker-filtered relevant results:  4/4 = 100%   the path the graph takes
```

The gap is the interesting part. The one unfiltered miss is a query about
Reliance's *green hydrogen capex* that retrieves Sun Pharma's statutory
**"Particulars of Energy Conservation"** annexure instead — solar rooftops at
Mohali, at similarity 0.517 against Reliance's own 0.467. To a 384-dimensional
sentence embedding, a pharma company's energy-efficiency disclosure and an
energy conglomerate's capex narrative are near-indistinguishable; the model has
no notion of *which company* a passage belongs to.

The graph sidesteps this by filtering on ticker whenever the planner names one,
which is why the filtered column is 100%. Both numbers are reported because the
unfiltered figure is the honest measure of the embeddings, and the filtered one
is what the pipeline actually relies on.

Running one suite **merges** into `eval/results.json` rather than replacing it,
so a cheap `--suite injection` pass doesn't discard the last factual
measurement. Each suite records its own `measured_at`, and the summary line
marks any figure carried over from an earlier run.

---

## Known limitations

- **Groq free tier is 100k tokens/day.** A full eval run consumes most of it.
  The daily-quota case is detected and reported with an actionable message
  rather than retried pointlessly; per-minute limits are retried with backoff.

- **The v1 SQL functions bypass the v2 retry and tracing layers.** They call
  Groq directly, exactly as they did in v1. This was deliberate — the spec was
  to wrap them, not rewrite them — but it means SQL generation has no
  rate-limit backoff, and its tokens are missing from the trace. A verified
  run shows this precisely: the `data_agent` TOOL span records its 0.575s
  latency and output, but carries no GENERATION child, while the planner,
  synthesizer and critic each report input/output token counts. **Traced token
  totals therefore understate real usage** by the SQL-generation call.

- **Langfuse reports cost as 0.** It has no built-in pricing for Groq's
  `llama-3.3-70b-versatile`, so token counts are captured but not costed. Add a
  model price in the Langfuse project settings if you need spend tracking.

- **Confidence calibration is the softest part.** `high`/`medium`/`low` is a
  ratio of verified to material claims. It is stable when the note is clean or
  clearly fabricated, but Groq is not bit-deterministic even at temperature 0,
  and borderline notes can land differently across runs. Treat it as a
  prompt to check the claim breakdown, not as a score.

- **`pypdf` extracts embedded text but cannot OCR.** Scanned annual reports
  will ingest as near-zero chunks. Prefer the digitally-published PDFs.

- **Numeric matching tolerates rescaling** (percent↔fraction, thousands,
  lakh/crore) to avoid false alarms on reformatted figures. A fabricated number
  that happens to be exactly 100× a real one would therefore pass.

- **No Dockerfile**, by choice — this runs locally.

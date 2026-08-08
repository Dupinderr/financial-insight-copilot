"""Planner Agent — decides which sub-agents a question needs."""

from ingestion.config import CORPUS_TICKERS
from observability.tracing import llm_call

from .utils import extract_json

KNOWN_TICKERS = sorted(set(CORPUS_TICKERS.values()))

PLANNER_PROMPT = """You are the planner for a financial research system. Decide which sub-agents are needed to answer the user's question.

Two sub-agents are available:

1. DATA AGENT — queries a SQLite table of daily NSE stock market data (2022-01-03 to 2026-07-27) for six tickers:
   RELIANCE.NS, TCS.NS, HDFCBANK.NS, SUNPHARMA.NS, MARUTI.NS, ITC.NS.
   Columns: Date, ticker, sector, Open, High, Low, Close, Volume, daily_return,
   is_outlier, ma_7, ma_30, volatility_30d, rsi_14.
   Use it for anything about prices, returns, volatility, momentum, volume or RSI.

2. RESEARCH AGENT — semantic search over company filings, investor presentations and
   news articles for: {corpus_tickers}.
   Use it for anything about strategy, business segments, management commentary,
   risks, capex, competitive position or qualitative narrative.

Question: {question}

Reply with ONLY a JSON object, no prose:
{{
  "use_data": true or false,
  "use_research": true or false,
  "data_question": "a self-contained question for the Data Agent, or null",
  "research_query": "a search query for the Research Agent, or null",
  "tickers": ["TICKER.NS", ...],
  "reasoning": "one sentence"
}}

Rules:
- Turn on BOTH agents when the question mixes numbers with narrative
  (e.g. "is X a good investment", "explain X's performance").
- "tickers" lists only tickers explicitly relevant to the question; use [] if none.
- Always turn on at least one agent."""


def plan(question: str) -> dict:
    """Route a question to the sub-agents that can answer it."""
    prompt = PLANNER_PROMPT.format(
        question=question,
        corpus_tickers=", ".join(sorted(CORPUS_TICKERS.keys())),
    )

    raw = llm_call(prompt, name="planner", max_tokens=400)
    parsed = extract_json(raw, default=None)

    if not isinstance(parsed, dict):
        # If the planner's output is unusable, run both agents — over-fetching
        # is far cheaper than silently answering with half the evidence.
        return {
            "use_data": True,
            "use_research": True,
            "data_question": question,
            "research_query": question,
            "tickers": [],
            "reasoning": "planner output unparseable; defaulting to both agents",
            "raw": raw,
        }

    use_data = bool(parsed.get("use_data", True))
    use_research = bool(parsed.get("use_research", True))

    if not use_data and not use_research:
        use_data = True

    tickers = parsed.get("tickers") or []
    if not isinstance(tickers, list):
        tickers = []
    tickers = [t for t in tickers if isinstance(t, str)]

    return {
        "use_data": use_data,
        "use_research": use_research,
        "data_question": parsed.get("data_question") or question,
        "research_query": parsed.get("research_query") or question,
        "tickers": tickers,
        "reasoning": parsed.get("reasoning", ""),
        "raw": raw,
    }

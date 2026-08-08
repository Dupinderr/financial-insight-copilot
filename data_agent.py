"""Data Agent — the v1 text-to-SQL pipeline, lifted out of app.py unchanged.

This module exists so that both app.py (the original /ask endpoint) and the v2
LangGraph can import the same text-to-SQL functions without a circular import.
The SQL logic below is a verbatim move from v1: prompts, retry behaviour and
schema description are untouched.

The only addition is query_financial_data() at the bottom — a thin interface
wrapper so the graph can call this as a single tool.
"""

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import create_engine, text

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Anchored to this file rather than the process CWD, so the eval harness and
# the graph can run from subdirectories without losing the database.
DB_PATH = Path(__file__).resolve().parent / "financial_data.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

MODEL = "llama-3.3-70b-versatile"

SCHEMA_DESCRIPTION = """
Table: stocks_with_features
Columns:
- Date (date): trading date. Data spans 2022-01-01 to 2026-07-27, one row per ticker per trading day.
- ticker (text): stock symbol, e.g. 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS', 'MARUTI.NS', 'ITC.NS'
- sector (text): one of 'Energy', 'IT', 'Banking', 'Pharma', 'Auto', 'FMCG'
- Open, High, Low, Close (float): daily price in INR
- Volume (integer): shares traded
- daily_return (float): day-over-day % change in Close
- is_outlier (boolean): flagged unusual daily_return
- ma_7, ma_30 (float): 7-day and 30-day moving average of Close
- volatility_30d (float): 30-day rolling std of daily_return
- rsi_14 (float): 14-day Relative Strength Index (0-100; >70 overbought, <30 oversold)

IMPORTANT RULES:
- All data is historical, already-recorded market data — never treat any date in it as "the future."
- For questions about "current" or "latest" status (e.g. current RSI, current price), always add
  `ORDER BY Date DESC LIMIT 1` filtered to the relevant ticker, to get only the most recent row.
- When using aggregate functions (COUNT, SUM, AVG) combined with ORDER BY on the aggregate,
  you MUST include a GROUP BY clause. Example:
  SELECT ticker, COUNT(*) as cnt FROM stocks_with_features WHERE is_outlier = 1
  GROUP BY ticker ORDER BY cnt DESC LIMIT 1
- Always alias computed/CASE columns with a clear name using AS, e.g. `AS status`, `AS avg_price`.
- When a query is meant to identify a single "top" item (most/highest/least), always SELECT
  both the identifying column AND the relevant metric (e.g. SELECT ticker, COUNT(*) as outlier_count),
  not just the identifying column alone.
"""


def _clean_sql(raw_sql: str) -> str:
    raw_sql = re.sub(r"^```sql\s*|```$", "", raw_sql, flags=re.MULTILINE).strip()
    return raw_sql


def generate_sql(question: str, previous_error: str = None, previous_sql: str = None) -> str:
    if previous_error:
        prompt = f"""You are a SQL expert. Your previous SQLite query failed. Fix it.

{SCHEMA_DESCRIPTION}

Question: {question}
Previous (failed) SQL: {previous_sql}
Error: {previous_error}

Return ONLY the corrected SQL query, no explanation, no markdown formatting.

SQL:"""
    else:
        prompt = f"""You are a SQL expert. Given this table schema, write a single SQLite query that answers the question.
Return ONLY the SQL query, no explanation, no markdown formatting.

{SCHEMA_DESCRIPTION}

Question: {question}

SQL:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
    )
    return _clean_sql(response.choices[0].message.content.strip())


def run_sql(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
    return pd.DataFrame(rows, columns=columns)


def generate_sql_with_retry(question: str, max_retries: int = 1):
    sql = generate_sql(question)
    for attempt in range(max_retries + 1):
        try:
            result_df = run_sql(sql)
            return sql, result_df, None
        except Exception as e:
            if attempt < max_retries:
                sql = generate_sql(question, previous_error=str(e), previous_sql=sql)
            else:
                return sql, None, str(e)


def explain_result(question: str, result_df: pd.DataFrame) -> str:
    result_str = result_df.to_string(index=False) if not result_df.empty else "No rows returned."

    prompt = f"""This is a SQL query result from a database of REAL, ALREADY-RECORDED historical stock market data (2022-2026). None of these dates are in the future — treat every date as historical fact.

Question: {question}
SQL result:
{result_str}

Give a short, clear, one-to-two sentence natural language answer based ONLY on the exact values shown in the SQL result above. Do not contradict the data. Do not refuse or mention future predictions — this is historical data analysis. The SQL result is complete and sufficient to answer the question — never ask for more data or say information is missing."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# --- v2 tool interface -------------------------------------------------------

MAX_TOOL_ROWS = 50


def query_financial_data(question: str, explain: bool = False) -> dict:
    """Run the v1 text-to-SQL pipeline and return a structured result.

    This is the Data Agent's tool interface for the LangGraph. It changes no
    SQL logic — it only packages the existing functions into one dict so the
    Synthesizer and Critic have a uniform shape to read from.

    Rows are capped at MAX_TOOL_ROWS for the prompt's benefit; row_count always
    reports the true size of the result set.
    """
    sql, result_df, error = generate_sql_with_retry(question)

    if error:
        return {
            "ok": False,
            "source": "sql",
            "question": question,
            "sql": sql,
            "rows": [],
            "row_count": 0,
            "answer": None,
            "error": error,
        }

    rows = result_df.head(MAX_TOOL_ROWS).to_dict(orient="records")

    return {
        "ok": True,
        "source": "sql",
        "question": question,
        "sql": sql,
        "rows": rows,
        "row_count": int(len(result_df)),
        "truncated": bool(len(result_df) > MAX_TOOL_ROWS),
        "answer": explain_result(question, result_df) if explain else None,
        "error": None,
    }


def ask_v1(question: str) -> dict:
    """The v1 answer, in the exact shape /ask has always returned.

    Lives here so the FastAPI route and the Streamlit app share one code path
    instead of duplicating it — the deployed app has no backend to call.
    """
    sql, result_df, error = generate_sql_with_retry(question)

    if error:
        return {"question": question, "generated_sql": sql, "error": error}

    return {
        "question": question,
        "generated_sql": sql,
        "result_preview": result_df.head(10).to_dict(orient="records"),
        "answer": explain_result(question, result_df),
    }


def format_sql_evidence(result: dict) -> str:
    """Render a query_financial_data() result as evidence text for a prompt."""
    if not result.get("ok"):
        return f"[SQL query failed: {result.get('error')}]"
    if not result["rows"]:
        return f"SQL: {result['sql']}\nResult: no rows returned."

    header = list(result["rows"][0].keys())
    lines = [" | ".join(header)]
    for row in result["rows"]:
        lines.append(" | ".join(str(row[c]) for c in header))

    note = f" (showing first {MAX_TOOL_ROWS} of {result['row_count']})" if result.get("truncated") else ""
    return f"SQL: {result['sql']}\nResult{note}:\n" + "\n".join(lines)

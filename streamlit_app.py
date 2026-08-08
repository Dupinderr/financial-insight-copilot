import os

import requests
import streamlit as st

st.set_page_config(page_title="Financial Insight Copilot", layout="centered")


def _load_config_into_env():
    """Populate the environment before anything reads it.

    Both must happen up front: data_agent builds its Groq client at import
    time from os.getenv, and the status panel below reports on these values.
    Loading .env lazily (as a side effect of the first import that happens to
    call load_dotenv) made the panel claim the Groq key was missing while
    Langfuse read as configured.

    Hosted deploys supply Streamlit secrets; locally there is no secrets.toml
    and .env is the source, so each is best-effort.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and not os.environ.get(key):
                os.environ[key] = value
    except Exception:
        pass


_load_config_into_env()

# Set COPILOT_API_URL to talk to a running FastAPI instance; leave it unset and
# the agents run inside this process. Streamlit Community Cloud runs only this
# script — there is no backend on localhost there — so in-process is the
# default rather than an afterthought.
API_BASE = (os.getenv("COPILOT_API_URL") or "").rstrip("/")
USE_API = bool(API_BASE)
ASK_URL = f"{API_BASE}/ask"
RESEARCH_URL = f"{API_BASE}/research"


def run_v1(question: str) -> dict:
    if USE_API:
        return requests.post(ASK_URL, json={"question": question}, timeout=60).json()
    from data_agent import ask_v1
    return ask_v1(question)


def run_v2(question: str, strip_unverified: bool) -> dict:
    if USE_API:
        return requests.post(
            RESEARCH_URL,
            json={"question": question, "strip_unverified": strip_unverified},
            timeout=300,
        ).json()
    from graph.research_graph import run_research
    return run_research(question, strip_unverified=strip_unverified)

CONFIDENCE_STYLE = {
    "high": ("🟢", "success", "Every numeric claim traced to a source."),
    "medium": ("🟡", "warning", "Most claims traced; some could not be verified."),
    "low": ("🔴", "error", "Several claims could not be traced to any source."),
}

st.title("📊 Financial Insight Copilot")

mode = st.sidebar.radio(
    "Mode",
    ["Quick Answer (v1)", "Research Note (v2)"],
    help=(
        "Quick Answer runs the original single-agent text-to-SQL pipeline.\n\n"
        "Research Note runs the multi-agent graph: planner → data + research "
        "agents → synthesizer → critic."
    ),
)

if "history" not in st.session_state:
    st.session_state.history = []
if "research_history" not in st.session_state:
    st.session_state.research_history = []


def render_status():
    """Self-diagnosis panel — there is no /health to curl on a hosted deploy."""
    with st.sidebar.expander("System status", expanded=False):
        st.caption(f"**Execution:** {'via API — ' + API_BASE if USE_API else 'in-process'}")

        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set — queries will fail.")
        else:
            st.caption("**Groq key:** set")

        if USE_API:
            return  # the rest lives in the API process, not this one

        try:
            from ingestion.retriever import corpus_status
            corpus = corpus_status()
            if corpus["available"]:
                st.caption(f"**Corpus:** {corpus['count']} chunks")
            else:
                st.warning(f"Corpus unavailable — {corpus['reason']}. "
                           "Research questions will fall back to SQL only.")
        except Exception as e:
            st.warning(f"Corpus check failed: {e}")

        try:
            from observability.tracing import tracing_status
            tracing = tracing_status()
            st.caption(
                f"**Langfuse:** {'on — ' + tracing['host'] if tracing['enabled'] else 'off'}"
            )
            if not tracing["enabled"] and tracing["reason"]:
                st.caption(f"_{tracing['reason']}_")
        except Exception as e:
            st.caption(f"**Langfuse:** check failed — {e}")


render_status()


# --- v1: original single-agent mode, unchanged -------------------------------

def render_quick_answer():
    st.caption("Ask questions about NSE stock data in plain English.")

    question = st.text_input(
        "Ask a question",
        placeholder="Which sector had the highest average volatility?",
    )

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                st.session_state.history.append(run_v1(question))
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach the backend at {API_BASE}. Is `app.py` running?")
            except Exception as e:
                st.error(f"Query failed: {e}")

    for entry in reversed(st.session_state.history):
        st.markdown(f"**Q: {entry['question']}**")

        if "error" in entry:
            st.error(f"SQL execution failed: {entry['error']}")
            with st.expander("Attempted SQL"):
                st.code(entry["generated_sql"], language="sql")
        else:
            st.write(entry["answer"])
            with st.expander("Generated SQL + raw result"):
                st.code(entry["generated_sql"], language="sql")
                if entry["result_preview"]:
                    st.dataframe(entry["result_preview"])
                else:
                    st.write("No rows returned.")
        st.divider()


# --- v2: multi-agent verified research note ----------------------------------

def render_confidence(entry):
    confidence = entry.get("confidence", "low")
    icon, level, blurb = CONFIDENCE_STYLE.get(confidence, CONFIDENCE_STYLE["low"])
    verification = entry.get("verification", {})

    getattr(st, level)(
        f"{icon} **Confidence: {confidence.upper()}** — {blurb}  \n"
        f"{verification.get('verified_count', 0)} claims verified, "
        f"{verification.get('unverified_count', 0)} unverified."
    )


def render_research_note():
    st.caption(
        "Multi-agent research: a planner routes the question, a data agent and a "
        "document agent gather evidence, and a critic verifies every number before you see it."
    )

    question = st.text_input(
        "Research question",
        placeholder="How has Reliance performed recently, and what is driving it?",
    )

    strip_unverified = st.sidebar.checkbox(
        "Strip unverified claims",
        value=False,
        help="Off (default): unverified numbers stay visible with a ⚠️ marker. "
             "On: they are removed from the note entirely.",
    )

    if st.button("Run research") and question:
        with st.spinner("Planning → gathering evidence → drafting → verifying..."):
            try:
                st.session_state.research_history.append(run_v2(question, strip_unverified))
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach the backend at {API_BASE}. Is `app.py` running?")
            except requests.exceptions.Timeout:
                st.error("The research run timed out.")
            except Exception as e:
                st.error(f"Research run failed: {e}")

    for entry in reversed(st.session_state.research_history):
        st.markdown(f"### {entry['question']}")

        if entry.get("error"):
            st.error(entry["error"])
            st.divider()
            continue

        render_confidence(entry)
        st.markdown(entry["answer"])

        plan = entry.get("plan", {})
        agents_used = []
        if plan.get("use_data"):
            agents_used.append("Data Agent")
        if plan.get("use_research"):
            agents_used.append("Research Agent")

        st.caption(
            f"Agents: {', '.join(agents_used) or 'none'} · "
            f"{entry.get('elapsed_s', '?')}s"
            + (f" · [Langfuse trace]({entry['trace_url']})" if entry.get("trace_url") else "")
        )

        with st.expander("Claim-by-claim verification"):
            claims = entry.get("verification", {}).get("claims", [])
            if not claims:
                st.write("No numeric claims were made.")
            for claim in claims:
                verdict = claim.get("verdict")
                icon = {"supported": "✅", "contextual": "➖"}.get(verdict, "⚠️")
                st.markdown(
                    f"{icon} **{verdict}** — `{claim.get('value')}`  \n"
                    f"{claim.get('claim', '')}  \n"
                    f"_{claim.get('note', '')}_"
                )

        sql_info = entry.get("sql")
        if sql_info and sql_info.get("query"):
            with st.expander("SQL evidence"):
                st.code(sql_info["query"], language="sql")
                if sql_info.get("rows"):
                    st.dataframe(sql_info["rows"])
                st.caption(f"{sql_info.get('row_count', 0)} rows returned")

        docs = entry.get("documents")
        if docs:
            with st.expander("Document evidence"):
                if docs.get("error"):
                    st.info(docs["error"])
                elif not docs.get("chunks"):
                    st.write("No relevant passages retrieved.")
                else:
                    st.caption(f"Sources: {', '.join(docs.get('sources', []))}")
                    for chunk in docs["chunks"]:
                        page = f", p.{chunk['page']}" if chunk.get("page", -1) != -1 else ""
                        st.markdown(
                            f"**{chunk.get('source')}{page}** "
                            f"(similarity {chunk.get('similarity')})"
                        )
                        st.write(chunk.get("text", ""))

        if plan.get("reasoning"):
            with st.expander("Planner reasoning"):
                st.write(plan["reasoning"])

        st.divider()


if mode.startswith("Quick"):
    render_quick_answer()
else:
    render_research_note()

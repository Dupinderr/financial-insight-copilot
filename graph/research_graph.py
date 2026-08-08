"""LangGraph assembly for the v2 multi-agent research pipeline.

    planner ──┬──> data_agent ────┬──> synthesizer ──> critic ──> END
              └──> research_agent ┘

The planner picks which of the two retrieval agents run; when it picks both they
execute in parallel. They write to disjoint keys in the state, so the fan-in to
the synthesizer needs no reducer.
"""

import time

from langgraph.graph import END, START, StateGraph

from agents.critic import criticize
from agents.planner import plan as run_planner
from agents.research_agent import research
from agents.synthesizer import synthesize
from data_agent import format_sql_evidence, query_financial_data
from observability.tracing import flush, trace_run, trace_span

from .state import ResearchState


# --- nodes -------------------------------------------------------------------

def planner_node(state: ResearchState) -> dict:
    question = state["question"]
    with trace_span("planner", as_type="agent", input={"question": question}) as span:
        decision = run_planner(question)
        span.update(output=decision)
    return {"plan": decision}


def data_node(state: ResearchState) -> dict:
    question = state["plan"].get("data_question") or state["question"]

    with trace_span("data_agent", as_type="tool", input={"question": question}) as span:
        result = query_financial_data(question)
        evidence = format_sql_evidence(result)
        span.update(
            output={
                "ok": result["ok"],
                "sql": result.get("sql"),
                "row_count": result.get("row_count"),
                "error": result.get("error"),
            }
        )

    return {"sql_result": result, "sql_evidence": evidence}


def research_node(state: ResearchState) -> dict:
    plan = state["plan"]
    query = plan.get("research_query") or state["question"]
    tickers = plan.get("tickers") or []

    with trace_span("research_agent", as_type="retriever", input={"query": query, "tickers": tickers}) as span:
        result = research(query, tickers=tickers)
        span.update(
            output={
                "ok": result["ok"],
                "chunk_count": len(result["chunks"]),
                "sources": sorted({c.get("source") for c in result["chunks"] if c.get("source")}),
                "error": result.get("error"),
            }
        )

    return {"doc_result": result, "doc_evidence": result.get("evidence") or None}


def synthesizer_node(state: ResearchState) -> dict:
    with trace_span(
        "synthesizer",
        as_type="agent",
        input={
            "question": state["question"],
            "has_sql": bool(state.get("sql_evidence")),
            "has_docs": bool(state.get("doc_evidence")),
        },
    ) as span:
        draft = synthesize(
            question=state["question"],
            sql_evidence=state.get("sql_evidence"),
            doc_evidence=state.get("doc_evidence"),
        )
        span.update(output=draft)

    return {"draft": draft}


def critic_node(state: ResearchState) -> dict:
    with trace_span("critic", as_type="evaluator", input={"draft": state["draft"]}) as span:
        critique = criticize(
            draft=state["draft"],
            sql_evidence=state.get("sql_evidence"),
            doc_evidence=state.get("doc_evidence"),
            strip_unverified=state.get("strip_unverified", False),
            sql_result=state.get("sql_result"),
            doc_result=state.get("doc_result"),
            question=state["question"],
        )
        span.update(
            output={
                "confidence": critique["confidence"],
                "verified": critique["verified_count"],
                "unverified": critique["unverified_count"],
                "claims": critique["claims"],
            }
        )

    return {
        "critique": critique,
        "verified_answer": critique["verified_answer"],
        "confidence": critique["confidence"],
    }


# --- routing -----------------------------------------------------------------

def route_after_planner(state: ResearchState) -> list[str]:
    """Fan out to whichever retrieval agents the planner asked for."""
    plan = state["plan"]
    targets = []

    if plan.get("use_data"):
        targets.append("data_agent")
    if plan.get("use_research"):
        targets.append("research_agent")

    return targets or ["data_agent"]


# --- assembly ----------------------------------------------------------------

_compiled = None


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("data_agent", data_node)
    builder.add_node("research_agent", research_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["data_agent", "research_agent"],
    )
    builder.add_edge("data_agent", "synthesizer")
    builder.add_edge("research_agent", "synthesizer")
    builder.add_edge("synthesizer", "critic")
    builder.add_edge("critic", END)

    return builder.compile()


def get_graph():
    """Compiled graph, built once per process."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_research(question: str, strip_unverified: bool = False) -> dict:
    """Run one full multi-agent pass and return a serialisable result."""
    started = time.time()

    with trace_run(
        "financial_research",
        input={"question": question},
        metadata={"strip_unverified": strip_unverified},
    ) as (span, trace_url):
        final = get_graph().invoke(
            {"question": question, "strip_unverified": strip_unverified}
        )
        span.update(
            output={
                "confidence": final.get("confidence"),
                "answer": final.get("verified_answer"),
            }
        )

    elapsed = round(time.time() - started, 2)
    critique = final.get("critique", {})
    plan = final.get("plan", {})
    sql_result = final.get("sql_result") or {}
    doc_result = final.get("doc_result") or {}

    flush()

    return {
        "question": question,
        "answer": final.get("verified_answer", ""),
        "draft": final.get("draft", ""),
        "confidence": final.get("confidence", "low"),
        "plan": {
            "use_data": plan.get("use_data"),
            "use_research": plan.get("use_research"),
            "tickers": plan.get("tickers", []),
            "reasoning": plan.get("reasoning", ""),
        },
        "verification": {
            "verified_count": critique.get("verified_count", 0),
            "unverified_count": critique.get("unverified_count", 0),
            "claims": critique.get("claims", []),
        },
        "sql": {
            "query": sql_result.get("sql"),
            "row_count": sql_result.get("row_count"),
            "rows": sql_result.get("rows", [])[:10],
            "error": sql_result.get("error"),
        } if sql_result else None,
        "documents": {
            "chunk_count": len(doc_result.get("chunks", [])),
            "sources": sorted({c.get("source") for c in doc_result.get("chunks", []) if c.get("source")}),
            "chunks": [
                {
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "similarity": c.get("similarity"),
                    "text": c.get("text", "")[:400],
                }
                for c in doc_result.get("chunks", [])[:5]
            ],
            "error": doc_result.get("error"),
        } if doc_result else None,
        "trace_url": trace_url,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "How volatile has TCS been in 2026, and what is driving it?"
    print(json.dumps(run_research(q), indent=2, default=str))

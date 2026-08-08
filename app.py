"""FastAPI service for Financial Insight Copilot.

/ask      — v1 single-agent text-to-SQL (behaviour unchanged from v1)
/research — v2 multi-agent verified research note
"""

from fastapi import FastAPI
from pydantic import BaseModel

# The text-to-SQL pipeline now lives in data_agent.py. It was moved out of this
# file verbatim so the v2 graph can import it without a circular import back
# into this module. /ask below still calls exactly the same functions it did in v1.
from data_agent import ask_v1

app = FastAPI(title="Financial Insight Copilot")


class Question(BaseModel):
    question: str


class ResearchRequest(BaseModel):
    question: str
    strip_unverified: bool = False


@app.post("/ask")
def ask(q: Question):
    return ask_v1(q.question)


@app.post("/research")
def research(req: ResearchRequest):
    """v2: planner -> data/research agents -> synthesizer -> critic.

    Imported lazily so that /ask keeps working (and the service keeps starting)
    even if a v2-only dependency is missing or the Chroma index isn't built.
    """
    from graph.research_graph import run_research

    try:
        return run_research(req.question, strip_unverified=req.strip_unverified)
    except Exception as e:
        return {"question": req.question, "error": str(e)}


@app.get("/health")
def health():
    """Reports which v2 subsystems are actually live."""
    from ingestion.retriever import corpus_status
    from observability.tracing import tracing_status

    return {
        "status": "ok",
        "corpus": corpus_status(),
        "tracing": tracing_status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

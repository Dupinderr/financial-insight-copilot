"""Research Agent — retrieves qualitative evidence from the document corpus."""

from ingestion.retriever import corpus_status, format_document_evidence, search_documents

PER_TICKER_K = 4
GENERAL_K = 6


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Drop chunks retrieved twice by different per-ticker searches."""
    seen = set()
    unique = []
    for chunk in chunks:
        key = (chunk.get("source"), chunk.get("page"), chunk.get("text", "")[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def research(query: str, tickers: list[str] | None = None) -> dict:
    """Search the corpus, optionally once per relevant company.

    Splitting the search per ticker stops a single company from monopolising
    the top-k when a question compares two of them.
    """
    status = corpus_status()
    if not status["available"]:
        return {
            "ok": False,
            "source": "documents",
            "query": query,
            "chunks": [],
            "evidence": "",
            "error": (
                f"Document corpus unavailable — {status['reason']}. "
                "Add files under ingestion/corpus/ and run `python -m ingestion.ingest`."
            ),
        }

    if tickers:
        chunks = []
        for ticker in tickers:
            result = search_documents(query, k=PER_TICKER_K, ticker=ticker)
            if result["ok"]:
                chunks.extend(result["chunks"])
        # A ticker-filtered search returns nothing when that company has no
        # documents, so fall back to an unfiltered pass rather than give up.
        if not chunks:
            fallback = search_documents(query, k=GENERAL_K)
            chunks = fallback["chunks"] if fallback["ok"] else []
    else:
        result = search_documents(query, k=GENERAL_K)
        chunks = result["chunks"] if result["ok"] else []

    chunks = _dedupe(chunks)
    chunks.sort(key=lambda c: c.get("similarity", 0), reverse=True)

    payload = {"ok": True, "source": "documents", "query": query, "chunks": chunks, "error": None}
    payload["evidence"] = format_document_evidence(payload)
    return payload

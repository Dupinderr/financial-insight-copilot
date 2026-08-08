"""Retriever tool over the Chroma corpus — the Research Agent's data access."""

from .config import COLLECTION_NAME, CORPUS_TICKERS, get_collection

DEFAULT_K = 5

# Chroma is configured for cosine distance, so similarity = 1 - distance.
# Below this the chunk is usually off-topic noise rather than weak-but-relevant.
MIN_SIMILARITY = 0.15

_collection = None


def _load_collection():
    global _collection
    if _collection is None:
        _collection = get_collection(create=False)
    return _collection


def corpus_status() -> dict:
    """Whether there is anything to retrieve from — used for graceful degradation."""
    collection = _load_collection()
    if collection is None:
        return {"available": False, "count": 0, "reason": f"collection '{COLLECTION_NAME}' not built yet"}

    count = collection.count()
    if count == 0:
        return {"available": False, "count": 0, "reason": "collection is empty"}

    return {"available": True, "count": count, "reason": None}


def search_documents(query: str, k: int = DEFAULT_K, ticker: str | None = None) -> dict:
    """Semantic search over the filings/news corpus.

    `ticker` accepts either the folder key (e.g. "TCS") or the DB symbol
    (e.g. "TCS.NS") and restricts results to that company.
    """
    status = corpus_status()
    if not status["available"]:
        return {
            "ok": False,
            "source": "documents",
            "query": query,
            "chunks": [],
            "error": f"No document corpus available — {status['reason']}.",
        }

    collection = _load_collection()

    where = None
    if ticker:
        symbol = CORPUS_TICKERS.get(ticker.upper().replace(".NS", ""), ticker)
        where = {"ticker": symbol}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(k, status["count"]),
            where=where,
        )
    except Exception as e:
        return {"ok": False, "source": "documents", "query": query, "chunks": [], "error": str(e)}

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - float(distance)
        if similarity < MIN_SIMILARITY:
            continue
        chunks.append(
            {
                "text": text,
                "source": meta.get("source"),
                "company": meta.get("company"),
                "ticker": meta.get("ticker"),
                "doc_type": meta.get("doc_type"),
                "page": meta.get("page"),
                "similarity": round(similarity, 4),
            }
        )

    return {
        "ok": True,
        "source": "documents",
        "query": query,
        "chunks": chunks,
        "error": None,
    }


def citation_label(chunk: dict) -> str:
    page = chunk.get("page", -1)
    page_part = f", p.{page}" if page and page != -1 else ""
    return f"{chunk.get('source')}{page_part}"


def format_document_evidence(result: dict) -> str:
    """Render retrieved chunks as numbered evidence for a prompt."""
    if not result.get("ok"):
        return f"[Document retrieval unavailable: {result.get('error')}]"
    if not result["chunks"]:
        return "No relevant document passages found."

    blocks = []
    for i, chunk in enumerate(result["chunks"], start=1):
        blocks.append(f"[DOC-{i}] ({citation_label(chunk)}, similarity {chunk['similarity']})\n{chunk['text']}")
    return "\n\n".join(blocks)

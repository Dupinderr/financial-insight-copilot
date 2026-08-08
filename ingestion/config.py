"""Shared paths and knobs for the Research Agent's document corpus."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = PROJECT_ROOT / "ingestion" / "corpus"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "financial_filings"

# Chunking, per spec: ~1000 characters with 200 characters of overlap.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Tickers scoped for the RAG corpus. Each maps to a subfolder of CORPUS_DIR and
# to the ticker symbol used in the SQLite table, so a research note can join
# retrieved text back to the quantitative data.
CORPUS_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
}

# chromadb's bundled all-MiniLM-L6-v2 downloads to ~/.cache/chroma by default.
# On this machine ~/.cache is root-owned (drwx------), so that write fails.
# Keeping the model inside the project avoids needing sudo.
MODEL_CACHE_DIR = PROJECT_ROOT / ".models" / "onnx"


def get_embedding_function():
    """all-MiniLM-L6-v2 via chromadb's bundled ONNX runtime.

    Same weights as the sentence-transformers version named in the spec, but
    served through onnxruntime so the project doesn't pull in ~2GB of torch.
    """
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = MODEL_CACHE_DIR / ONNXMiniLM_L6_V2.MODEL_NAME
    return ONNXMiniLM_L6_V2()


def get_collection(create: bool = False):
    """Open the persisted Chroma collection, or None if it doesn't exist yet."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = get_embedding_function()

    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        return None

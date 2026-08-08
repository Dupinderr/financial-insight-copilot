"""Build the Chroma collection from ingestion/corpus/.

Usage:
    python -m ingestion.ingest            # incremental (upsert)
    python -m ingestion.ingest --rebuild  # drop the collection and start over
"""

import argparse
import sys

from .chunker import chunk_records
from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    CORPUS_DIR,
    get_collection,
)
from .loaders import load_corpus

BATCH_SIZE = 100


def rebuild_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass


def ingest(rebuild: bool = False) -> int:
    if rebuild:
        rebuild_collection()

    print(f"Reading corpus from {CORPUS_DIR}")
    records = load_corpus()

    if not records:
        print(
            "\nNo documents found.\n"
            f"Drop PDFs / .txt files into {CORPUS_DIR}/<RELIANCE|TCS|SUNPHARMA>/\n"
            "See ingestion/corpus/README.md for the suggested document list.",
            file=sys.stderr,
        )
        return 0

    chunks = chunk_records(records)
    print(f"Loaded {len(records)} document sections -> {len(chunks)} chunks "
          f"({CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap)")

    collection = get_collection(create=True)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    total = collection.count()
    print(f"\nCollection '{COLLECTION_NAME}' now holds {total} chunks at {CHROMA_DIR}")

    by_company = {}
    for c in chunks:
        by_company[c["metadata"]["company"]] = by_company.get(c["metadata"]["company"], 0) + 1
    for company, n in sorted(by_company.items()):
        print(f"  {company}: {n} chunks")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="drop the collection first")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)

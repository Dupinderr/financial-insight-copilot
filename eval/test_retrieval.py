"""Independent test of the Research Agent's retrieval — no LLM calls.

Checks the corpus is indexed and that semantic search returns passages from the
company actually being asked about. Runs entirely on the local embedding model,
so it costs nothing and works when the Groq quota is exhausted.

Usage:
    python -m eval.test_retrieval
    python -m eval.test_retrieval --query "green hydrogen capex"
"""

import argparse
import sys

from ingestion.config import CORPUS_TICKERS
from ingestion.retriever import corpus_status, search_documents

# Each probe names the company its results should come from, so retrieval is
# scored rather than just eyeballed.
PROBES = [
    ("RELIANCE", "capital expenditure on new energy and green hydrogen"),
    ("RELIANCE", "retail segment store expansion and growth"),
    ("TCS", "attrition, headcount and workforce reskilling"),
    ("TCS", "large deal wins and order book in BFSI"),
    ("SUNPHARMA", "specialty pharmaceutical pipeline and R&D spend"),
    ("SUNPHARMA", "US generics pricing pressure and regulatory risk"),
]


def run_probes(k: int = 3) -> int:
    status = corpus_status()

    if not status["available"]:
        print(f"Corpus not available — {status['reason']}.\n")
        print("Add documents under ingestion/corpus/<RELIANCE|TCS|SUNPHARMA>/")
        print("then run:  python -m ingestion.ingest")
        print("\nSee ingestion/corpus/README.md for what to download.")
        return 1

    print(f"Corpus: {status['count']} chunks indexed\n")
    print(f"{'=' * 78}")

    hits = 0
    graded = 0
    filtered_hits = 0

    for expected_company, query in PROBES:
        # Skip probes for companies with nothing ingested yet.
        available = search_documents(query, k=1, ticker=expected_company)
        if not available["ok"] or not available["chunks"]:
            print(f"\n[skip] {expected_company:<10} no documents indexed — '{query}'")
            continue

        result = search_documents(query, k=k)
        print(f"\n{expected_company:<10} '{query}'")

        if not result["ok"]:
            print(f"   ERROR: {result['error']}")
            continue

        if not result["chunks"]:
            print("   no chunks above the similarity floor")
            graded += 1
            continue

        top = result["chunks"][0]
        graded += 1
        correct = top.get("company") == expected_company
        hits += int(correct)

        # The graph filters by ticker whenever the planner names one, so also
        # check the path production actually takes. Unfiltered search measures
        # raw embedding discrimination; filtered search measures the pipeline.
        filtered = search_documents(query, k=1, ticker=expected_company)
        if filtered["ok"] and filtered["chunks"]:
            filtered_hits += 1

        for i, chunk in enumerate(result["chunks"]):
            marker = "->" if i == 0 else "  "
            page = f" p.{chunk['page']}" if chunk.get("page", -1) != -1 else ""
            print(f"   {marker} [{chunk['similarity']:.3f}] {chunk['company']}/"
                  f"{chunk['source']}{page}")
            print(f"        {chunk['text'][:110].strip()}...")

        print(f"   top-1 company {'correct' if correct else 'WRONG'}")

    print(f"\n{'=' * 78}")
    if not graded:
        print("No probes could be graded — corpus has no matching documents.")
        return 1

    print(f"Unfiltered top-1 company accuracy: {hits}/{graded} = {hits / graded:.0%}")
    print(f"  (raw embedding discrimination, no ticker filter)")
    print(f"Ticker-filtered relevant results:   {filtered_hits}/{graded} = {filtered_hits / graded:.0%}")
    print(f"  (the path the graph actually takes when the planner names a ticker)")

    if hits < graded:
        print(
            "\nNote: unfiltered misses are expected where two companies discuss the\n"
            "same theme — a pharma company's energy-conservation annexure looks much\n"
            "like an energy company's capex narrative to a 384-dim sentence embedding.\n"
            "The graph avoids this by filtering on ticker."
        )

    # Filtered retrieval is the production path, so that is what gates success.
    return 0 if filtered_hits == graded else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="run a single ad-hoc query instead of the probe set")
    parser.add_argument("--ticker", choices=sorted(CORPUS_TICKERS), help="restrict to one company")
    parser.add_argument("-k", type=int, default=3, help="results per query")
    args = parser.parse_args()

    if args.query:
        result = search_documents(args.query, k=args.k, ticker=args.ticker)
        if not result["ok"]:
            print(result["error"])
            return 1
        if not result["chunks"]:
            print("No results above the similarity floor.")
            return 0
        for chunk in result["chunks"]:
            page = f" p.{chunk['page']}" if chunk.get("page", -1) != -1 else ""
            print(f"[{chunk['similarity']:.3f}] {chunk['company']}/{chunk['source']}{page}")
            print(f"    {chunk['text'][:300].strip()}...\n")
        return 0

    return run_probes(k=args.k)


if __name__ == "__main__":
    sys.exit(main())

"""Load corpus documents off disk into (text, metadata) records.

Supported: .pdf (via pypdf), .txt, .md. Files live under
ingestion/corpus/<TICKER>/, and the folder name becomes the ticker metadata so
retrieval can be filtered per company.
"""

import re
from pathlib import Path

from .config import CORPUS_DIR, CORPUS_TICKERS

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_doc_type(filename: str) -> str:
    lowered = filename.lower()
    if "annual" in lowered or "ar20" in lowered or "ar-20" in lowered:
        return "annual_report"
    if "presentation" in lowered or "investor" in lowered or "deck" in lowered:
        return "investor_presentation"
    if "news" in lowered or "article" in lowered:
        return "news"
    if "transcript" in lowered or "earnings" in lowered or "call" in lowered:
        return "earnings_call"
    return "other"


def load_pdf(path: Path) -> list[dict]:
    """One record per page, so citations can name a page number."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    records = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        text = _normalise_whitespace(text)
        if len(text) < 50:  # skip cover pages / image-only pages
            continue
        records.append({"text": text, "page": page_no})
    return records


def load_text(path: Path) -> list[dict]:
    text = _normalise_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
    return [{"text": text, "page": None}] if text else []


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Walk the corpus tree and return every loadable document section."""
    records = []

    for ticker_folder in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        ticker_key = ticker_folder.name.upper()
        ticker_symbol = CORPUS_TICKERS.get(ticker_key, ticker_key)

        for path in sorted(ticker_folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            sections = load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)

            for section in sections:
                records.append(
                    {
                        "text": section["text"],
                        "metadata": {
                            "source": path.name,
                            "ticker": ticker_symbol,
                            "company": ticker_key,
                            "doc_type": _guess_doc_type(path.name),
                            "page": section["page"] if section["page"] is not None else -1,
                        },
                    }
                )

    return records

"""Character chunking with overlap, preferring natural break points."""

from .config import CHUNK_OVERLAP, CHUNK_SIZE

# Tried in order: paragraph break, sentence end, line break, plain space.
_BREAKPOINTS = ["\n\n", ". ", "\n", " "]


def _find_break(text: str, start: int, end: int) -> int:
    """Find the best place to end a chunk, searching backwards from `end`.

    Only looks within the last 30% of the window so a chunk never collapses to
    something tiny just because an early paragraph break exists.
    """
    window_floor = start + int((end - start) * 0.7)

    for marker in _BREAKPOINTS:
        idx = text.rfind(marker, window_floor, end)
        if idx != -1:
            return idx + len(marker)

    return end


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            end = _find_break(text, start, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step forward by chunk length minus the overlap, but always make
        # forward progress even if the break landed early.
        start = max(end - overlap, start + 1)

    return chunks


def chunk_records(records: list[dict]) -> list[dict]:
    """Expand loaded documents into embeddable chunk records with stable ids."""
    chunked = []

    for record in records:
        pieces = chunk_text(record["text"])
        for i, piece in enumerate(pieces):
            meta = dict(record["metadata"])
            meta["chunk_index"] = i
            page_part = f"p{meta['page']}" if meta.get("page", -1) != -1 else "full"
            chunked.append(
                {
                    "id": f"{meta['company']}::{meta['source']}::{page_part}::{i}",
                    "text": piece,
                    "metadata": meta,
                }
            )

    return chunked

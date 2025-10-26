# llmextract/chunker.py

import logging
from typing import Iterator, NamedTuple

logger = logging.getLogger(__name__)


class TextChunk(NamedTuple):
    """
    Represents a chunk of text and its starting position in the original document.
    """

    text: str
    start_char: int


def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, keep_trailing: bool = True
) -> Iterator[TextChunk]:
    """
    Split long text into possibly overlapping chunks.

    Args:
      text: full source text
      chunk_size: maximum characters per chunk (> chunk_overlap)
      chunk_overlap: overlap in characters between consecutive chunks (>= 0)
      keep_trailing: whether to yield a final short remainder chunk

    Yields TextChunk instances.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0.")
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    text_len = len(text or "")
    if text_len == 0:
        return

    start = 0
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            # final chunk
            if keep_trailing or start == 0:
                yield TextChunk(text=text[start:text_len], start_char=start)
            break

        yield TextChunk(text=text[start:end], start_char=start)
        start += chunk_size - chunk_overlap

    logger.debug(
        "Chunking complete. text_len=%d chunk_size=%d chunk_overlap=%d keep_trailing=%s",
        text_len,
        chunk_size,
        chunk_overlap,
        keep_trailing,
    )

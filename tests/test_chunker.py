# tests/test_chunker.py

import pytest
from llmextract.chunker import chunk_text, TextChunk


def test_basic_chunking():
    """Tests that text is split into correct, non-overlapping chunks."""
    text = "abcdefghij"  # 10 chars
    chunks = list(chunk_text(text, chunk_size=4, chunk_overlap=0))
    assert len(chunks) == 3
    assert chunks[0] == TextChunk(text="abcd", start_char=0)
    assert chunks[1] == TextChunk(text="efgh", start_char=4)
    assert chunks[2] == TextChunk(text="ij", start_char=8)


def test_chunking_with_overlap():
    """Tests that chunks overlap correctly and the final chunk is included."""
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    chunks = list(chunk_text(text, chunk_size=10, chunk_overlap=3))

    assert len(chunks) == 4
    assert chunks[0] == TextChunk(text="abcdefghij", start_char=0)
    assert chunks[1] == TextChunk(text="hijklmnopq", start_char=7)
    assert chunks[2] == TextChunk(text="opqrstuvwx", start_char=14)
    assert chunks[3] == TextChunk(text="vwxyz", start_char=21)


def test_text_smaller_than_chunk_size():
    """Tests that a single chunk is returned if the text is small."""
    text = "short text"
    chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=10))
    assert len(chunks) == 1
    assert chunks[0] == TextChunk(text="short text", start_char=0)


def test_empty_text():
    """Tests that no chunks are generated for empty text."""
    text = ""
    chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=10))
    assert len(chunks) == 0


def test_invalid_chunk_config_raises_error():
    """Tests that chunk_size > chunk_overlap is enforced."""
    with pytest.raises(
        ValueError, match="chunk_size must be greater than chunk_overlap"
    ):
        list(chunk_text("some text", chunk_size=10, chunk_overlap=10))

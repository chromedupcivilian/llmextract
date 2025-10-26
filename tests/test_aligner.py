# tests/test_aligner.py

from llmextract.aligner import align_extractions
from llmextract.data_models import Extraction


def test_align_in_order_extractions():
    """Tests alignment when LLM returns extractions in the correct order."""
    text = "The quick brown fox jumps over the lazy dog."
    extractions = [
        Extraction(extraction_class="animal", extraction_text="fox"),
        Extraction(extraction_class="animal", extraction_text="dog"),
    ]

    aligned = align_extractions(extractions, text)

    assert len(aligned) == 2
    # Add assert to satisfy type checker
    assert aligned[0].char_interval is not None
    assert aligned[0].char_interval.start == 16
    assert aligned[0].char_interval.end == 19

    assert aligned[1].char_interval is not None
    assert aligned[1].char_interval.start == 40
    assert aligned[1].char_interval.end == 43


def test_align_out_of_order_extractions():
    """Tests alignment when LLM returns extractions out of order."""
    text = "The quick brown fox jumps over the lazy dog."
    extractions = [
        Extraction(extraction_class="animal", extraction_text="dog"),
        Extraction(extraction_class="animal", extraction_text="fox"),
    ]

    aligned = align_extractions(extractions, text)

    assert len(aligned) == 2
    assert aligned[0].char_interval is not None
    assert aligned[0].char_interval.start == 40
    assert aligned[0].char_interval.end == 43

    assert aligned[1].char_interval is not None
    assert aligned[1].char_interval.start == 16
    assert aligned[1].char_interval.end == 19


def test_align_case_insensitivity():
    """Tests that alignment is case-insensitive."""
    text = "The quick BROWN FOX jumps over the lazy dog."
    extractions = [Extraction(extraction_class="animal", extraction_text="brown fox")]

    aligned = align_extractions(extractions, text)

    assert aligned[0].char_interval is not None
    assert aligned[0].char_interval.start == 10
    assert aligned[0].char_interval.end == 19


def test_align_extraction_not_found():
    """Tests that unalignable extractions have no char_interval."""
    text = "The quick brown fox."
    extractions = [Extraction(extraction_class="animal", extraction_text="cat")]

    aligned = align_extractions(extractions, text)

    assert len(aligned) == 1
    assert aligned[0].char_interval is None

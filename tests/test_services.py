# tests/test_services.py

from unittest.mock import MagicMock, patch

# Update the import to reflect the new structure
from llmextract.services import extract
from llmextract.data_models import ExampleData, Extraction


# We now mock the provider factory in the 'services' module where it is used.
@patch("llmextract.services.get_llm_provider")
def test_extract_end_to_end_mocked(mock_get_provider):
    """
    Verifies the entire extract pipeline using a mocked LLM.
    """
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"extractions": [{"extraction_class": "fruit", "extraction_text": "apple"}]}'
    )
    mock_llm.invoke.return_value = mock_response
    mock_get_provider.return_value = mock_llm

    text = "I like to eat an apple every day."
    prompt = "Extract fruits."
    examples = [
        ExampleData(
            text="Bananas are yellow.",
            extractions=[
                Extraction(extraction_class="fruit", extraction_text="Bananas")
            ],
        )
    ]

    result = extract(
        text,
        prompt,
        examples,
        model_name="mock-model",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert result.text == text
    assert len(result.extractions) == 1
    extraction = result.extractions[0]
    assert extraction.extraction_class == "fruit"
    assert extraction.extraction_text == "apple"
    assert extraction.char_interval is not None
    assert extraction.char_interval.start == 17
    assert extraction.char_interval.end == 22

    print("\nTest passed: Mocked end-to-end extraction successful.")

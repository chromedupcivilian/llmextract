# llmextract/prompts.py

import json
import logging
from typing import List, Optional

from .data_models import ExampleData

logger = logging.getLogger(__name__)


def format_prompt(
    prompt_description: str,
    examples: List[ExampleData],
    text: str,
    max_examples: Optional[int] = None,
) -> str:
    """
    Build the prompt sent to the LLM.

    The model is instructed to reply with only a single JSON object:
    {"extractions": [ ... ]} and no additional prose.
    """
    if max_examples is not None and len(examples) > max_examples:
        logger.debug(
            "Truncating examples from %d to %d (max_examples=%d).",
            len(examples),
            max_examples,
            max_examples,
        )
        examples = examples[-max_examples:]

    parts: List[str] = []
    parts.append("You are an information extraction assistant.")
    parts.append(
        "IMPORTANT: Respond with a single, valid JSON object and nothing else. "
        'The object MUST have a top-level key "extractions" which is a list.'
    )
    parts.append(
        "Each extraction object must look like:\n"
        '{"extraction_class": "string", "extraction_text": "string", "attributes": {}}'
    )
    parts.append("\n-- PROMPT DESCRIPTION --")
    parts.append(prompt_description)
    parts.append("\n-- EXAMPLES --")

    for example in examples:
        extractions_list = [
            ext.model_dump(exclude={"char_interval"}, exclude_none=True)
            for ext in example.extractions
        ]
        example_json = json.dumps(
            {"extractions": extractions_list}, indent=2, ensure_ascii=False
        )
        parts.append(f"Text:\n'''\n{example.text}\n'''")
        parts.append(f"JSON Output:\n{example_json}")

    parts.append("\n-- TASK --")
    parts.append('If there are no extractions, return {"extractions": []}.')
    parts.append(f"Text:\n'''\n{text}\n'''")
    parts.append("JSON Output:")

    return "\n\n".join(parts)

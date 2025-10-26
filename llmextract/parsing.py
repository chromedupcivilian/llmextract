# llmextract/parsing.py
import json
import logging
import re
from typing import Any, Dict, List, Optional

from .aligner import align_extractions
from .chunker import TextChunk
from .data_models import Extraction

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """
    If the LLM output contains a triple-backtick code fence (``` or ```json),
    return the inner content. Otherwise return the original text.
    """
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _find_json_candidates(text: Optional[str]) -> List[str]:
    """
    Scan text and return substrings that look like balanced JSON objects or arrays.
    This is a lightweight scanner that avoids counting braces inside JSON strings.
    """
    candidates: List[str] = []
    if not text:
        return candidates

    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue

        opening = ch
        depth = 0
        in_string = False
        esc = False
        start = i
        j = i
        while j < n:
            c = text[j]
            if c == '"' and not esc:
                in_string = not in_string
            if c == "\\" and not esc:
                esc = True
            else:
                esc = False

            if not in_string:
                if c == opening:
                    depth += 1
                elif (opening == "{" and c == "}") or (opening == "[" and c == "]"):
                    depth -= 1

                if depth == 0:
                    candidates.append(text[start : j + 1])
                    i = j + 1
                    break
            j += 1
        else:
            # no matching close found; advance one char from start
            i = start + 1

    return candidates


def transform_llm_extractions(raw_extractions: Any) -> List[Dict[str, Any]]:
    """
    Normalize many common LLM output shapes to a list of dicts:
      {"extraction_class", "extraction_text", "attributes"}.
    """
    class_synonyms = {
        "extraction_class",
        "class",
        "type",
        "label",
        "category",
        "name",
    }
    text_synonyms = {
        "extraction_text",
        "text",
        "value",
        "span",
        "extracted_text",
        "item",
    }
    attr_synonyms = {"attributes", "attrs", "properties", "metadata", "meta"}

    corrected: List[Dict[str, Any]] = []

    if raw_extractions is None:
        return corrected

    if isinstance(raw_extractions, dict) and "extractions" in raw_extractions:
        raw_extractions = raw_extractions["extractions"]

    if isinstance(raw_extractions, (str, dict)):
        items = [raw_extractions]
    elif isinstance(raw_extractions, list):
        items = raw_extractions
    else:
        items = [str(raw_extractions)]

    for item in items:
        if isinstance(item, list):
            corrected.extend(transform_llm_extractions(item))
            continue

        if isinstance(item, dict):
            if "extractions" in item and isinstance(item["extractions"], list):
                corrected.extend(transform_llm_extractions(item["extractions"]))
                continue

            map_keys = {k.lower(): k for k in item.keys()}

            class_key = next(
                (map_keys[k] for k in map_keys if k in class_synonyms), None
            )
            text_key = next((map_keys[k] for k in map_keys if k in text_synonyms), None)
            attr_key = next((map_keys[k] for k in map_keys if k in attr_synonyms), None)

            if class_key and text_key:
                ext_class = item.get(class_key)
                ext_text = item.get(text_key)
                attrs = item.get(attr_key, {})
                if not isinstance(attrs, dict):
                    attrs = {"value": attrs}
                corrected.append(
                    {
                        "extraction_class": str(ext_class).strip(),
                        "extraction_text": str(ext_text).strip(),
                        "attributes": attrs or {},
                    }
                )
                continue

            if len(item) == 1:
                key, val = next(iter(item.items()))
                corrected.append(
                    {
                        "extraction_class": str(key).strip(),
                        "extraction_text": str(val).strip(),
                        "attributes": {},
                    }
                )
                continue

            if "label" in item and ("span" in item or "text" in item):
                ext_class = item.get("label")
                ext_text = item.get("span") or item.get("text")
                corrected.append(
                    {
                        "extraction_class": str(ext_class).strip(),
                        "extraction_text": str(ext_text).strip(),
                        "attributes": {},
                    }
                )
                continue

            corrected.append(
                {
                    "extraction_class": "unknown",
                    "extraction_text": json.dumps(item, ensure_ascii=False),
                    "attributes": {},
                }
            )
            continue

        if isinstance(item, str):
            s = item.strip()
            try:
                parsed = json.loads(s)
                corrected.extend(transform_llm_extractions(parsed))
                continue
            except Exception:
                pass

            m = re.match(r"^\s*([^:–—\-]+?)\s*[:\-–—]\s*(.+)$", s)
            if m:
                corrected.append(
                    {
                        "extraction_class": m.group(1).strip(),
                        "extraction_text": m.group(2).strip(),
                        "attributes": {},
                    }
                )
                continue

            corrected.append(
                {"extraction_class": "unknown", "extraction_text": s, "attributes": {}}
            )
            continue

        corrected.append(
            {
                "extraction_class": "unknown",
                "extraction_text": str(item),
                "attributes": {},
            }
        )

    return corrected


def parse_and_align_chunk(
    llm_output_content: str, chunk: TextChunk
) -> List[Extraction]:
    """
    Parse LLM output for a single chunk and align extractions to the chunk text.
    Returns a list of Extraction objects (with char_interval relative to full document).
    """
    content = _strip_code_fence(llm_output_content or "")

    raw_extractions: Optional[List[Any]] = None

    # Try to parse whole content as JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "extractions" in parsed:
            raw_extractions = parsed["extractions"]
        elif isinstance(parsed, list):
            raw_extractions = parsed
    except Exception:
        # try scanning for balanced JSON objects/arrays
        candidates = _find_json_candidates(content)
        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict) and "extractions" in parsed:
                    raw_extractions = parsed["extractions"]
                    break
                elif isinstance(parsed, list):
                    raw_extractions = parsed
                    break
            except Exception:
                continue

    if raw_extractions is None:
        logger.warning(
            "No valid JSON 'extractions' found in LLM output for chunk starting at %d. Truncated output: %.200r",
            chunk.start_char,
            content[:200],
        )
        return []

    transformed = transform_llm_extractions(raw_extractions)

    validated: List[Extraction] = []
    for idx, item in enumerate(transformed):
        try:
            ext = Extraction(**item)
            validated.append(ext)
        except Exception as e:
            logger.warning(
                "Skipping invalid extraction (chunk_start=%d, idx=%d): %s",
                chunk.start_char,
                idx,
                e,
            )
            continue

    if not validated:
        return []

    aligned_in_chunk = align_extractions(validated, chunk.text)

    for ext in aligned_in_chunk:
        if ext.char_interval:
            ext.char_interval.start += chunk.start_char
            ext.char_interval.end += chunk.start_char

    return aligned_in_chunk

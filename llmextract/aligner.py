# llmextract/aligner.py
import difflib
import logging
import re
import unicodedata
from typing import List, Optional, Tuple

from .data_models import CharInterval, Extraction

logger = logging.getLogger(__name__)


def _clean_for_pattern(s: Optional[str]) -> str:
    """
    Normalize input for pattern building:
      - Unicode normalize (NFC)
      - Remove common zero-width characters

    Returns a string safe to split/tokenize for regex construction.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    # remove zero-width and BOM characters
    zero_width_chars = (chr(0x200B), chr(0x200C), chr(0x200D), chr(0xFEFF))
    for zw in zero_width_chars:
        s = s.replace(zw, "")
    return s


def align_extractions(
    extractions: List[Extraction],
    original_text: str,
    fuzzy_threshold: float = 0.85,
) -> List[Extraction]:
    """
    Align a list of Extraction objects to character offsets in original_text.

    Strategy attempted in order:
      1) Exact, case-insensitive substring search starting from last match end.
      2) Exact, case-insensitive global search.
      3) Regex search that treats whitespace runs as the regex '\\s+' (flexible spacing).
      4) A lightweight fuzzy-match fallback using difflib.SequenceMatcher.

    If alignment fails for an extraction, its char_interval is left as None.
    """

    lower_original = original_text.lower()
    last_match_end = 0
    aligned: List[Extraction] = []

    for idx, extraction in enumerate(extractions):
        raw_text = extraction.extraction_text
        if not raw_text or not raw_text.strip():
            logger.warning(
                "Skipping alignment for empty extraction_text (index=%d).", idx
            )
            aligned.append(extraction)
            continue

        normalized_raw = _clean_for_pattern(raw_text)
        lower_search = normalized_raw.lower()

        start_index: int = -1

        # 1) Direct in-order search
        try:
            start_index = lower_original.find(lower_search, last_match_end)
        except Exception:
            start_index = -1

        # 2) Fallback to global search
        if start_index == -1:
            start_index = lower_original.find(lower_search)
            if start_index != -1:
                logger.debug(
                    "Extraction '%s' found out of order (index=%d); fell back to full-text search.",
                    raw_text,
                    idx,
                )

        # 3) Regex with flexible whitespace
        if start_index == -1:
            try:
                tokens = [t for t in re.split(r"\s+", normalized_raw) if t]
                if tokens:
                    token_patterns = [re.escape(tok) for tok in tokens]
                    pattern = r"\b" + r"\s+".join(token_patterns) + r"\b"
                else:
                    pattern = re.escape(normalized_raw)

                m = re.search(pattern, original_text, flags=re.IGNORECASE)
            except re.error as e:
                logger.debug(
                    "Regex build/search failed for extraction '%s': %s", raw_text, e
                )
                m = None

            if m:
                sidx, eidx = m.start(), m.end()
                try:
                    extraction.char_interval = CharInterval(start=sidx, end=eidx)
                    logger.debug(
                        "Aligned extraction via regex (index=%d): '%s' -> [%d:%d]",
                        idx,
                        raw_text,
                        sidx,
                        eidx,
                    )
                    if sidx >= last_match_end:
                        last_match_end = eidx
                    aligned.append(extraction)
                    continue
                except Exception as e:
                    logger.warning(
                        "Failed to set CharInterval from regex match for '%s': %s",
                        raw_text,
                        e,
                    )
                    # fall through to fuzzy/failure path

        # 4) Fuzzy fallback
        if start_index == -1:
            lowered = lower_search
            tokens = [t for t in re.split(r"\s+", lowered) if t]
            candidates: List[int] = []

            # find anchor positions for the longest token
            if tokens:
                longest_token = max(tokens, key=len)
                if len(longest_token) >= 4:
                    pos = 0
                    while True:
                        pos = lower_original.find(longest_token, pos)
                        if pos == -1:
                            break
                        candidates.append(pos)
                        pos += max(1, len(longest_token))

            # if none found, sample windows across the document
            if not candidates:
                step = max(1, len(lowered) // 10)
                candidates = list(range(0, max(1, len(lower_original) - 1), step))

            best_ratio = 0.0
            best_span: Optional[Tuple[int, int]] = None
            for cand in candidates:
                target_len = max(1, int(len(lowered) * 1.2))
                cand_end = min(len(lower_original), cand + target_len)
                candidate_sub = lower_original[cand:cand_end]
                if not candidate_sub:
                    continue
                ratio = difflib.SequenceMatcher(None, lowered, candidate_sub).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (cand, cand_end)

            if best_ratio >= fuzzy_threshold and best_span is not None:
                sidx, eidx = best_span
                try:
                    extraction.char_interval = CharInterval(start=sidx, end=eidx)
                    logger.debug(
                        "Aligned extraction via fuzzy match (index=%d) ratio=%.3f: '%s' -> [%d:%d]",
                        idx,
                        best_ratio,
                        raw_text,
                        sidx,
                        eidx,
                    )
                    if sidx >= last_match_end:
                        last_match_end = eidx
                    aligned.append(extraction)
                    continue
                except Exception as e:
                    logger.warning(
                        "Failed to set CharInterval from fuzzy match for '%s': %s",
                        raw_text,
                        e,
                    )
                    # fall through to not-found case

        # Final direct-found handling (if earlier direct find succeeded)
        if start_index != -1:
            end_index = start_index + len(raw_text)
            end_index = min(end_index, len(original_text))
            try:
                extraction.char_interval = CharInterval(
                    start=start_index, end=end_index
                )
                if start_index >= last_match_end:
                    last_match_end = end_index
                logger.debug(
                    "Aligned extraction (direct) (index=%d): '%s' -> [%d:%d]",
                    idx,
                    raw_text,
                    start_index,
                    end_index,
                )
            except Exception as e:
                logger.warning(
                    "Failed to set CharInterval for direct match for '%s': %s",
                    raw_text,
                    e,
                )
        else:
            logger.warning("Could not align extraction (index=%d): '%s'", idx, raw_text)

        aligned.append(extraction)

    return aligned

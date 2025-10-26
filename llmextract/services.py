# llmextract/services.py

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Dict, List, Optional, Tuple

from .chunker import chunk_text, TextChunk
from .data_models import AnnotatedDocument, ExampleData, Extraction
from .parsing import parse_and_align_chunk
from .prompts import format_prompt
from .providers import get_llm_provider

logger = logging.getLogger(__name__)


def _configure_verbose_logging(verbose: int) -> None:
    pkg_logger = logging.getLogger("llmextract")
    if verbose and not any(
        isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        pkg_logger.addHandler(handler)

    if verbose >= 2:
        pkg_logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        pkg_logger.setLevel(logging.INFO)
    else:
        pkg_logger.setLevel(logging.WARNING)


def _normalize_text_for_dedupe(s: str) -> str:
    return " ".join(s.split()).casefold()


def _dedupe_extractions(extractions: List[Extraction]) -> List[Extraction]:
    seen: Dict[Tuple[str, str], Extraction] = {}
    for ext in extractions:
        key = (ext.extraction_class, _normalize_text_for_dedupe(ext.extraction_text))
        current = seen.get(key)
        if current is None:
            seen[key] = ext
            continue

        cur_start = (
            current.char_interval.start if current.char_interval else float("inf")
        )
        new_start = ext.char_interval.start if ext.char_interval else float("inf")
        if new_start < cur_start:
            seen[key] = ext

    return list(seen.values())


def extract(
    text: str,
    prompt_description: str,
    examples: List[ExampleData],
    model_name: str,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    max_workers: int = 10,
    verbose: int = 0,
    error_mode: str = "return",
    retries: int = 2,
    retry_backoff: float = 1.0,
    dedupe: bool = False,
) -> AnnotatedDocument:
    """
    Synchronous extraction with per-chunk isolation and optional retries.

    error_mode: "return" (default) collects per-chunk errors in metadata["errors"],
                "raise" will re-raise the first chunk exception.
    """
    _configure_verbose_logging(verbose)

    # lazy import HumanMessage to avoid import-time dependency
    try:
        from langchain_core.messages import HumanMessage  # type: ignore
    except Exception:
        HumanMessage = None  # type: ignore

    if HumanMessage is None:
        raise RuntimeError(
            "langchain_core is required to call extract(); install langchain or "
            "use a compatible provider."
        )

    llm = get_llm_provider(model_name, provider_kwargs)
    chunks = list(chunk_text(text, chunk_size, chunk_overlap))

    def _process_chunk(chunk_index: int, chunk: TextChunk) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        # Build prompt first (catch errors early)
        try:
            prompt = format_prompt(prompt_description, examples, chunk.text)
        except Exception as e:
            logger.exception("Failed to build prompt for chunk %d: %s", chunk_index, e)
            return {
                "success": False,
                "error": f"prompt_error: {e}",
                "exception": e,
                "chunk_index": chunk_index,
            }

        # Log full prompt in DEBUG (full-verbose) mode
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("LLM prompt (chunk %d):\n%s", chunk_index, prompt)
            except Exception:
                logger.debug("LLM prompt (chunk %d): <unserializable>", chunk_index)

        for attempt in range(max(1, retries)):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = getattr(response, "content", None)

                # Log raw LLM output in DEBUG (full verbose) mode
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(
                            "LLM raw response (chunk %d):\n%s", chunk_index, content
                        )
                    except Exception:
                        logger.debug(
                            "LLM raw response (chunk %d): <unserializable>", chunk_index
                        )

                if not isinstance(content, str):
                    # If 'content' isn't a string, log the response object for debugging
                    logger.debug(
                        "LLM response (chunk %d) object repr: %r", chunk_index, response
                    )
                    raise TypeError(
                        f"Expected str content from LLM, got {type(content)}"
                    )
                return {
                    "success": True,
                    "extractions": parse_and_align_chunk(content, chunk),
                }
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Chunk %d: attempt %d failed: %s", chunk_index, attempt + 1, e
                )
                if attempt + 1 < retries:
                    sleep_for = retry_backoff * (2**attempt) + random.random() * 0.1
                    time.sleep(sleep_for)
                    continue
                return {
                    "success": False,
                    "error": str(last_exc),
                    "exception": last_exc,
                    "chunk_index": chunk_index,
                }

        # Fallback return to satisfy static analyzers (shouldn't be reached)
        logger.error("Unexpected flow in _process_chunk (chunk_index=%d)", chunk_index)
        return {
            "success": False,
            "error": "internal_error",
            "exception": None,
            "chunk_index": chunk_index,
        }

    all_extractions: List[Extraction] = []
    errors: List[Dict[str, Any]] = []

    if not chunks:
        return AnnotatedDocument(text=text, extractions=[], metadata={})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_map: Dict[Future, Tuple[int, TextChunk]] = {
            executor.submit(_process_chunk, idx, chunk): (idx, chunk)
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures_map):
            mapping = futures_map.get(future)
            chunk_idx = mapping[0] if mapping else None
            try:
                result = future.result()
            except Exception as e:
                logger.exception("Future for chunk raised unexpectedly: %s", e)
                if error_mode == "raise":
                    raise
                errors.append({"chunk_index": chunk_idx, "error": str(e)})
                continue

            if not isinstance(result, dict):
                logger.warning("Worker returned unexpected type: %s", type(result))
                if error_mode == "raise":
                    raise RuntimeError("Unexpected worker result type")
                errors.append(
                    {"chunk_index": chunk_idx, "error": "unexpected result type"}
                )
                continue

            if result.get("success") is True:
                extras = result.get("extractions")
                if isinstance(extras, list):
                    all_extractions.extend(extras)
                else:
                    logger.warning(
                        "Chunk %s reported success but 'extractions' is not a list",
                        chunk_idx,
                    )
                    if error_mode == "raise":
                        raise RuntimeError("'extractions' must be a list")
                    errors.append(
                        {"chunk_index": chunk_idx, "error": "'extractions' not a list"}
                    )
            else:
                err_msg = result.get("error", "unknown error")
                err_chunk_idx = result.get("chunk_index", chunk_idx)
                errors.append({"chunk_index": err_chunk_idx, "error": err_msg})
                if error_mode == "raise":
                    exc = result.get("exception")
                    if isinstance(exc, Exception):
                        raise exc
                    raise RuntimeError(err_msg)

    if dedupe:
        all_extractions = _dedupe_extractions(all_extractions)

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "num_extractions": len(all_extractions),
    }
    if provider_kwargs:
        metadata.update(provider_kwargs)
    if errors:
        metadata["errors"] = errors

    return AnnotatedDocument(text=text, extractions=all_extractions, metadata=metadata)


async def aextract(
    text: str,
    prompt_description: str,
    examples: List[ExampleData],
    model_name: str,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    max_concurrency: Optional[int] = None,
    verbose: int = 0,
    error_mode: str = "return",
    retries: int = 2,
    retry_backoff: float = 1.0,
    dedupe: bool = False,
) -> AnnotatedDocument:
    _configure_verbose_logging(verbose)

    # lazy import
    try:
        from langchain_core.messages import HumanMessage  # type: ignore
    except Exception:
        HumanMessage = None  # type: ignore

    if HumanMessage is None:
        raise RuntimeError(
            "langchain_core is required to call aextract(); install langchain or use a compatible provider."
        )

    llm = get_llm_provider(model_name, provider_kwargs)
    chunks = list(chunk_text(text, chunk_size, chunk_overlap))

    sem = asyncio.Semaphore(max_concurrency or max(1, min(10, len(chunks))))

    async def _process_chunk_async(
        chunk_index: int, chunk: TextChunk
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        try:
            prompt = format_prompt(prompt_description, examples, chunk.text)
        except Exception as e:
            logger.exception(
                "Failed to build prompt for async chunk %d: %s", chunk_index, e
            )
            return {
                "success": False,
                "error": f"prompt_error: {e}",
                "exception": e,
                "chunk_index": chunk_index,
            }

        # Log full prompt in DEBUG (full-verbose) mode
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("LLM prompt (async chunk %d):\n%s", chunk_index, prompt)
            except Exception:
                logger.debug(
                    "LLM prompt (async chunk %d): <unserializable>", chunk_index
                )

        async with sem:
            for attempt in range(max(1, retries)):
                try:
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    content = getattr(response, "content", None)

                    # Log raw LLM output in DEBUG (full verbose) mode
                    if logger.isEnabledFor(logging.DEBUG):
                        try:
                            logger.debug(
                                "LLM raw response (async chunk %d):\n%s",
                                chunk_index,
                                content,
                            )
                        except Exception:
                            logger.debug(
                                "LLM raw response (async chunk %d): <unserializable>",
                                chunk_index,
                            )

                    if not isinstance(content, str):
                        logger.debug(
                            "LLM async response (chunk %d) object repr: %r",
                            chunk_index,
                            response,
                        )
                        raise TypeError(
                            f"Expected str content from LLM, got {type(content)}"
                        )
                    return {
                        "success": True,
                        "extractions": parse_and_align_chunk(content, chunk),
                    }
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "Async chunk %d: attempt %d failed: %s",
                        chunk_index,
                        attempt + 1,
                        e,
                    )
                    if attempt + 1 < retries:
                        await asyncio.sleep(
                            retry_backoff * (2**attempt) + random.random() * 0.1
                        )
                        continue
                    return {
                        "success": False,
                        "error": str(last_exc),
                        "exception": last_exc,
                        "chunk_index": chunk_index,
                    }

        # Fallback return to satisfy static analyzers (shouldn't be reached)
        logger.error(
            "Unexpected flow in _process_chunk_async (chunk_index=%d)", chunk_index
        )
        return {
            "success": False,
            "error": "internal_error",
            "exception": None,
            "chunk_index": chunk_index,
        }

    tasks = [
        asyncio.create_task(_process_chunk_async(idx, chunk))
        for idx, chunk in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)

    all_extractions: List[Extraction] = []
    errors: List[Dict[str, Any]] = []

    for res in results:
        if not isinstance(res, dict):
            logger.warning("Async worker returned unexpected type: %s", type(res))
            if error_mode == "raise":
                raise RuntimeError("Unexpected async worker result type")
            errors.append({"chunk_index": None, "error": "unexpected result type"})
            continue

        if res.get("success") is True:
            extras = res.get("extractions")
            if isinstance(extras, list):
                all_extractions.extend(extras)
            else:
                logger.warning(
                    "Async chunk reported success but 'extractions' is not a list"
                )
                if error_mode == "raise":
                    raise RuntimeError("'extractions' must be a list")
                errors.append(
                    {
                        "chunk_index": res.get("chunk_index"),
                        "error": "'extractions' not a list",
                    }
                )
        else:
            err_msg = res.get("error", "unknown error")
            err_chunk_idx = res.get("chunk_index")
            errors.append({"chunk_index": err_chunk_idx, "error": err_msg})
            if error_mode == "raise":
                exc = res.get("exception")
                if isinstance(exc, Exception):
                    raise exc
                raise RuntimeError(err_msg)

    if dedupe:
        all_extractions = _dedupe_extractions(all_extractions)

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "num_extractions": len(all_extractions),
    }
    if provider_kwargs:
        metadata.update(provider_kwargs)
    if errors:
        metadata["errors"] = errors

    return AnnotatedDocument(text=text, extractions=all_extractions, metadata=metadata)

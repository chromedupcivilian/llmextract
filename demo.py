#!/usr/bin/env python3
"""
Demo script for llmextract.

Usage:
  python demo.py           # default verbosity (INFO-level logs suppressed)
  python demo.py -v        # verbosity = 1 (INFO for llmextract)
  python demo.py -vv       # verbosity = 2 (DEBUG) — shows raw LLM outputs
You can also set LLME_FORCE_SIMULATE=1 in the environment to skip external calls.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llmextract import (
    AnnotatedDocument,
    CharInterval,
    ExampleData,
    Extraction,
    aextract,
    configure_logging,
    extract,
    visualize,
)


def build_provider_kwargs(doc_id: str) -> Dict[str, Any]:
    provider = os.getenv("LLME_PROVIDER")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    ollama_base = os.getenv("OLLAMA_BASE_URL")

    pk: Dict[str, Any] = {}
    if provider:
        pk["provider"] = provider
    if api_key:
        pk["api_key"] = api_key
    if ollama_base:
        pk["ollama_base_url"] = ollama_base

    pk["doc_id"] = doc_id
    return pk


def simulate_extraction(
    doc_id: str,
    model_name: str,
    prompt: str,
    examples: List[ExampleData],
    text: str,
) -> AnnotatedDocument:
    found: List[Extraction] = []

    for ex in examples:
        for example_ex in ex.extractions:
            needle = example_ex.extraction_text or ""
            if not needle.strip():
                continue
            for m in re.finditer(re.escape(needle), text, flags=re.IGNORECASE):
                s, e = m.start(), m.end()
                found.append(
                    Extraction(
                        extraction_class=example_ex.extraction_class,
                        extraction_text=text[s:e],
                        attributes=example_ex.attributes or {},
                        char_interval=CharInterval(start=s, end=e),
                    )
                )

    metadata = {
        "model_name": model_name,
        "simulated": True,
        "doc_id": doc_id,
        "prompt_snippet": (prompt or "")[:200],
    }
    return AnnotatedDocument(text=text, extractions=found, metadata=metadata)


def pretty_print_doc(doc: AnnotatedDocument) -> None:
    print("\n--- Extraction Summary ---")
    meta = doc.metadata or {}
    print(f"Model: {meta.get('model_name')}, doc_id: {meta.get('doc_id')}")
    extras = doc.extractions or []
    print(f"Total extractions: {len(extras)}")
    sorted_extras = sorted(
        extras, key=lambda e: (e.char_interval.start if e.char_interval else -1)
    )
    for ext in sorted_extras:
        interval = (
            f"[{ext.char_interval.start}:{ext.char_interval.end}]"
            if ext.char_interval
            else "[N/A]"
        )
        attrs = ext.attributes or {}
        print(
            f" - {ext.extraction_class}: '{ext.extraction_text}' {interval} attrs={attrs}"
        )


def run_sync_task(
    doc_id: str,
    model_name: str,
    prompt: str,
    examples: List[ExampleData],
    text: str,
    provider_kwargs: Dict[str, Any],
    *,
    simulate_on_fail: bool = True,
    verbose: int = 1,
) -> Optional[AnnotatedDocument]:
    print("\n" + ("-" * 60))
    print(f"Running sync extraction for doc_id='{doc_id}' model='{model_name}'")
    start = time.time()

    try:
        doc = extract(
            text=text,
            prompt_description=prompt,
            examples=examples,
            model_name=model_name,
            provider_kwargs=provider_kwargs,
            chunk_size=int(os.getenv("LLME_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("LLME_CHUNK_OVERLAP", "50")),
            max_workers=int(os.getenv("LLME_MAX_WORKERS", "4")),
            verbose=verbose,
            error_mode="return",
            retries=int(os.getenv("LLME_RETRIES", "2")),
            retry_backoff=float(os.getenv("LLME_RETRY_BACKOFF", "0.5")),
            dedupe=True,
        )
        took = time.time() - start
        print(f"Sync extraction completed in {took:.2f}s")
        pretty_print_doc(doc)
        return doc
    except Exception as e:
        print(f"Sync extraction failed for model='{model_name}' doc='{doc_id}': {e}")
        if simulate_on_fail:
            print("Falling back to simulated extraction (offline).")
            sim = simulate_extraction(doc_id, model_name, prompt, examples, text)
            pretty_print_doc(sim)
            return sim
        return None


def run_async_task(
    doc_id: str,
    model_name: str,
    prompt: str,
    examples: List[ExampleData],
    text: str,
    provider_kwargs: Dict[str, Any],
    *,
    simulate_on_fail: bool = True,
    verbose: int = 1,
) -> Optional[AnnotatedDocument]:
    print("\n" + ("-" * 60))
    print(f"Running async extraction for doc_id='{doc_id}' model='{model_name}'")
    start = time.time()

    async def _run_async():
        return await aextract(
            text=text,
            prompt_description=prompt,
            examples=examples,
            model_name=model_name,
            provider_kwargs=provider_kwargs,
            chunk_size=int(os.getenv("LLME_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("LLME_CHUNK_OVERLAP", "50")),
            max_concurrency=int(os.getenv("LLME_MAX_CONCURRENCY", "4")),
            verbose=verbose,
            error_mode="return",
            retries=int(os.getenv("LLME_RETRIES", "2")),
            retry_backoff=float(os.getenv("LLME_RETRY_BACKOFF", "0.5")),
            dedupe=True,
        )

    try:
        doc = asyncio.run(_run_async())
        took = time.time() - start
        print(f"Async extraction completed in {took:.2f}s")
        pretty_print_doc(doc)
        return doc
    except Exception as e:
        print(f"Async extraction failed for model='{model_name}' doc='{doc_id}': {e}")
        if simulate_on_fail:
            print("Falling back to simulated extraction (offline).")
            sim = simulate_extraction(doc_id, model_name, prompt, examples, text)
            pretty_print_doc(sim)
            return sim
        return None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Demo for llmextract")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=None,
        help="Increase verbosity: -v INFO, -vv DEBUG (raw LLM output)",
    )
    args = parser.parse_args()

    # verbosity: CLI overrides env variable if provided
    env_verbosity = int(os.getenv("LLME_VERBOSE", "1"))
    verbosity = env_verbosity if args.verbose is None else args.verbose
    print(
        f"Starting demo (verbosity={verbosity}). Toggle verbosity with -v / -vv (use -vv to see raw LLM outputs)."
    )
    configure_logging(verbosity)

    force_simulate = bool(int(os.getenv("LLME_FORCE_SIMULATE", "0")))

    models_to_try = os.getenv(
        "LLME_MODELS",
        "mistralai/mistral-7b-instruct:free,qwen3-coder:480b-cloud",
    ).split(",")
    models_to_try = [m.strip() for m in models_to_try if m.strip()]

    fantasy_prompt = "Extract all character names, locations, and important items."
    fantasy_examples = [
        ExampleData(
            text="Garen found the Dragon's Eye in the Cursed Mountains.",
            extractions=[
                Extraction(extraction_class="character", extraction_text="Garen"),
                Extraction(extraction_class="item", extraction_text="Dragon's Eye"),
                Extraction(
                    extraction_class="location", extraction_text="Cursed Mountains"
                ),
            ],
        )
    ]
    fantasy_text = (
        "In Eldoria, Alistair found a manuscript about the Sunstone. "
        "He told Lyra, who agreed to help. Garen later found the Dragon's Eye."
    )

    medical_prompt = "Extract patient names, medications, and conditions."
    medical_examples = [
        ExampleData(
            text="Jane took 20mg of Zoloft for depression.",
            extractions=[
                Extraction(extraction_class="patient", extraction_text="Jane"),
                Extraction(extraction_class="medication", extraction_text="Zoloft"),
                Extraction(extraction_class="condition", extraction_text="depression"),
            ],
        )
    ]
    medical_text = "The patient, John Doe, was prescribed Lisinopril for hypertension. John Doe returns next week."

    tasks = [
        {
            "doc_id": "fantasy_quest",
            "prompt": fantasy_prompt,
            "examples": fantasy_examples,
            "text": fantasy_text,
        },
        {
            "doc_id": "medical_note",
            "prompt": medical_prompt,
            "examples": medical_examples,
            "text": medical_text,
        },
    ]

    all_results: List[AnnotatedDocument] = []

    for task in tasks:
        for model_name in models_to_try:
            doc_id = task["doc_id"]
            provider_kwargs = build_provider_kwargs(doc_id)
            simulate_mode = force_simulate

            # Sync run
            try:
                result_sync = None
                if not simulate_mode:
                    result_sync = run_sync_task(
                        doc_id=doc_id,
                        model_name=model_name,
                        prompt=task["prompt"],
                        examples=task["examples"],
                        text=task["text"],
                        provider_kwargs=provider_kwargs,
                        simulate_on_fail=True,
                        verbose=verbosity,
                    )
                else:
                    print("Force-simulate enabled; running simulated sync extraction.")
                    result_sync = simulate_extraction(
                        doc_id,
                        model_name,
                        task["prompt"],
                        task["examples"],
                        task["text"],
                    )

                if result_sync:
                    all_results.append(result_sync)
            except Exception as e:
                print(f"Error during sync run for model {model_name}: {e}")

            # Async run (separate)
            try:
                result_async = None
                if not simulate_mode:
                    result_async = run_async_task(
                        doc_id=doc_id,
                        model_name=model_name,
                        prompt=task["prompt"],
                        examples=task["examples"],
                        text=task["text"],
                        provider_kwargs=provider_kwargs,
                        simulate_on_fail=True,
                        verbose=verbosity,
                    )
                else:
                    print("Force-simulate enabled; running simulated async extraction.")
                    result_async = simulate_extraction(
                        doc_id,
                        model_name,
                        task["prompt"],
                        task["examples"],
                        task["text"],
                    )

                if result_async:
                    all_results.append(result_async)
            except Exception as e:
                print(f"Error during async run for model {model_name}: {e}")

    if all_results:
        out_file = os.getenv("LLME_OUTPUT_FILE", "llmextract_report.html")
        try:
            html = visualize(all_results)
            with open(out_file, "w", encoding="utf-8") as fh:
                fh.write(html)
            print(f"\n✅ Visualization saved to: {out_file}")
        except Exception as e:
            print(f"Failed to produce visualization: {e}")
    else:
        print("\nNo extraction results to visualize.")


if __name__ == "__main__":
    main()

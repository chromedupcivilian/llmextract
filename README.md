# llmextract

llmextract is a small, pragmatic Python library for extracting structured information
from unstructured text using large language models (LLMs). It is designed to be
stable, easy to integrate, and to produce grounded results (including character
intervals in the source text). The library uses LangChain provider wrappers and
Pydantic models for robust, typed outputs.

Highlights
- Multi-provider support (OpenRouter / OpenAI-compatible and Ollama) via LangChain.
- Structured outputs represented with Pydantic models (Extraction, AnnotatedDocument).
- Source grounding: extractions are aligned to character offsets in the source text.
- Long-document support via chunking with configurable size & overlap.
- Synchronous and asynchronous APIs.
- Self-contained HTML visualization of extraction results.

Table of contents
- Features
- Installation
- Environment variables
- Quick start (sync)
- Quick start (async)
- Visualization
- Demo CLI
- Advanced usage and options
- API reference
- Data models
- Troubleshooting & tips
- Development / testing
- License

---

## Features

- Multi-provider support: OpenRouter/OpenAI-compatible and Ollama.
- Robust parsing and alignment with fallbacks and logging.
- Configurable chunking and concurrency for large documents.
- Logging at three verbosity levels; raw prompts and raw LLM outputs are visible at full verbosity (-vv).
- Simple HTML visualization that shows extractions in context, with tooltips and a downloadable JSON export.

---

## Installation

Recommended: install from PyPI:

```bash
pip install llmextract
```

Core runtime dependencies (declared in pyproject.toml):
- python >= 3.10
- pydantic >= 2.0
- langchain, langchain-openai, langchain-ollama (provider integrations)
- python-dotenv (for the demo)

Optional (recommended for development / improved matching):
- tenacity (better retry primitives)
- rapidfuzz (faster / more accurate fuzzy matching)
- pytest, pytest-asyncio, ruff (development & testing)

You can add optional packages to your project or to `pyproject.toml` extras.

---

## Environment variables

llmextract reads provider credentials and demo controls from environment variables:

- `OPENROUTER_API_KEY` — API key used for OpenRouter / OpenAI-compatible providers.
- `OPENAI_API_KEY` — alternative API key environment variable.
- `OLLAMA_BASE_URL` — base URL for a remote or local Ollama server.
- Demo-specific:
  - `LLME_FORCE_SIMULATE=1` — force demo to run in simulation mode (no network).
  - `LLME_MODELS` — comma-separated default models used by the demo.
  - `LLME_VERBOSE` — default verbosity (0,1,2) used by demo if CLI flag is not provided.
  - `LLME_CHUNK_SIZE`, `LLME_CHUNK_OVERLAP`, `LLME_RETRIES`, `LLME_RETRY_BACKOFF`, `LLME_MAX_WORKERS`, `LLME_MAX_CONCURRENCY` — demo tuning variables.

---

## Quick start — synchronous

Minimal synchronous example. This shows how to call `extract()` and read the results.

```python
from dotenv import load_dotenv
from llmextract import extract, ExampleData, Extraction, configure_logging

load_dotenv()
# Optionally configure package-level logging (0=warning,1=info,2=debug)
configure_logging(1)

prompt = "Extract patient names, medications, and conditions."
examples = [
    ExampleData(
        text="Jane took 20mg of Zoloft for depression.",
        extractions=[
            Extraction(extraction_class="patient", extraction_text="Jane"),
            Extraction(extraction_class="medication", extraction_text="Zoloft"),
            Extraction(extraction_class="condition", extraction_text="depression"),
        ],
    )
]

text = "The patient, John Doe, was prescribed Lisinopril for hypertension."

result_doc = extract(
    text=text,
    prompt_description=prompt,
    examples=examples,
    model_name="mistralai/mistral-7b-instruct:free",
    provider_kwargs={"provider": "openrouter", "api_key": "<YOUR_KEY_HERE>"},
    chunk_size=1000,
    chunk_overlap=100,
    verbose=1,          # 0=quiet, 1=info, 2=debug (raw prompt & response)
    error_mode="return",# or "raise"
    retries=2,
    retry_backoff=0.5,
    dedupe=True,
)

for ext in result_doc.extractions:
    print(ext.extraction_class, ":", ext.extraction_text)
    if ext.char_interval:
        print("  interval:", ext.char_interval.start, ext.char_interval.end)
```

Notes:
- `verbose=2` (or calling `configure_logging(2)` / running the demo with `-vv`) will emit the full prompt and raw LLM response at DEBUG level — useful for debugging model behavior.
- If `error_mode="return"` the function collects per-chunk errors into `result_doc.metadata["errors"]` rather than raising immediately.

---

## Quick start — asynchronous

Use the async entry-point `aextract()` for concurrency-friendly environments.

```python
import asyncio
from llmextract import aextract, ExampleData, Extraction

async def run():
    examples = [ ... ]  # same as above
    doc = await aextract(
        text="...",
        prompt_description="...",
        examples=examples,
        model_name="ollama-model",
        provider_kwargs={"provider": "ollama", "ollama_base_url": "http://localhost:11434"},
        max_concurrency=4,
        verbose=2,
    )
    print(len(doc.extractions))

asyncio.run(run())
```

---

## Visualization

`visualize()` produces a self-contained HTML string that you can save and open in a browser.

```python
from llmextract import visualize

html = visualize(result_doc)            # single document
# or visualize([doc1, doc2]) for multiple
with open("report.html", "w", encoding="utf-8") as fh:
    fh.write(html)
```

The generated HTML:
- Lets you select document and model (if multiple were provided).
- Shows a metadata panel and a legend for extraction classes.
- Highlights extracted spans in the text and shows attributes in tooltips.
- Includes a "Download JSON" button with the underlying serialized results.

Security note: the visualization HTML escapes content and sanitizes script-closing sequences, but you should only open generated HTML from trusted sources.

---

## Demo CLI

A ready-to-run demo is provided at `demo.py`. Example usage:

```bash
# run demo with default verbosity (info-level suppressed)
python demo.py

# increase verbosity to INFO
python demo.py -v

# full verbosity (DEBUG): logs full prompt + raw LLM response
python demo.py -vv

# force the demo to work offline (use the built-in simulator)
LLME_FORCE_SIMULATE=1 python demo.py -vv
```

Demo environment variables (optional):
- `LLME_FORCE_SIMULATE=1` — always use the deterministic simulator (no network).
- `LLME_MODELS` — comma-separated models used by the demo.
- `LLME_OUTPUT_FILE` — path for the generated visualization HTML (default `llmextract_report.html`).

---

## Advanced usage & options

When calling `extract()` / `aextract()` you can tune behavior:

- chunking
  - `chunk_size` (default 4000): characters per chunk
  - `chunk_overlap` (default 200)
- concurrency / workers
  - `max_workers` (sync) and `max_concurrency` (async)
- error handling
  - `error_mode`: `"return"` (collect per-chunk errors in metadata) or `"raise"` (raise on first chunk failure)
  - `retries` and `retry_backoff` (simple retry/backoff per chunk)
- `verbose`:
  - 0: minimal (warnings and above)
  - 1: info-level package logs
  - 2: debug — logs full prompt and raw LLM outputs for each chunk (useful for debugging)

Returned `AnnotatedDocument.metadata` will include operational info:
- model_name, chunk_size, chunk_overlap, num_chunks, num_extractions
- provider kwargs merged in (if supplied)
- `errors` key when error_mode="return" and chunks failed

Dedupe: set `dedupe=True` to perform a simple deduplication pass (by extraction class and normalized text).

Provider configuration
- Pass `provider_kwargs` to `extract()` / `aextract()` to control provider and connection details:
  - `provider`: `"openrouter"` or `"ollama"` (if omitted, provider is inferred)
  - `api_key`: for OpenRouter/OpenAI-compatible providers
  - `base_url`: OpenRouter base URL (example: `"https://openrouter.ai/api/v1"`)
  - `ollama_base_url`: base URL for Ollama (default: `"http://localhost:11434"`)
  - `default_headers`: additional headers for OpenRouter/OpenAI-compatible clients

Example provider usage:

```python
result = extract(
    ...,
    model_name="mistralai/mistral-7b-instruct:free",
    provider_kwargs={"provider": "openrouter", "api_key": os.getenv("OPENROUTER_API_KEY")},
)
```

---

## API reference (summary)

- extract(text, prompt_description, examples, model_name, provider_kwargs=None, chunk_size=4000, chunk_overlap=200, max_workers=10, verbose=0, error_mode="return", retries=2, retry_backoff=1.0, dedupe=False) -> AnnotatedDocument

- aextract(text, prompt_description, examples, model_name, provider_kwargs=None, chunk_size=4000, chunk_overlap=200, max_concurrency=None, verbose=0, error_mode="return", retries=2, retry_backoff=1.0, dedupe=False) -> AnnotatedDocument (async)

- visualize(AnnotatedDocument | List[AnnotatedDocument]) -> str (HTML)

- configure_logging(verbosity: int = 0) — convenience to set llmextract package logging level:
  - 0 => WARNING
  - 1 => INFO
  - 2 => DEBUG

---

## Data models

All models are Pydantic v2 models exported from the package:

- Extraction
  - `extraction_class: str` — label or category (e.g., "medication")
  - `extraction_text: str` — the exact extracted substring
  - `attributes: Dict[str, Any]` — optional structured attributes
  - `char_interval: Optional[CharInterval]` — optional character interval (start inclusive, end exclusive)

- CharInterval
  - `start: int` — inclusive start index
  - `end: int` — exclusive end index (must be > start)

- ExampleData
  - `text: str` — example text
  - `extractions: List[Extraction]` — correct extractions for the example

- AnnotatedDocument
  - `text: str` — original document text
  - `extractions: List[Extraction]`
  - `metadata: Dict[str, Any]` — operation metadata (e.g., model_name, chunk sizes, provider info)

---

## Troubleshooting & tips

- Empty visualization dropdown / no results:
  - Confirm you passed one or more AnnotatedDocument(s) to `visualize()` and that each has `text` and `metadata`. The visualizer defaults `doc_id` to `Document_1` and `model_name` to `Unknown_Model` if missing, so check the generated JSON in the HTML (open "View source").
- Debugging model outputs:
  - Use `verbose=2` (or `configure_logging(2)` / demo `-vv`) to see the full prompt and raw LLM output for each chunk.
- Alignment failures:
  - Alignment uses normalized text searches, flexible whitespace regex, and a lightweight fuzzy fallback. If many extractions fail to align, try:
    - Improving examples in prompts to match expected formats
    - Inspecting raw LLM output (verbose=2) for unexpected formatting (e.g., extra prose, missing fields)
- Prompt engineering:
  - Encourage the model to respond strictly with JSON. We instruct models to return only a single JSON object with an "extractions" key, but some models still emit human explanations — use verbose logging to inspect and adjust.

---

## Development & testing

Recommended dev dependencies (add to `project.optional-dependencies` or your dev env):
- ruff (linting)
- pytest & pytest-asyncio (tests)
- rapidfuzz (optional fuzzy matching)
- tenacity (optional retries)

Suggested local workflow:
1. Create a virtual environment.
2. Install dev deps: `pip install -e ".[dev]"` or `pip install ruff pytest pytest-asyncio`.
3. Run tests: `pytest`.
4. Lint: `ruff check .`.

I encourage contributions — open an issue to discuss changes.

---

## Roadmap / future work

- Optional persistence of raw LLM responses to metadata (opt-in).
- Pluggable alignment strategies (configurable fuzzy match engines).
- Additional provider adapters (OpenAI official SDK adapters, other local engines).
- More advanced visualization options (stacked overlapping spans, multiple highlight styles).

---

## License

Apache-2.0 — see the LICENSE file for details.

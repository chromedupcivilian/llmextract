# llmextract/providers.py

import os
import logging
from typing import Any, Dict, Optional

from pydantic import SecretStr

logger = logging.getLogger(__name__)


def get_llm_provider(
    model_name: str, provider_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Factory returning a LangChain Chat model instance.

    provider_kwargs may include:
      - provider: "openrouter" | "ollama" (preferred)
      - api_key, base_url, ollama_base_url, default_headers
    """
    kwargs = provider_kwargs or {}
    provider_hint = kwargs.get("provider")
    is_openrouter_style = "/" in model_name

    # Lazy imports to avoid requiring provider libs at import time
    ChatOpenAI = None
    ChatOllama = None
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI  # type: ignore

        ChatOpenAI = _ChatOpenAI
    except Exception:
        ChatOpenAI = None

    try:
        from langchain_ollama import ChatOllama as _ChatOllama  # type: ignore

        ChatOllama = _ChatOllama
    except Exception:
        ChatOllama = None

    provider = provider_hint
    if not provider:
        if is_openrouter_style and (
            kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        ):
            provider = "openrouter"
        else:
            provider = "ollama"

    provider = str(provider).lower()

    if provider in ("openrouter", "openai", "openai-compatible"):
        if ChatOpenAI is None:
            raise RuntimeError(
                "ChatOpenAI (langchain_openai) is not installed. Install langchain-openai to use this provider."
            )
        api_key = (
            kwargs.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "API key not provided for OpenRouter/OpenAI-compatible provider. "
                "Provide provider_kwargs['api_key'] or set OPENROUTER_API_KEY/OPENAI_API_KEY."
            )
        base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
        headers = kwargs.get("default_headers", {})
        logger.debug("Initializing ChatOpenAI (OpenRouter/OpenAI-compatible) model.")
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url,
            default_headers=headers,
            temperature=0.0,
        )

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError(
                "ChatOllama (langchain_ollama) is not installed. Install langchain-ollama to use Ollama."
            )
        base_url = kwargs.get("ollama_base_url", "http://localhost:11434")
        logger.debug("Initializing ChatOllama model.")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.0)

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: 'openrouter', 'ollama'."
    )

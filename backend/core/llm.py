"""
backend/core/llm.py
LLM client factory — supports both standard and streaming responses.
Uses Google Gemini API (free tier: 9,000 requests/day).
Wraps creation with tenacity retry to survive transient 503s and
brief rate-limit windows without crashing the request.
"""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import get_settings
from .logging import get_logger

logger = get_logger("llm")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "timeout", "connection"))


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _build_llm(
    model: str,
    api_key: str,
    temperature: float,
    streaming: bool = False,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        max_output_tokens=4096,
        streaming=streaming,
    )


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Standard LLM for chain calls. temperature=0 → deterministic RAG answers."""
    settings = get_settings()
    return _build_llm(settings.gemini_model, settings.gemini_api_key, temperature)


def get_streaming_llm() -> ChatGoogleGenerativeAI:
    """Streaming LLM — NOT cached (new instance per request for SSE)."""
    settings = get_settings()
    return _build_llm(settings.gemini_model, settings.gemini_api_key, 0.0, streaming=True)

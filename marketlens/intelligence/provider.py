"""
MarketLens - Provider-Neutral LLM Adapter Layer.

Defines the common LLMProvider interface, standard response containers,
a deterministic MockLLMProvider for offline testing, and a concrete
OpenAICompatibleProvider for production inference.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from marketlens.config import MarketLensSettings, get_secret_or_env, load_settings
from marketlens.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger("marketlens.intelligence.provider")

# Documented pricing per 1M tokens (input_cost_per_token, output_cost_per_token)
KNOWN_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI GPT-4o mini
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o-mini-2024-07-18": (0.15 / 1_000_000, 0.60 / 1_000_000),
    # OpenAI GPT-4o
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-2024-08-06": (2.50 / 1_000_000, 10.00 / 1_000_000),
    # Legacy models
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}


def _mask_api_key(key: Optional[str]) -> str:
    """Mask sensitive API key for safe logging and error reporting."""
    if not key:
        return "<none>"
    if len(key) <= 8:
        return "***"
    return f"{key[:3]}...{key[-4:]}"


def estimate_token_cost(
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """
    Calculate estimated cost in USD based on documented pricing.

    Returns None if model pricing is unknown or token counts are missing.
    Never invents or fabricates costs.
    """
    if not model or input_tokens is None or output_tokens is None:
        return None

    model_clean = model.lower().strip()
    if model_clean in KNOWN_MODEL_PRICING:
        in_price, out_price = KNOWN_MODEL_PRICING[model_clean]
        cost = (input_tokens * in_price) + (output_tokens * out_price)
        return round(cost, 6)

    return None


@dataclass
class LLMResponse:
    """Standardized response container across all LLM providers."""

    content: str
    raw_response: Optional[dict[str, Any]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class defining the provider-neutral LLM interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text or JSON completion given user and system prompts.

        Parameters
        ----------
        prompt : str
            The formatted user prompt.
        system_prompt : Optional[str]
            Optional system instructions.
        temperature : float, default 0.0
            Sampling temperature.
        max_tokens : int, default 2500
            Maximum tokens to generate.

        Returns
        -------
        LLMResponse
            Standardized response object containing content and usage statistics.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is fully configured with required credentials."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the active model name."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name identifier."""


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for offline testing and continuous integration.

    Allows tests to inject fixed JSON responses, custom callback generators,
    simulated latency, or deliberate errors without network calls.
    """

    def __init__(
        self,
        response_content: Optional[Union[str, dict[str, Any]]] = None,
        *,
        response_generator: Optional[Callable[[str], str]] = None,
        model_name: str = "mock-intelligence-v1",
        input_tokens: int = 450,
        output_tokens: int = 350,
        latency_ms: float = 42.0,
        error_to_raise: Optional[Exception] = None,
    ) -> None:
        self.model_name = model_name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.error_to_raise = error_to_raise
        self.response_generator = response_generator
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None
        self.call_count = 0

        if isinstance(response_content, dict):
            self.response_content = json.dumps(response_content)
        else:
            self.response_content = response_content or "{}"

    def set_response(self, content: Union[str, dict[str, Any]]) -> None:
        """Dynamically update mock response content."""
        if isinstance(content, dict):
            self.response_content = json.dumps(content)
        else:
            self.response_content = content

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        **kwargs: Any,
    ) -> LLMResponse:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.call_count += 1

        if self.error_to_raise:
            raise self.error_to_raise

        if self.response_generator:
            content = self.response_generator(prompt)
        else:
            content = self.response_content

        total_tokens = self.input_tokens + self.output_tokens
        return LLMResponse(
            content=content,
            raw_response={"mock": True, "call_count": self.call_count},
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=total_tokens,
            model=self.model_name,
            latency_ms=self.latency_ms,
        )

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.model_name

    def get_provider_name(self) -> str:
        return "mock"


class OpenAICompatibleProvider(LLMProvider):
    """
    Concrete adapter for OpenAI and OpenAI-compatible REST endpoints.

    Implemented using Python's standard library `urllib.request` to avoid
    introducing heavy runtime dependencies. Credentials are read strictly from
    environment variables or settings.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        # Read from explicit param, or env variables / Streamlit secrets bridge
        self.api_key = (
            api_key
            or get_secret_or_env("MARKETLENS_LLM_API_KEY")
            or get_secret_or_env("OPENAI_API_KEY")
        )
        self.model = (
            model
            or get_secret_or_env("MARKETLENS_LLM_MODEL")
            or get_secret_or_env("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        base = (
            api_base
            or get_secret_or_env("MARKETLENS_LLM_API_BASE")
            or get_secret_or_env("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.api_base = base.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "openai_compatible"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.is_available():
            raise LLMAuthenticationError(
                "LLM API credentials are not configured. "
                "Set MARKETLENS_LLM_API_KEY or OPENAI_API_KEY."
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Allow extra kwargs override
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "MarketLens-Intelligence/1.0",
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes, headers=headers, method="POST"
        )

        logger.debug(
            "Dispatching LLM request to %s (model=%s, key=%s)",
            url,
            self.model,
            _mask_api_key(self.api_key),
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

        except urllib.error.HTTPError as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            code = exc.code
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
                err_data = json.loads(err_body)
                err_msg = err_data.get("error", {}).get("message", err_body)
            except Exception:
                err_msg = str(exc)

            if code == 401 or code == 403:
                raise LLMAuthenticationError(
                    f"LLM authentication failed (status {code}): {err_msg}"
                ) from exc
            if code == 429:
                raise LLMRateLimitError(
                    f"LLM rate limit or quota exceeded (status 429): {err_msg}"
                ) from exc
            raise LLMError(
                f"LLM API request failed with status {code}: {err_msg}"
            ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:
            if "timed out" in str(exc).lower() or isinstance(exc, TimeoutError):
                raise LLMTimeoutError(
                    f"LLM request timed out after {self.timeout_seconds}s"
                ) from exc
            raise LLMError(f"LLM network communication error: {exc}") from exc

        except Exception as exc:
            raise LLMError(
                f"Unexpected error communicating with LLM provider: {exc}"
            ) from exc

        choices = resp_json.get("choices", [])
        if not choices:
            raise LLMError("LLM response contained no choices")

        content = choices[0].get("message", {}).get("content", "")
        usage = resp_json.get("usage", {})
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        tot_tok = usage.get("total_tokens")

        return LLMResponse(
            content=content,
            raw_response=resp_json,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            model=resp_json.get("model", self.model),
            latency_ms=latency_ms,
        )


def create_llm_provider(
    settings: Optional[MarketLensSettings] = None,
    **overrides: Any,
) -> Optional[LLMProvider]:
    """
    Factory function to initialize configured LLM provider.

    Supports OS environment variables and Streamlit Community Cloud secrets.
    Returns None if no provider is configured, enabling graceful offline operation.
    """
    cfg = settings or load_settings()

    explicit_provider = (
        overrides.get("llm_provider")
        or cfg.llm_provider
        or get_secret_or_env("MARKETLENS_LLM_PROVIDER")
    )

    # If provider is not explicitly set, check if an OpenAI key exists in env/secrets
    if not explicit_provider:
        detected_key = (
            overrides.get("llm_api_key")
            or cfg.llm_api_key
            or get_secret_or_env("MARKETLENS_LLM_API_KEY")
            or get_secret_or_env("OPENAI_API_KEY")
        )
        if detected_key:
            provider_name = "openai"
        else:
            provider_name = None
    else:
        provider_name = explicit_provider

    if not provider_name or str(provider_name).strip().lower() in (
        "",
        "none",
        "null",
        "disabled",
    ):
        logger.info(
            "No LLM provider configured; operating in offline/unavailable mode."
        )
        return None

    prov = str(provider_name).strip().lower()

    if prov == "mock":
        return MockLLMProvider(
            model_name=overrides.get("llm_model") or cfg.llm_model or "mock-model",
            **{
                k: v
                for k, v in overrides.items()
                if k not in ("llm_provider", "llm_model")
            },
        )

    if prov in ("openai", "openai_compatible", "azure", "vllm", "ollama"):
        return OpenAICompatibleProvider(
            api_key=overrides.get("llm_api_key")
            or cfg.llm_api_key
            or get_secret_or_env("MARKETLENS_LLM_API_KEY")
            or get_secret_or_env("OPENAI_API_KEY"),
            model=overrides.get("llm_model")
            or cfg.llm_model
            or get_secret_or_env("MARKETLENS_LLM_MODEL")
            or get_secret_or_env("OPENAI_MODEL"),
            api_base=overrides.get("llm_api_base")
            or cfg.llm_api_base
            or get_secret_or_env("MARKETLENS_LLM_API_BASE")
            or get_secret_or_env("OPENAI_BASE_URL"),
            timeout_seconds=overrides.get("llm_timeout_seconds")
            or cfg.llm_timeout_seconds,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider_name}'. "
        "Supported: 'mock', 'openai', 'openai_compatible'."
    )

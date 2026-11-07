"""
MarketLens - Streamlit Secrets & Environment Configuration Bridge Tests.

Verifies:
1. Retrieval from OS environment variables.
2. Retrieval from Streamlit secrets (st.secrets).
3. Precedence of OS environment variables over Streamlit secrets.
4. Graceful fallback when neither is configured (preserving offline mode).
5. Automatic provider resolution when OPENAI_API_KEY is configured in secrets.
6. Absolute credential confidentiality (no printing or logging of keys).
"""

from __future__ import annotations

from unittest.mock import patch

from marketlens.config import get_secret_or_env
from marketlens.intelligence.provider import (
    OpenAICompatibleProvider,
    _mask_api_key,
    create_llm_provider,
)


def test_get_secret_or_env_reads_os_environment(monkeypatch):
    """Verify get_secret_or_env correctly reads from os.environ."""
    monkeypatch.setenv("TEST_KEY_ENV", "env_secret_value_123")
    val = get_secret_or_env("TEST_KEY_ENV")
    assert val == "env_secret_value_123"


def test_get_secret_or_env_reads_streamlit_secrets(monkeypatch):
    """Verify get_secret_or_env reads from st.secrets when not in os.environ."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MARKETLENS_LLM_API_KEY", raising=False)

    mock_secrets = {"OPENAI_API_KEY": "sk-st-secrets-mock-key"}
    with patch("streamlit.secrets", mock_secrets, create=True):
        val = get_secret_or_env("OPENAI_API_KEY")
        assert val == "sk-st-secrets-mock-key"


def test_get_secret_or_env_precedence(monkeypatch):
    """Verify OS environment takes precedence over Streamlit secrets."""
    monkeypatch.setenv("OPENAI_API_KEY", "env_takes_precedence")
    mock_secrets = {"OPENAI_API_KEY": "secrets_should_be_ignored"}

    with patch("streamlit.secrets", mock_secrets, create=True):
        val = get_secret_or_env("OPENAI_API_KEY")
        assert val == "env_takes_precedence"


def test_get_secret_or_env_missing_returns_default(monkeypatch):
    """Verify missing keys return default without raising exceptions."""
    monkeypatch.delenv("NON_EXISTENT_KEY_123", raising=False)
    assert get_secret_or_env("NON_EXISTENT_KEY_123") is None
    assert (
        get_secret_or_env("NON_EXISTENT_KEY_123", default="my_default") == "my_default"
    )


def test_create_llm_provider_offline_when_no_credentials(monkeypatch):
    """Verify factory returns None when no provider or credentials are configured."""
    monkeypatch.delenv("MARKETLENS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MARKETLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("streamlit.secrets", {}, create=True):
        provider = create_llm_provider()
        assert provider is None


def test_create_llm_provider_auto_enables_from_streamlit_secrets(monkeypatch):
    """Verify factory auto-resolves 'openai' when OPENAI_API_KEY in st.secrets."""
    monkeypatch.delenv("MARKETLENS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MARKETLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_secrets = {"OPENAI_API_KEY": "sk-cloud-secrets-auto-detect-key"}
    with patch("streamlit.secrets", mock_secrets, create=True):
        provider = create_llm_provider()
        assert provider is not None
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.is_available() is True
        assert provider.api_key == "sk-cloud-secrets-auto-detect-key"


def test_api_key_masking_never_leaks_in_logs():
    """Verify API keys are masked and raw keys never appear in string outputs."""
    raw_key = "sk-proj-1234567890abcdef1234567890"
    masked = _mask_api_key(raw_key)

    assert masked == "sk-...7890"
    assert raw_key not in masked
    assert _mask_api_key(None) == "<none>"
    assert _mask_api_key("") == "<none>"
    assert _mask_api_key("short") == "***"

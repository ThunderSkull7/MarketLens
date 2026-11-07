"""
MarketLens - Centralised Configuration.

Environment-based settings with clear defaults for local development.
All configuration values are validated on access.

Environment variables are read with the ``MARKETLENS_`` prefix
(e.g. ``MARKETLENS_DATA_DIR``). A ``.env`` file in the project root is
loaded automatically (see ``.env.example``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class MarketLensSettings(BaseSettings):
    """Pydantic-based settings with environment variable support."""

    # Data directory configuration
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")
    raw_data_dir: Path = Field(
        default=Path("./data/raw"),
        description="Raw input data directory",
    )
    processed_data_dir: Path = Field(
        default=Path("./data/processed"),
        description="Processed ingestion output directory",
    )

    # Logging configuration
    logging_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(
        default=None,
        description="Optional log file path; if None, logs to stderr",
    )

    # CSV ingestion configuration
    max_upload_size_mb: float = Field(
        default=100.0,
        description="Maximum allowed CSV file size in megabytes",
    )
    max_accepted_rows: int = Field(
        default=100_000,
        description="Maximum number of rows to accept in ingestion",
    )

    # Environment
    environment: str = Field(default="development", description="Environment name")

    # LLM Intelligence configuration
    llm_provider: Optional[str] = Field(
        default=None,
        description="LLM provider name ('mock', 'openai', etc.); None disables LLM",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model identifier (e.g. 'gpt-4o-mini')",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="LLM API key (read exclusively from environment variables)",
    )
    llm_api_base: Optional[str] = Field(
        default=None,
        description="Optional API base URL for custom or local endpoints",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="LLM generation temperature (0.0 for deterministic)",
    )
    llm_max_tokens: int = Field(
        default=2500,
        description="Maximum tokens for LLM generation",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for LLM API requests in seconds",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Maximum retries for transient LLM errors",
    )
    llm_retry_backoff_factor: float = Field(
        default=1.5,
        description="Backoff factor for transient retry backoff",
    )
    llm_max_context_reviews: int = Field(
        default=30,
        description="Maximum number of representative reviews to include in prompt",
    )
    llm_max_review_chars: int = Field(
        default=300,
        description="Maximum characters per review excerpt in prompt context",
    )
    reports_dir: Path = Field(
        default=Path("./reports"),
        description="Directory for generated reports and intelligence artifacts",
    )

    # Configure settings source - pydantic v2 SettingsConfigDict
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MARKETLENS_",
        extra="ignore",
    )

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        if v.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Logging level must be one of {sorted(_VALID_LOG_LEVELS)}"
            )
        return v.upper()

    @field_validator("max_upload_size_mb")
    @classmethod
    def validate_max_upload_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_upload_size_mb must be positive")
        return v

    @field_validator("max_accepted_rows")
    @classmethod
    def validate_max_accepted_rows(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_accepted_rows must be positive")
        return v


def load_settings() -> MarketLensSettings:
    """Load and validate project settings from the environment."""
    return MarketLensSettings()


def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve a configuration value from OS environment or Streamlit secrets.

    Precedence:
    1. OS environment variable (os.environ[key])
    2. Streamlit secrets (st.secrets[key] or st.secrets.get(key) if available)
    3. default value

    Credentials are never printed, logged, or serialized to disk.
    """
    import os

    val = os.environ.get(key)
    if val is not None and str(val).strip() != "":
        return str(val).strip()

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if key in st.secrets:
                secret_val = st.secrets[key]
                if secret_val is not None and str(secret_val).strip() != "":
                    return str(secret_val).strip()
    except Exception:
        # StreamlitSecretNotFoundError, FileNotFoundError, or uninitialized runtime
        pass

    return default


# Module-level default instance for local development; call load_settings()
# for an explicitly validated instance in application code.
settings = load_settings()

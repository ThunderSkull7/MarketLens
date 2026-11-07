"""
Custom exceptions for the MarketLens ingestion pipeline.

Each exception provides clear error messages and preserves context
for downstream handling and logging.
"""

from __future__ import annotations

from typing import Any, Optional


class MarketLensError(Exception):
    """Base exception for all MarketLens-related errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} [details: {self.details}]"
        return self.message


class UnsupportedFileTypeError(MarketLensError):
    """Raised when an unsupported file type is encountered."""

    def __init__(self, file_extension: str, supported: list[str] | None = None) -> None:
        self.file_extension = file_extension
        supported_str = ", ".join(supported) if supported else "unknown"
        message = (
            f"Unsupported file type: '{file_extension}'. "
            f"Supported types: {supported_str}"
        )
        super().__init__(
            message,
            {"file_extension": file_extension, "supported": supported},
        )


class InvalidCSVError(MarketLensError):
    """Raised when a CSV file is malformed or cannot be parsed."""

    def __init__(
        self, message: str, original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.original_error = original_error


class MissingRequiredColumnsError(MarketLensError):
    """Raised when required canonical columns are missing from the dataset."""

    def __init__(self, missing: list[str], available: list[str]) -> None:
        self.missing = missing
        self.available = available
        missing_str = ", ".join(missing)
        available_str = ", ".join(available[:5])
        if len(available) > 5:
            available_str += "..."
        message = (
            f"Missing required columns: {missing_str}. "
            f"Available columns: {available_str}"
        )
        super().__init__(message, {"missing": missing, "available": available})


class AmbiguousColumnMappingError(MarketLensError):
    """Raised when column mapping is ambiguous and cannot be resolved."""

    def __init__(self, column: str, candidates: list[str]) -> None:
        self.column = column
        self.candidates = candidates
        candidates_str = ", ".join(candidates)
        message = (
            f"Ambiguous mapping for column '{column}': "
            f"found candidates: {candidates_str}"
        )
        super().__init__(message, {"column": column, "candidates": candidates})


class FileSizeLimitExceededError(MarketLensError):
    """Raised when the uploaded file exceeds the maximum allowed size."""

    def __init__(self, file_size_mb: float, max_size_mb: float) -> None:
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        message = (
            f"File size ({file_size_mb:.2f} MB) exceeds maximum limit "
            f"of {max_size_mb:.2f} MB"
        )
        super().__init__(
            message, {"file_size_mb": file_size_mb, "max_size_mb": max_size_mb}
        )


class DataValidationError(MarketLensError):
    """Raised when data validation fails for one or more records."""

    def __init__(
        self,
        message: str,
        rejected_count: int = 0,
        flagged_count: int = 0,
        accepted_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.rejected_count = rejected_count
        self.flagged_count = flagged_count
        self.accepted_count = accepted_count


# ---------------------------------------------------------------------------
# LLM Intelligence Layer Exceptions
# ---------------------------------------------------------------------------


class LLMError(MarketLensError):
    """Base exception for LLM intelligence layer errors."""


class LLMUnavailableError(LLMError):
    """Raised when no LLM provider is configured or available."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM API credentials are missing, invalid, or unauthorized."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider responds with rate limit or quota exceeded."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider request times out."""


class LLMResponseValidationError(LLMError):
    """Raised when LLM output is malformed or fails schema validation."""

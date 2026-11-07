"""MarketLens - Marketing Intelligence Platform package."""

from marketlens.config import load_settings, settings
from marketlens.exceptions import (
    AmbiguousColumnMappingError,
    DataValidationError,
    FileSizeLimitExceededError,
    InvalidCSVError,
    MarketLensError,
    MissingRequiredColumnsError,
    UnsupportedFileTypeError,
)
from marketlens.ingestion import ingest

__all__ = [
    "settings",
    "load_settings",
    "ingest",
    "MarketLensError",
    "UnsupportedFileTypeError",
    "InvalidCSVError",
    "MissingRequiredColumnsError",
    "AmbiguousColumnMappingError",
    "FileSizeLimitExceededError",
    "DataValidationError",
]

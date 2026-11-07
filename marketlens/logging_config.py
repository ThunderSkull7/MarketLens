"""
MarketLens - Logging Configuration.

Provides structured application logging with configurable levels
and output destinations. Follows the principle of logging meaningful
information without exposing private data.

Log levels:
- DEBUG: Detailed diagnostic information (development only)
- INFO: Successful ingestion progress, key milestones
- WARNING: Recoverable data-quality problems
- ERROR: Failed operations, unexpected conditions
- CRITICAL: Severe errors requiring immediate attention
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from marketlens.config import settings


def _setup_logger() -> logging.Logger:
    """Set up and configure the MarketLens logger."""

    logger_name = "marketlens"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()  # Avoid duplicate handlers

    # Set level from settings
    level = getattr(logging, settings.logging_level, logging.INFO)
    logger.setLevel(level)

    # Console handler with format
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler
    if settings.log_file is not None:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger


# Module-level logger instance
logger = _setup_logger()


# Convenience functions for common logging patterns
def info(msg: str, *args, **kwargs) -> None:
    """Log at INFO level with module context."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log at WARNING level."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log at ERROR level."""
    logger.error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs) -> None:
    """Log at DEBUG level."""
    logger.debug(msg, *args, **kwargs)


def exception(msg: str, *args, exc_info: bool = True, **kwargs) -> None:
    """Log at ERROR level with exception information."""
    logger.exception(msg, *args, exc_info=exc_info, **kwargs)

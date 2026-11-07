"""
MarketLens - Intelligence Report Persistence.

Provides serialization and deserialization for IntelligenceReport models,
persisting structured JSON outputs separately from datasets and trained model artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from marketlens.intelligence.schemas import IntelligenceReport

logger = logging.getLogger("marketlens.intelligence.persistence")


def save_intelligence_report(
    report: IntelligenceReport,
    output_dir: Union[str, Path] = "reports",
    filename: Optional[str] = None,
) -> Path:
    """
    Serialize and save an IntelligenceReport to disk as structured JSON.

    Parameters
    ----------
    report : IntelligenceReport
        The validated intelligence report instance.
    output_dir : Union[str, Path], default "reports"
        Target directory for intelligence reports.
    filename : Optional[str]
        Optional custom filename; defaults to 'intelligence_report_<ts>.json'.

    Returns
    -------
    Path
        Full path to the saved JSON file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligence_report_{ts}.json"
    else:
        # Strip directory components to prevent path traversal
        filename = Path(filename).name

    file_path = out_path / filename

    # Use Pydantic's model_dump(mode="json") for clean serializability
    report_dict = report.model_dump(mode="json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    logger.info("Saved intelligence report to: %s", file_path)
    return file_path


def load_intelligence_report(filepath: Union[str, Path]) -> IntelligenceReport:
    """
    Load and validate an IntelligenceReport from a saved JSON file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the saved report JSON file.

    Returns
    -------
    IntelligenceReport
        Reconstituted, fully validated IntelligenceReport instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Intelligence report not found at: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    report = IntelligenceReport.model_validate_json(content)
    logger.info("Loaded intelligence report %s from %s", report.report_id, path)
    return report

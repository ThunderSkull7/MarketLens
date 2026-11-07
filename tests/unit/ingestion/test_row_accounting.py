"""Unit tests for row-accounting metadata and max_rows handling."""

from __future__ import annotations

from marketlens.ingestion import ingest


def test_row_accounting_full_file():
    """Test row-accounting when reading the full source file without limits."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    assert result.total_source_rows == 20
    assert result.rows_parsed == 20
    assert result.rows_omitted_limit == 0
    assert result.total_rows == 20

    accounting = result.row_accounting
    assert accounting["total_source_rows"] == 20
    assert accounting["rows_parsed"] == 20
    assert accounting["rows_omitted_limit"] == 0
    assert accounting["rows_accepted"] == result.accepted_count
    assert accounting["rows_flagged"] == result.flagged_count
    assert accounting["rows_rejected"] == result.rejected_count


def test_row_accounting_with_max_rows():
    """Test reporting omitted rows when max_rows is configured."""
    max_limit = 5
    result = ingest("data/raw/synthetic_feedback_demo.csv", max_rows=max_limit)

    assert result.total_source_rows == 20
    assert result.rows_parsed == 5
    assert result.rows_omitted_limit == 15
    assert result.total_rows == 5

    accounting = result.row_accounting
    assert accounting["total_source_rows"] == 20
    assert accounting["rows_parsed"] == 5
    assert accounting["rows_omitted_limit"] == 15

    # Check summary output reflects omitted rows
    summary_text = result.summary()
    assert "Rows omitted due to limit: 15" in summary_text
    assert "Total rows in source file: 20" in summary_text


def test_row_accounting_synthetic_file_with_multiline_quotes(tmp_path):
    """Test accurate line counting when fields contain quoted newlines."""
    csv_file = tmp_path / "multiline.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("review_id,review_text,rating\n")
        f.write('R001,"This review has\nmultiple lines\nof text.",5.0\n')
        f.write('R002,"Another\nmultiline review",4.0\n')
        f.write('R003,"Single line review",3.0\n')

    # Total records in file = 3
    result = ingest(csv_file, max_rows=2)
    assert result.total_source_rows == 3
    assert result.rows_parsed == 2
    assert result.rows_omitted_limit == 1

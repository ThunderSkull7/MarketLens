# MarketLens - Data Documentation

## Dataset Provenance

This project uses a synthetic demonstration dataset for development and testing.
Real customer feedback datasets will be integrated in subsequent batches.

### Selected Dataset for Development

**Name**: Synthetic Customer Feedback Dataset
**Source**: Generated demonstration data for MarketLens
**Licence**: CC0 (Public domain) - free for any use
**Retrieval date**: 2026-11-05

### Intended Usage

This dataset is specifically designed for:
- CSV ingestion pipeline testing
- Column mapping validation
- Data quality report generation
- Schema validation exercises

It is **not** derived from real customer feedback and any resemblance to
actual customer data is purely coincidental.

### Synthetic Dataset Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| review_id | Yes | string | Unique identifier for each feedback entry |
| review_text | Yes | string | The customer feedback text |
| rating | No | float | Numerical rating (1-5 scale) |
| created_at | No | datetime | Timestamp when feedback was created |
| product_or_service | No | string | Product or service category |
| source | No | string | Data provenance information |
| language | No | string | Language code (e.g., 'en', 'fr') |
| verified_label | No | string | Verified purchaser flag |

### Expected Storage Location

```
data/raw/synthetic_feedback_demo.csv
```

The synthetic dataset is included in the repository under `data/raw/` for
development and testing purposes only.

### Demonstration Dataset

A small demonstration CSV file is provided at `data/raw/synthetic_feedback_demo.csv`
containing 20 sample records covering various scenarios including:
- Valid and invalid ratings
- Duplicate and unique review IDs
- Various data quality scenarios

This file is **not** intended for real-world analysis and should not be used
as a basis for business decisions.

### Obtaining Real Datasets (For Future Batches)

When integrating real datasets, the following sources are recommended:

1. **Amazon Reviews** - Available at Amazon Research Datasets portal
   - Licence: Varies by dataset; check specific terms
   - Note: Requires AWS account for some datasets

2. **Yelp Open Dataset** - Available at Yelp Dataset Dataset site
   - Licence: Creative Commons BY-NC-SA 4.0
   - Note: Intended for research and non-commercial use

3. **UCI Machine Learning Repository** - Various review datasets
   - Licence: Varies by specific dataset

### Dataset Acquisition Instructions

Real datasets should be:
1. Downloaded from the official source repository
2. Verified for licensing compatibility
3. Placed in `data/raw/` with appropriate naming
4. Referenced in `data/README.md` with source, licence, and retrieval date
5. **Never committed to Git** if they contain private customer data
6. Large datasets should be stored externally and downloaded in CI/CD as needed

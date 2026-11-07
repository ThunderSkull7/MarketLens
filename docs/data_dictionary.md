# MarketLens Data Dictionary — Canonical Fields

*Last updated: 2026-11-07 — Batch 3 (Sentiment Analysis & Topic Discovery)*

Every ingested dataset is mapped onto these canonical fields. Unknown source
columns are preserved in both ingestion records and cleaned outputs.

## Canonical Ingestion Fields

### review_id (required)

| Property | Value |
|---|---|
| Purpose | Stable feedback identifier |
| Type | `string` (non-null, non-empty) |
| Common aliases | `id`, `feedback_id`, `comment_id` |
| Validation rules | Missing/blank value ⇒ **rejected**; duplicated value ⇒ **flagged** |

### review_text (required)

| Property | Value |
|---|---|
| Purpose | Customer feedback content |
| Type | `string` (non-null, non-empty) |
| Common aliases | `review`, `feedback`, `comment`, `description` |
| Validation rules | Missing/blank ⇒ **rejected**; very short (< 3 chars) ⇒ **flagged** |
| Cleaning operations | Whitespace collapsed, line endings normalised (`\n`), punctuation and emojis preserved |

### rating (optional)

| Property | Value |
|---|---|
| Purpose | Numeric rating attached to the feedback |
| Type | `float` |
| Common aliases | `score`, `stars`, `ratings`, `value` |
| Validation rules | Non-numeric or out-of-scale ⇒ flagged/counted; scale is configurable (1–5, 1–10, 1–3, 0–1 thumbs); default 1–5 |
| Cleaning operations | Scale-aware normalisation; `clamp=False` by default (out-of-scale ratings become `None` and flagged as `rating_out_of_scale` rather than silently distorted) |

### created_at (optional)

| Property | Value |
|---|---|
| Purpose | Feedback timestamp |
| Type | `date` / ISO datetime |
| Common aliases | `date`, `timestamp`, `created`, `time` |
| Validation rules | Unparseable date ⇒ counted as invalid date |
| Cleaning operations | Standardised to `datetime.date`; invalid dates converted to `None` without inventing sentinels |

### product_or_service (optional)

| Property | Value |
|---|---|
| Purpose | Product or service category |
| Type | `string` |
| Common aliases | `product`, `category`, `item`, `sku` |
| Cleaning operations | Whitespace trimmed; blank strings converted to `None` |

### source (optional)

| Property | Value |
|---|---|
| Purpose | Data provenance (channel/platform) |
| Type | `string` |
| Common aliases | `origin`, `provider`, `provider_name` |
| Cleaning operations | Whitespace trimmed; blank strings converted to `None` |

### language (optional)

| Property | Value |
|---|---|
| Purpose | Language code of the feedback |
| Type | `string` (e.g. `en`, `fr`) |
| Common aliases | `lang`, `language_code` |
| Cleaning operations | Lowercased, whitespace trimmed; blank strings converted to `None` |

### verified_label (optional)

| Property | Value |
|---|---|
| Purpose | Optional verified-purchase / verified-sentiment label |
| Type | `string` |
| Common aliases | `verified`, `verification` |
| Cleaning operations | Lowercased, whitespace trimmed |

---

## Traceability & Cleaning Metadata

| Field | Type | Description |
|---|---|---|
| `_row_index` | `int` | Original row index from source CSV for 100% data traceability |
| `_status` | `string` | Classification status (`accepted`, `flagged`, `rejected`) |
| `_reasons` | `list[str]` | Reasons assigned during ingestion validation |
| `is_flagged` | `bool` | True if record required validation attention |
| `is_duplicate` | `bool` | True if duplicate key detected (detection distinct from deletion) |

---

## Row Accounting Metadata

| Field | Type | Description |
|---|---|---|
| `total_source_rows` | `int` or `None` | Total records in the source file, reliably counted with multiline CSV support |
| `rows_parsed` | `int` | Number of rows read by the parser |
| `rows_omitted_limit` | `int` | Rows omitted when `max_rows` is specified |
| `rows_accepted` | `int` | Rows passing all mandatory and quality checks |
| `rows_flagged` | `int` | Rows passing mandatory checks but carrying quality flags |
| `rows_rejected` | `int` | Rows failing mandatory checks (missing ID or text) |

---

## NLP & Machine Learning Fields (Batch 3)

| Field | Type | Values / Range | Description |
|---|---|---|---|
| `sentiment` | `string` | `"positive"`, `"neutral"`, `"negative"` | Weak sentiment label derived from rating via `rating_to_sentiment()` or external annotation. Excluded if rating is missing or invalid. |
| `predicted_sentiment` | `string` | `"positive"`, `"neutral"`, `"negative"` | Output class predicted by the TF-IDF + Logistic Regression supervised classifier. |
| `sentiment_confidence` | `float` | $[0.0, 1.0]$ | Highest predicted class probability ($\max(P(c))$). |
| `sentiment_margin` | `float` | $[0.0, 1.0]$ | Probability margin between the top-1 and top-2 predicted classes. |
| `is_uncertain` | `bool` | `True` / `False` | Uncertainty guard: `True` if `sentiment_margin < 0.15` or `sentiment_confidence < 0.50`. |
| `topic_cluster_id` | `int` | $0, \dots, k-1$ | Cluster ID assigned by unsupervised TF-IDF + K-Means clustering. |
| `topic_label` | `string` | e.g. `"Topic 0: battery, power, charge"` | Descriptive name synthesized from top cluster centroid terms. |
| `topic_keywords` | `list[str]` | e.g. `["battery", "charge", "life"]` | Top terms defining the centroid coordinates of the assigned cluster. |

---

## Ambient Data

| Dataset | Purpose | Where |
|---|---|---|
| `data/raw/synthetic_feedback_demo.csv` | Clearly-labelled **synthetic** demo data (not real feedback) for dev/testing | Committed |
| Real datasets | Documented in `docs/model_evaluation.md` (UCI Sentiment Sentences, CC BY 4.0); never committed | `data/raw/` (local only) |

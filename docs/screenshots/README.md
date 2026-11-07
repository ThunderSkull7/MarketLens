# MarketLens — UI Screenshot Capture Guide

This guide provides instructions for capturing genuine UI screenshots of the Streamlit dashboard for documentation and portfolio presentation.

> **Integrity Notice:** All screenshots must be captured directly from the live application running locally or on Streamlit Community Cloud using the built-in synthetic demo dataset (`data/raw/synthetic_feedback_demo.csv`). Never fabricate mockups or edit rendered output.

---

## Capture Standards

- **Viewport Dimension:** 1920 × 1080 (16:9 Full HD) or 1440 × 900 (16:10).
- **Browser Settings:** Clean browser profile (incognito/private mode), light mode theme, zoom level set to 100%, developer tools closed, no bookmarks bar.
- **Data Source:** Built-in Synthetic Demo Dataset (click *"Load Demo Dataset"* in the sidebar).
- **Target Directory:** `docs/screenshots/`
- **Format:** Lossless `.png` format.

---

## Required Screenshots & Capture Procedures

### 1. `01_overview_dashboard.png`
- **Navigation:** Select **Overview** in sidebar.
- **State:** Demo dataset loaded (20 records).
- **Visible Elements:**
  - Executive KPI cards: Total Records (20), Average Rating (4.15), Discovered Themes (3), Model Uncertainty (10%).
  - Rating Distribution bar chart and Sentiment Split bar chart.
  - Data Quality Indicators and Key Insights summary callouts.

### 2. `02_upload_quality_inspection.png`
- **Navigation:** Select **Data Upload & Quality** in sidebar.
- **State:** Demo dataset loaded.
- **Visible Elements:**
  - CSV file status: `synthetic_feedback_demo.csv` (1.7 KB, 20 rows).
  - Row accounting cards: Accepted (20), Rejected (0), Flagged (0).
  - Detected schema mapping table (canonical field mapping).
  - Cleaned dataset preview table.

### 3. `03_marketing_analytics.png`
- **Navigation:** Select **Marketing Analytics** in sidebar.
- **State:** Demo dataset loaded.
- **Visible Elements:**
  - Filtering controls (rating sliders, date filters).
  - Rating Dynamics KPI cards and Estimated NPS Proxy.
  - Feedback Volume Temporal Trendline.
  - N-gram word frequency distribution chart.

### 4. `04_sentiment_analysis.png`
- **Navigation:** Select **Sentiment Analysis** in sidebar.
- **State:** Model fitted on demo dataset.
- **Visible Elements:**
  - Classifier Architecture info: `TF-IDF + Logistic Regression` with negation-preserving tokenizer.
  - Evaluation metrics: Accuracy, Precision, Recall, Macro-F1.
  - Uncertainty breakdown metric card and distribution histogram.
  - Interactive sentiment inference widget.

### 5. `05_topic_discovery.png`
- **Navigation:** Select **Topic & Theme Discovery** in sidebar.
- **State:** K-Means clustering ($k=3$) fitted.
- **Visible Elements:**
  - Cluster summary cards with top centroid keywords.
  - Silhouette score quality badge.
  - Representative customer quotes for each topic cluster.

### 6. `06_feedback_explorer.png`
- **Navigation:** Select **Customer Feedback Explorer** in sidebar.
- **State:** Full demo dataset browse view.
- **Visible Elements:**
  - Substring search box and facet filters.
  - Paginated customer reviews table with rating and sentiment tags.
  - Expandable review detail card showing metadata.

### 7. `07_ai_marketing_intelligence.png`
- **Navigation:** Select **AI Marketing Intelligence** in sidebar.
- **State:** Configured provider (or mock provider) active.
- **Mandatory Requirement:** Must display the persistent **Data Privacy Notice**:
  > 🔒 **Data Privacy Notice (Provider / Model):** When generating an AI intelligence report, selected customer review excerpts, record identifiers, and aggregate metrics will be transmitted to the configured external LLM provider. **Automated PII removal is not implemented.** ...
- **Visible Elements:**
  - Action bar with *"Generate Intelligence Report"* button.
  - Synthesized Executive Summary, Theme Insights, and Actionable Recommendations with priority and evidence tags.

### 8. `08_reports_and_exports.png`
- **Navigation:** Select **Reports & Export Hub** in sidebar.
- **State:** Generated analytical artifacts ready.
- **Visible Elements:**
  - Five download cards with download buttons:
    1. Cleaned Dataset (`.csv`)
    2. Analytics Summary (`.json`)
    3. Sentiment Predictions (`.csv`)
    4. Topic Discovery Clusters (`.json`)
    5. Executive Intelligence Report (`.json` / `.md`)

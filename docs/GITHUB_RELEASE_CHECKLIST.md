# MarketLens — GitHub Release & Publication Checklist

This checklist provides the exact metadata, repository settings, and operational steps for publishing MarketLens as a public GitHub portfolio project.

---

## 1. Repository Metadata

| Setting | Value |
| :--- | :--- |
| **Repository Name** | `MarketLens` |
| **Description** | `End-to-End Marketing Intelligence Platform combining data analytics, classical NLP, sentiment analysis, topic discovery, and LLM-powered insights.` |
| **Visibility** | Public |
| **License** | MIT License (`LICENSE`) |
| **Default Branch** | `main` |
| **Topics** | `python`, `machine-learning`, `nlp`, `sentiment-analysis`, `topic-modeling`, `streamlit`, `llm`, `marketing-intelligence`, `data-analytics`, `data-quality` |

---

## 2. GitHub "About" Section Configuration

- **Description:** Paste the exact description above.
- **Website / Demo URL:** Link to deployed Streamlit Community Cloud URL (or leave blank until deployed).
- **Include in Home Page:**
  - [x] Releases
  - [x] Packages
  - [ ] Environments (disable unless deploying via GitHub Environments)

---

## 3. Initial Publication Checklist

Prior to running `git push`:

1. **Clean Workspace Audit:**
   - Confirm `.gitignore` excludes `.env`, `*.log`, `*.joblib`, `*.pkl`, `.venv/`, `.pytest_cache/`, and `.ruff_cache/`.
   - Confirm `data/raw/synthetic_feedback_demo.csv` and all `.gitkeep` files are tracked.
2. **Local Test & Lint Sign-off:**
   - [x] `ruff check .` passes with 0 errors.
   - [x] `ruff format --check .` passes (all 64 files formatted).
   - [x] `pytest` passes with 170 passing tests.
3. **First Git Commit:**
   ```bash
   git add .
   git commit -m "feat: initial release of MarketLens platform (Batches 1-6.1)"
   git branch -M main
   git remote add origin https://github.com/<your-username>/MarketLens.git
   git push -u origin main
   ```

---

## 4. Post-Publication CI Verification Checklist

Immediately following the first push:

1. **GitHub Actions Trigger:**
   - Navigate to the **Actions** tab on GitHub.
   - Confirm workflow `.github/workflows/ci.yml` is triggered on branch `main`.
2. **Matrix Validation:**
   - Confirm test matrix runs on both `python-version: "3.11"` and `python-version: "3.12"`.
   - Verify `ruff check .`, `ruff format --check .`, and `pytest` pass remotely on `ubuntu-latest`.
3. **Badge Status:**
   - Once the remote workflow succeeds, update the CI badge in `README.md` if desired.

---

## 5. UI Screenshots Checklist

Follow the capture procedures in [`docs/screenshots/README.md`](screenshots/README.md):
- [ ] `01_overview_dashboard.png`
- [ ] `02_upload_quality_inspection.png`
- [ ] `03_marketing_analytics.png`
- [ ] `04_sentiment_analysis.png`
- [ ] `05_topic_discovery.png`
- [ ] `06_feedback_explorer.png`
- [ ] `07_ai_marketing_intelligence.png` (verifying Data Privacy Notice is visible)
- [ ] `08_reports_and_exports.png`

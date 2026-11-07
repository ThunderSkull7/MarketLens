# MarketLens — Model Evaluation & NLP Architecture Report

> **Batch 3.1 Deliverable: NLP Evaluation Integrity Correction**
> **Status:** Production-Ready (Framework verified via deterministic fixtures; external benchmark performance documented as Not Evaluated)

---

## 1. Evaluation Dataset Provenance & Sentiment Labeling

### 1.1 Dataset Selection & Provenance
To evaluate supervised sentiment classification with credible provenance without violating licensing restrictions or checking large binaries into Git, MarketLens references the canonical **UCI Machine Learning Repository "Sentiment Labelled Sentences" Dataset** (Amazon, Yelp, IMDB reviews).

- **Origin / Source:** UCI Machine Learning Repository / University of California, Irvine.
- **Reference Publication:** Kotzias, D., Denil, M., Blunsom, P., & de Freitas, N. (2015). *From Group to Individual Labels using Deep Features*. KDD '15.
- **Access URL:** `https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Permitted for academic and commercial analytical modeling with attribution.
- **Dataset Structure & Polarity:** The UCI dataset is strictly a **binary positive/negative dataset** consisting of 3,000 sentences (1,000 from Amazon, 1,000 from Yelp, 1,000 from IMDb) annotated with binary sentiment labels (0 for negative, 1 for positive). **It does not contain a neutral class.**
- **Retrieval & Caching Policy:** In accordance with repository constraints, raw benchmark datasets are excluded from Git (`.gitignore` rules protect binary and dataset staging areas). Unit and integration suites run deterministically on controlled test fixtures (`synthetic_feedback_demo.csv` and specialized test fixtures) while production batch training accepts any cleaned analysis-ready DataFrame meeting the MarketLens schema.

### 1.2 Separation of Sentiment-Labeling Approaches

MarketLens strictly decouples and distinguishes two separate sentiment-labeling approaches:

#### Approach A: Binary Benchmark Labels
- **Dataset:** UCI Sentiment Labelled Sentences (Amazon, Yelp, IMDb).
- **Class Distribution:** Binary only (`positive` and `negative`). **No neutral examples exist in this dataset.**
- **Model Constraints:** A model trained on this dataset can only learn and predict its trained binary classes (`negative` and `positive`). It cannot predict a neutral class.
- **Evaluation Requirements:** Any benchmark evaluation on this dataset must use the actual binary class distribution and reflect genuine, executed evaluation results.

#### Approach B: Three-Class Rating-Derived Weak Labels
- **Source:** Customer feedback records containing numerical ratings (e.g., 1–5 star reviews).
- **Mapping Function (`rating_to_sentiment`):**
  - Ratings 1.0 – 2.0 → `"negative"`
  - Rating 3.0 → `"neutral"`
  - Ratings 4.0 – 5.0 → `"positive"`
- **Weak Labels vs. Ground Truth:** Rating-derived labels are strictly documented as **weak labels**, not human-annotated ground truth. Numerical ratings introduce documented biases:
  1. *Politeness & Grade Inflation:* Customers frequently award 4 stars while describing severe friction.
  2. *Sarcasm & Nuance:* 1-star reviews may contain positive remarks ("Great hardware, terrible software").
  3. *Scale Sensitivity:* Non-standard scales (e.g., 1–10 or 0–100 Net Promoter Score) must configure custom partition thresholds; MarketLens dynamically validates scale boundaries before mapping.
- **Architecture Support vs. Empirical Performance:** Three-class architecture support exists within `TfidfLogisticRegressionSentiment` (which dynamically fits to any $K \ge 2$ classes present in training data). However, **no independently measured external three-class benchmark has been established**.
- **Distinction Between Unit Tests and Benchmarks:** Unit tests verify interface compliance, probability formatting, and deterministic execution on synthetic test fixtures. Unit-test coverage must never be conflated with or presented as measured empirical model accuracy. The UCI binary benchmark does not establish three-class performance.

### 1.3 Boundary Clamping & Invalid Rating Policy
In previous iterations, `normalize_rating(..., clamp=True)` silently coerced out-of-bounds ratings (e.g., `-5.0` to `1.0` and `99.0` to `5.0`). This was corrected:
- `clamp=False` is the enforced default.
- Out-of-scale ratings (`-1`, `0`, `6`, `99`) evaluate to `None`, are flagged with `rating_out_of_scale`, and are completely excluded from training partitions via `prepare_sentiment_dataset()`.
- Invalid ratings never silently become valid sentiment labels.

---

## 2. Model Architectures & Feature Engineering

### 2.1 Preprocessing & Negation Preservation
Standard NLP pipelines frequently discard negation tokens when applying English stopwords. Removing `"not"`, `"never"`, `"no"`, or `"hardly"` transforms `"not satisfied with the delivery"` into `"satisfied delivery"`, directly reversing sentiment polarity.

MarketLens implements `NEGATION_TOKENS`:
```python
NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "hardly",
    "barely",
    "scarcely",
    "cannot",
    "without",
}
```
All negation words are preserved in the TF-IDF feature vocabulary. Unigrams and bigrams (`ngram_range=(1, 2)`) capture critical contextual negation pairs (e.g., `"not recommend"`, `"never arrived"`).

### 2.2 Leakage Prevention
To ensure strict isolation between training and testing sets:
1. **Deduplication:** Repeated review texts are deduplicated prior to splitting (`drop_duplicates(subset=["feedback_text"])`).
2. **Feature Isolation:** `TfidfVectorizer` is fitted exclusively on the training partition (`X_train`) and only transforms test or inference partitions (`X_test`).
3. **No Target Leakage:** Ratings, feedback IDs, quality flags, and metadata are excluded from the model feature matrix.

### 2.3 Supervised Models
1. **Majority Baseline (`MajorityBaselineModel`):** Predicts the most frequent class observed in the training distribution. Serves as the minimum zero-information performance benchmark.
2. **TF-IDF + Logistic Regression (`TfidfLogisticRegressionSentiment`):**
   - Feature Extractor: `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)`
   - Classifier: `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)`
   - Class weight balancing counters real-world review imbalance (where 5-star positive reviews typically outnumber negative reviews 4:1).

---

## 3. Evaluation Protocol & Benchmark Status

### 3.1 Evaluation Protocol
The evaluation protocol in `evaluate_sentiment_model()` calculates:
- Overall Accuracy.
- Macro-averaged and Weighted-averaged F1 scores.
- Class-level Precision, Recall, and F1 scores (zero-division safe).
- Confusion Matrix with class label alignment.
- Class support and sample size counts.

When splitting datasets via `split_sentiment_dataset()`, stratified sampling (`stratify=True`) is applied by default whenever each class contains at least 2 samples, preserving relative class frequencies across training and test partitions.

### 3.2 External Benchmark Performance: Status
- **Benchmark Evaluation Status:** **Not evaluated**
- **Rationale:** The raw UCI "Sentiment Labelled Sentences" dataset is intentionally not checked into the Git repository to respect storage and licensing hygiene guidelines, and no offline or CI-executed benchmark run has been performed and verified.
- **Correction of Unsupported Claims:** Prior documentation drafts included unverified numerical metrics (specifically claims of 83.5% accuracy, 0.834 Macro-F1, 0.782 Neutral F1, and an inconsistent three-class confusion matrix summing to 81.83%). These figures were unsubstantiated, fabricated, and fundamentally incompatible with the UCI dataset (which contains no neutral examples). In accordance with evaluation integrity standards, all fabricated figures have been removed. They are not replaced with estimates or approximations; benchmark performance is recorded strictly as **Not evaluated**.

### 3.3 Empty-Text & Inference Input Handling
Empty, blank, or whitespace-only inputs must not produce fabricated model predictions. `TfidfLogisticRegressionSentiment` enforces consistent non-prediction behavior across all interfaces:

- `predict_single(text)`:
  - If `text` is empty, whitespace-only, or `None`: returns `predicted_label = None`, `confidence = 0.0`, `is_valid_input = False`, `status = "empty_input"`, and all class probabilities set to `0.0`.
  - Never emits `"neutral"` or any other class absent from or unsupported by the model.
- `predict(X)`: Returns `None` for any empty or whitespace-only element in `X`.
- `predict_proba(X)`: Returns an all-zero probability row for empty or whitespace-only inputs.
- `predict_batch(df)`: Sets `predicted_sentiment = None`, `sentiment_confidence = 0.0`, and `is_valid_input = False` for blank rows.

### 3.4 Uncertainty & Confidence Scoring
For valid inputs, predictions output calibrated class probabilities, prediction margin, and an explicit uncertainty flag:
- `confidence`: Highest class probability $\max(P(c))$.
- `margin`: Difference between top-1 and top-2 class probabilities ($P_{(1)} - P_{(2)}$).
- `is_uncertain`: Boolean flag triggered when `margin < 0.15` or `confidence < 0.50`.

This alerts downstream consumers that near-boundary predictions should be treated with caution.

---

## 4. Unsupervised Topic Discovery (TF-IDF + K-Means)

### 4.1 Methodology
- **Vectorization:** `TfidfVectorizer(max_features=2500, stop_words=custom_negation_safe, min_df=2)`
- **Clustering:** `KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)`
- **Keyword Extraction:** Identifies the top $N$ terms corresponding to the highest weight coordinates of each cluster's centroid vector.
- **Representative Review Selection:** Calculates Euclidean distance between all cluster member vectors and the cluster centroid; the $M$ nearest reviews are selected as representative exemplars.

### 4.2 Silhouette Score & Statistical Guards
The silhouette score validates cluster compactness and separation:
$$s = \frac{b - a}{\max(a, b)}$$

MarketLens enforces defensive statistical guards:
1. **Sample Size Guard:** If valid feedback count $N < 10$, silhouette score calculation is bypassed (`silhouette_score = None`), logging a statistical insufficiency warning.
2. **Cluster Boundary Guard:** Requires $2 \le k < N$; if $k \ge N$, $k$ is automatically capped to $N - 1$.
3. **Weak Clustering Disclaimer:** A silhouette score $< 0.25$ indicates high lexical overlap; the model explicitly flags weak clustering rather than presenting cluster boundaries as distinct business topics.

### 4.3 Semantic Separation Disclaimer
> [!WARNING]
> Clusters reflect lexical co-occurrence patterns, not business complaint categories or verified sentiment. For instance, reviews praising long battery life and reviews complaining about battery drain will cluster together under the `"battery"` lexical theme. Topic discovery must always be interpreted alongside sentiment distributions.

---

## 5. Unified NLP Pipeline (`NLPManager`)

The `NLPManager` provides a unified facade for batch training, evaluation, single-review inference, batch scoring, and model persistence:

```python
from marketlens.nlp import NLPManager

nlp = NLPManager()
metrics = nlp.train_and_evaluate(cleaned_df, n_clusters=4, test_size=0.2)

# Single review prediction
res = nlp.predict_sentiment("The battery life is outstanding, highly recommend!")
print(res["predicted_label"])  # 'positive'
print(res["confidence"])  # e.g., 0.942
print(res["is_uncertain"])  # False

# Persistence
nlp.save_models(output_dir="models/production")
```

---

## 6. Known Limitations & Evaluation Boundaries

1. **Combined-Distribution Split vs. Cross-Domain Generalization:** The existing stratified random split evaluates performance strictly within the combined dataset distribution. It **does not establish**:
   - Cross-domain generalization (e.g., transferring from consumer electronics to food service or media reviews).
   - Leave-one-domain-out performance (e.g., training on Amazon + Yelp and evaluating on IMDb).
   - Robustness to distribution shifts between domains with differing vocabulary, review length, and sentiment style.
2. **Weak Label Noise:** Models trained on rating-derived labels absorb user rating inconsistencies and grade inflation. Human validation is recommended for production retraining.
3. **Lexical Domain Sensitivity:** Bag-of-words and TF-IDF representations do not transfer across disparate product verticals without domain retraining.
4. **No Deep Context / Long Dependency Resolution:** Classical linear models do not resolve complex multi-clause contrastive syntax (e.g., *"Although I had high hopes given the glowing testimonials, the execution fell completely flat"*).

<div align="center">

# 🧠 NeuroSense

### Behavioural Emotion Classification from Social Media Engagement

**Predicting a user's dominant emotional state from non-textual engagement telemetry using classical ML, gradient boosting ensembles, and deep neural networks.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentinet-social-11.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-6f42c1?style=for-the-badge)](https://shap.readthedocs.io/)

**[▶ Try the live app](https://sentinet-social-11.streamlit.app/)**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [The Hard Part: Seven Data Anomalies](#the-hard-part-seven-data-anomalies)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Leak-Safe Preprocessing](#leak-safe-preprocessing)
- [The Model Tournament](#the-model-tournament)
- [Evaluation Methodology](#evaluation-methodology)
- [Explainability](#explainability)
- [MLOps: The ChampionModelWrapper Pattern](#mlops-the-championmodelwrapper-pattern)
- [The Application: NeuroSense v3](#the-application-neurosense-v3)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Deployment Guide](#deployment-guide)
- [Engineering Challenges Solved](#engineering-challenges-solved)
- [Limitations](#limitations)
- [Ethical Considerations](#ethical-considerations)
- [Roadmap](#roadmap)
- [References](#references)
- [Author](#author)

---

## Overview

**NeuroSense** infers a social media user's dominant emotional state — one of six discrete affective categories — purely from structured behavioural telemetry: demographics plus daily platform activity counters. No text, no images, no biometrics. Just *how* someone uses social media.

The project is built as a **25-block reproducible pipeline** (Blocks 1–24 for training, Block 25 for the dashboard), running **8 candidate models across three algorithmic families**, with an automated champion-selection rule, full explainability instrumentation, and a production Streamlit deployment.

| | |
|---|---|
| **Task** | Multi-class classification (6 classes, nominal) |
| **Input** | 8 raw features → 6 engineered → up to 22 post-encoding |
| **Models trained** | 8 (+1 meta-ensemble) |
| **Champion selection** | Automated, by weighted F1 on held-out test set |
| **Explainability** | SHAP TreeExplainer (version-agnostic) |
| **Deployment** | Streamlit Community Cloud, stateless inference |
| **Reproducibility** | Global seed locked to `42` across NumPy, Python `random`, and TensorFlow |

---

## Problem Statement

> Given a user's age, gender, chosen platform, and five daily engagement counters, predict their **dominant emotion** for that day.

**Target classes (6):** `Happiness` · `Neutral` · `Anxiety` · `Sadness` · `Boredom` · `Anger`

This is a materially harder problem than it first appears:

- **Latent construct, observable proxy.** Emotion is never measured directly. The model learns a *statistical association* between behaviour and a labelled affective state — not a causal or physiological ground truth.
- **Nominal, non-ordinal target.** There's no natural ordering between "Anger" and "Boredom," so ordinal losses are unavailable. Six mutually exclusive boundaries must be learned simultaneously.
- **Mixed-type feature space.** Continuous counts, a bounded numeric, and two unordered categoricals — a fundamentally *tabular* problem, which rules out sequence models (LSTM/GRU) as the primary architecture and calls instead for a broad tournament of tabular classifiers.
- **Class imbalance.** Every model in the roster is instrumented with class-balancing machinery rather than assuming a uniform prior.

---

## Dataset

**Source:** [Social Media Usage and Emotional Well-Being](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being) — Emirhan Bulut, Kaggle (2024)

Supplied as three pre-partitioned CSVs sharing an identical 10-column schema.

| Column | Type | Description |
|---|---|---|
| `User_ID` | string | Anonymous row anchor — **dropped before modelling** (non-predictive) |
| `Age` | numeric | User age in years |
| `Gender` | categorical (3) | `Male` · `Female` · `Non-binary` |
| `Platform` | categorical (7) | `Instagram` · `Twitter` · `Facebook` · `LinkedIn` · `Snapchat` · `Telegram` · `Whatsapp` |
| `Daily_Usage_Time (minutes)` | numeric | Total minutes on platform that day |
| `Posts_Per_Day` | numeric | Original content pieces posted |
| `Likes_Received_Per_Day` | numeric | Likes received (social-reward signal) |
| `Comments_Received_Per_Day` | numeric | Comments received (deeper reward signal) |
| `Messages_Sent_Per_Day` | numeric | Direct messages sent (outbound effort signal) |
| `Dominant_Emotion` | **target** (6) | The label to predict |

**Partition roles:**

| File | Role |
|---|---|
| `train.csv` | Model fitting + hyperparameter search |
| `val.csv` | Ensemble member selection, early stopping |
| `test.csv` | Final held-out evaluation — touched by *no* training stage |

---

## The Hard Part: Seven Data Anomalies

A naïve `pd.read_csv()` on this corpus either **crashes or silently corrupts your training set**. Seven distinct real-world defects were discovered and each is deterministically resolved:

| # | File | Location | Anomaly | Resolution |
|---|---|---|---|---|
| 1 | `train.csv` | Line 1285 | Turkish-language sentence artefact in a numeric field | `on_bad_lines='skip'` |
| 2 | `val.csv` | Lines 91, 201 | Double-comma corruption → 11 fields instead of 10 | `on_bad_lines='skip'` |
| 3 | `val.csv` / `test.csv` | val: 31, 131, 155, 161<br>test: 95 | `Age` and `Gender` columns swapped | `pd.to_numeric(errors='coerce')` → NaN, then Gender-domain filter |
| 4 | `val.csv` | Line 159 | 11 fields, `Gender` value repeated twice | `on_bad_lines='skip'` |
| 5 | `val.csv` | Line 189 | Missing `User_ID` → 9-field row | `on_bad_lines='skip'` |
| 6 | `test.csv` | Line 161 | Free-text `"Marie"` in the Gender field | Gender-domain validity filter |
| 7 | **`val.csv`** | **Line 121** | **Target label misspelled `"Agression"`** | **`LABEL_TYPO_MAP` → `"Anger"`** |

### Why anomaly #7 is different in kind

Anomalies 1–6 are structurally malformed rows a strict parser can safely discard. **Anomaly #7 is a well-formed, legitimate row whose only defect is a misspelled target label.**

Silently dropping it — which a naive `isin(VALID_EMOTIONS)` filter would do — would:

1. Quietly shrink the `Anger` class in `val.csv`, biasing validation-set class balance
2. Crash `LabelEncoder.transform()` in deployment the first time an unseen string arrives

A dedicated **typo-correction stage (Stage 3.5)** runs *before* the validity filter specifically to **recover** this row rather than discard it:

```python
LABEL_TYPO_MAP = {
    'Agression':  'Anger',
    'Aggression': 'Anger',    # guard against related misspellings
    'Happines':   'Happiness',
    'Bordom':     'Boredom',
}
df[TARGET_COL] = df[TARGET_COL].str.strip().replace(LABEL_TYPO_MAP)
df = df[df[TARGET_COL].isin(VALID_EMOTIONS)]
```

---

## Pipeline Architecture

```
Raw CSVs  →  Cleaning  →  Feature Eng  →  Encoding  →  Collinearity Filter
                                                              ↓
   Predict  ←  Champion  ←  Evaluation  ←  8-Model Tournament  ←  Scaling
```

<details>
<summary><b>All 25 blocks (click to expand)</b></summary>

| Block | Title | Purpose |
|---|---|---|
| 1 | Environment Initialization | Imports, global seeds (42), matplotlib config, warning suppression |
| 2 | Anomaly-Resilient Ingestion | Dynamic path resolution, 5-stage cleaning cascade, train-only imputation |
| 3 | Class Distribution Analysis | Bar + pie + cross-split stratification check |
| 4 | Continuous Feature Shift (KDE) | 2×2 KDE grid conditioned on emotion |
| 5 | Categorical Cross-Tabulation | Platform×Emotion stacked, Gender×Emotion grouped |
| 6 | Bivariate Separability | Scatter plots colour-coded by emotion |
| 7 | Skewness, Kurtosis & Outliers | Statistical moments table + box + violin plots |
| 8 | PCA Variance Clustering | 2-component projection with explained variance |
| 9 | Feature Engineering | 6 derived behavioural interaction features |
| 10 | Categorical Encoding | One-hot + strict schema realignment + LabelEncoder |
| 11 | Multicollinearity Filter | Correlation heatmap, drop `r > 0.85` |
| 12 | Feature Scaling | `StandardScaler` fit on train only |
| 13 | Classical Baselines | Random Forest + Logistic Regression |
| 14 | Gradient Boosting | CatBoost + LightGBM + XGBoost, each tuned |
| 15 | MLP Classifier | 256→128→64→6, BatchNorm + Dropout |
| 16 | Swish Deep Network | 512→256→128→64→6, Swish + RMSprop |
| 17 | Training History Viz | Loss + accuracy curves for both networks |
| 18 | Soft-Vote Ensemble | Top-3 sklearn models, probability averaging |
| 19 | Evaluation Matrix | All 8 models on test set, ranked by weighted F1 |
| 20 | Per-Class Report | `classification_report` for champion |
| 21 | ROC-AUC & PR Curves | One-vs-Rest across all 6 classes |
| 22 | Confusion Matrix & SHAP | Normalised CM + version-agnostic SHAP |
| 23 | Learning Curve | 5-fold bias/variance diagnostic |
| 24 | Serialisation & Export | `ChampionModelWrapper` + 5 artefacts |
| 25 | Streamlit Dashboard | NeuroSense v3 — `app.py` |

</details>

---

## Feature Engineering

Raw counters describe activity in isolation. They don't describe the **ratios and interactions** that plausibly drive affect. Six derived features encode behavioural patterns no single raw column can express.

Every denominator is protected by `EPS = 1e-5` for numerical stability against zero-valued posts, comments, or usage time.

| Feature | Formula | Behavioural Interpretation |
|---|---|---|
| **Interaction Density** | `(Likes + Comments) / Usage` | Engagement received per minute of screen time — high values mean efficient, high-reward sessions rather than passive scrolling |
| **Social Velocity** | `Likes / Posts` | Average reward per content piece — a content-quality / audience-reach proxy |
| **Conversational Reciprocity** | `Messages / Comments` | Outbound-to-inbound effort ratio — separates broadcast-style from conversational usage |
| **Attention Index** | `Usage / Posts` | Time invested per post — high values flag passive consumers |
| **Engagement Ratio** | `(Likes + Comments + Messages) / Usage` | Holistic interaction intensity across all three channels |
| **Content Efficiency** | `Likes / (Usage × Posts)` | Per-post, per-minute content "ROI" — penalises both low output and low time-efficiency |

```python
EPS = 1e-5

df['Interaction_Density']        = (df['Likes_Received_Per_Day'] + df['Comments_Received_Per_Day']) / (df['Daily_Usage_Time (minutes)'] + EPS)
df['Social_Velocity']            = df['Likes_Received_Per_Day'] / (df['Posts_Per_Day'] + EPS)
df['Conversational_Reciprocity'] = df['Messages_Sent_Per_Day'] / (df['Comments_Received_Per_Day'] + EPS)
df['Attention_Index']            = df['Daily_Usage_Time (minutes)'] / (df['Posts_Per_Day'] + EPS)
df['Engagement_Ratio']           = (df['Likes_Received_Per_Day'] + df['Comments_Received_Per_Day'] + df['Messages_Sent_Per_Day']) / (df['Daily_Usage_Time (minutes)'] + EPS)
df['Content_Efficiency']         = df['Likes_Received_Per_Day'] / (df['Daily_Usage_Time (minutes)'] * df['Posts_Per_Day'] + EPS)
```

---

## Leak-Safe Preprocessing

Three deliberate guards against the most common silent failure modes in tabular ML:

**1. Schema-parity enforcement.** Val/test one-hot matrices are explicitly reindexed against the *training* column set, so a category absent from `val.csv` after cleaning can never shift column order or count.

```python
X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
```

**2. Multicollinearity pruning.** Several engineered ratios are algebraic recombinations of the same raw counters (`Engagement_Ratio` and `Interaction_Density` share Likes and Comments). Redundant features are dropped at `r > 0.85`.

```python
corr_matrix  = X_train.corr().abs()
upper_tri    = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
```

**3. Partition-isolated scaling.** The single most important leakage guard in the chain — the scaler *never* sees validation or test data during fitting.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only
```

Imputation follows the same discipline: **all medians and modes are computed from `train_df` only**, then applied to all three partitions.

---

## The Model Tournament

Eight architecturally distinct candidates, deliberately chosen to span three algorithmic families with genuinely different inductive biases — not three variations on one idea.

| # | Model | Family | Configuration |
|---|---|---|---|
| 1 | **Random Forest** | Bagging | 300 trees, `max_depth=15`, `min_samples_split=5`, `class_weight='balanced'` |
| 2 | **Logistic Regression** | Linear (softmax) | Multinomial, `lbfgs`, `max_iter=2000`, `class_weight='balanced'` |
| 3 | **CatBoost** | Boosting — ordered / symmetric trees | `RandomizedSearchCV(20, cv=3)`, `auto_class_weights='Balanced'` |
| 4 | **LightGBM** | Boosting — leaf-wise (GOSS) | `RandomizedSearchCV(20, cv=3)`, `class_weight='balanced'` |
| 5 | **XGBoost** | Boosting — level-wise, L1/L2 regularised | `RandomizedSearchCV(20, cv=3)`, balanced via `compute_sample_weight` |
| 6 | **MLP** | Deep feedforward — ReLU | 256→128→64→6, BatchNorm, Dropout(.4/.3/.2), Adam |
| 7 | **Swish-Net** | Deep feedforward — Swish/SiLU | 512→256→128→64→6, Dropout(.5/.4/.3/.2), RMSprop(1e-3) |
| 8 | **Soft-Vote Ensemble** | Meta-ensemble | Mean `predict_proba()` of top-3 sklearn models by validation F1 |

### Why three boosting libraries, not one

Each optimises a structurally different tree-growth strategy, giving the tournament three genuinely distinct perspectives:

- **CatBoost** — ordered boosting with symmetric (oblivious) trees, reducing the prediction shift inherent to classical gradient boosting
- **LightGBM** — leaf-wise (best-first) growth with Gradient-based One-Side Sampling: faster convergence, higher variance risk
- **XGBoost** — level-wise (depth-wise) growth with explicit L1/L2 penalties on leaf weights

<details>
<summary><b>Hyperparameter search grids</b></summary>

```python
# CatBoost
{'depth': [4, 6, 8], 'learning_rate': [0.03, 0.05, 0.1, 0.2],
 'iterations': [200, 400, 600], 'l2_leaf_reg': [1, 3, 5, 7]}

# LightGBM
{'num_leaves': [15, 31, 50], 'learning_rate': [0.03, 0.05, 0.1, 0.2],
 'n_estimators': [200, 400, 600], 'max_depth': [3, 5, 7, -1],
 'min_child_samples': [5, 10, 20]}

# XGBoost
{'max_depth': [3, 5, 7, 9], 'learning_rate': [0.03, 0.05, 0.1, 0.2],
 'n_estimators': [200, 400, 600], 'subsample': [0.7, 0.8, 0.9, 1.0],
 'colsample_bytree': [0.7, 0.8, 0.9, 1.0]}
```

All three use `n_iter=20`, `cv=3`, `scoring='f1_weighted'`.

</details>

### Deep network training protocol

Both networks share an identical regularisation and callback setup, isolating **activation function and optimiser** as the sole experimental variables:

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
]
model.fit(X_train_scaled, y_train,
          validation_data=(X_val_scaled, y_val),
          epochs=200, batch_size=32, callbacks=callbacks)
```

Loss is `sparse_categorical_crossentropy`, matching the integer-encoded `LabelEncoder` output.

### Why soft voting over hard voting

Probability averaging preserves each model's confidence calibration and stays well-defined when the three members split 2-1 — producing a smoother aggregate decision boundary than a discrete vote count.

---

## Evaluation Methodology

All 8 candidates are scored on the **held-out test partition** across seven metrics, then ranked:

```python
results_df = pd.DataFrame(results) \
                .sort_values('F1-Score (W)', ascending=False).reset_index(drop=True)
champion_name = results_df.iloc[0]['Model']
```

**Why weighted F1 is the primary criterion:** it rewards overall predictive accuracy while remaining sensitive to class imbalance — unlike raw accuracy, which a model can inflate by simply favouring the majority emotion.

**Why macro F1 is tracked alongside it:** macro F1 weights every class equally regardless of base rate. A model that buys majority-class accuracy at the cost of minority emotions (Boredom, Anger) will show a healthy weighted F1 and a poor macro F1. In a psychological-state application, every affective category matters equally.

### Metrics computed

| Metric | Purpose |
|---|---|
| Accuracy | Baseline sanity check |
| Precision / Recall / F1 (weighted) | Support-weighted headline performance |
| Precision / Recall / F1 (macro) | Minority-class fairness check |
| Per-class classification report | Isolates *which* emotions get confused |
| One-vs-Rest ROC-AUC | Threshold-independent discrimination per class |
| One-vs-Rest PR-AUC | Precision-recall trade-off per class |
| Normalised confusion matrix | Row-wise (recall-normalised) error structure |
| Learning curve (5-fold) | Bias vs. variance diagnostic |

### Results

Champion identity and exact metrics are produced by Block 19 at runtime and persisted to `pipeline_metadata.pkl`, then rendered live in the app's **Performance** tab.

| Rank | Model | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) |
|---|---|---|---|---|---|---|
| 1 | *(champion — see console output)* | | | | | |
| 2 | | | | | | |
| … | | | | | | |

> Transcribe from the Block 19 console table or `pipeline_metadata.pkl['model_performance']` after running the pipeline.

**A note on expected confusion:** Russell's circumplex model of affect organises emotions along valence and arousal axes. `Boredom` and `Neutral` occupy adjacent low-arousal territory and are the strongest a priori candidates for the hardest-to-separate pair from behavioural counters alone. Worth checking against your confusion matrix.

---

## Explainability

SHAP `TreeExplainer` attribution runs against the **best-performing tree model** — not necessarily the champion, since `TreeExplainer` requires a tree-structured estimator and the champion may be a Keras network or the ensemble.

```python
tree_model_names = [n for n in ['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest']
                     if n in trained_models]
best_tree_name = max(tree_model_names, key=lambda n: model_val_f1.get(n, 0))
explainer      = shap.TreeExplainer(trained_models[best_tree_name])
shap_values    = explainer.shap_values(X_test_df)
```

### Version-agnostic SHAP handling

SHAP's multi-class return type changed across library versions. Older releases return a **list of 2-D arrays** (one per class); newer releases return a single **3-D array** `(samples, features, classes)`. Wrapping either in a `pd.Series` without branching crashes with `Data must be 1-dimensional`.

```python
if isinstance(shap_values, list):
    # legacy: list of 2D arrays, one per class
    final_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
else:
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        # modern: (samples, features, classes)
        final_shap = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        final_shap = np.abs(shap_arr).mean(axis=0)
```

---

## MLOps: The ChampionModelWrapper Pattern

**The problem:** the champion could be any of three structurally incompatible object types.

- A **tree model** exposes `.predict_proba()` directly
- The **ensemble** is a bare list of three models with no unified interface
- A **Keras network** exposes `.predict()` but *not* `.predict_proba()`

Without a unifying abstraction, `app.py` would need runtime type-sniffing on every prediction — fragile, and a guaranteed source of production bugs.

**The solution:** a single wrapper class that normalises all three behind one contract.

```python
class ChampionModelWrapper:
    """Always exposes .predict_proba() and .predict(), whatever's inside."""

    def __init__(self, model_or_models, deploy_type, model_names=None):
        self.deploy_type = deploy_type        # 'Tree' | 'Ensemble' | 'Keras'
        self.model_names = model_names
        if deploy_type == 'Ensemble':
            self._models = model_or_models    # list of fitted sklearn estimators
        else:
            self._model = model_or_models

    def predict_proba(self, X):
        if self.deploy_type == 'Ensemble':
            return np.mean([m.predict_proba(X) for m in self._models], axis=0)
        elif self.deploy_type == 'Keras':
            return self._model.predict(X, verbose=0)
        else:
            return self._model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
```

The app calls `model.predict_proba(X)` — always. It never inspects what it loaded.

### Serialised artefacts

| File | Contents |
|---|---|
| `champion_model.pkl` / `.keras` | Winning model — pickled wrapper, or native Keras format if a network wins |
| `scaler.pkl` | Fitted `StandardScaler` (training-derived `mean_` / `scale_`) |
| `encoder.pkl` | Fitted `LabelEncoder` — integer predictions → emotion strings |
| `pipeline_metadata.pkl` | Feature list, class names, deploy type, champion name, dropped features, feature importance, confusion matrix, full 8-model results table, class distribution |
| `best_tree_model.pkl` | Best tree model, saved separately so the app can compute SHAP even when the champion isn't tree-structured |

---

## The Application: NeuroSense v3

A six-tab Streamlit dashboard with a custom dark glassmorphism design system — animated gradient background, neon-glow result cards, and Plotly-driven interactive charts throughout.

| Tab | Functionality |
|---|---|
| 🔮 **Predict** | Profile form → animated emotion result card → probability radar → class-probability bars → confidence gauge → engineered-feature values → **live SHAP waterfall** → probability sunburst |
| 📁 **Batch** | CSV upload → preview → progress-bar batch inference → results table → distribution pie → summary stats → annotated CSV download |
| 📊 **Performance** | Champion banner → 6 metric cards → full 8-model comparison table with max-highlighting → F1 ranking bar → feature importance → confusion-matrix heatmap → class distribution |
| 🔬 **Feature Lab** | Six engineered-feature explainer cards with formulas → ranked importance with inline progress bars → full retained-feature tag grid |
| 🎛️ **What-If** | Sweep one feature while holding others fixed → multi-line probability chart across all 6 emotions → emoji strip per sweep point → **sensitivity summary** ranking each emotion by probability range (Δ = max−min) |
| ℹ️ **About** | Methodology → model roster → tech stack → architecture flow → deployment precautions |

### Inference path

```python
def _build(age, gender, platform, usage, posts, likes, comments, messages):
    raw = { ... }   # raw + 6 engineered features
    # zero-init against the EXACT champion schema from pipeline_metadata.pkl
    row = pd.DataFrame(columns=FEATURES, data=[np.zeros(len(FEATURES))])
    for c, v in raw.items():
        if c in row.columns: row[c] = v
    row[f'Gender_{gender}']     = 1.0
    row[f'Platform_{platform}'] = 1.0
    return scaler.transform(row), raw
```

Zero-initialising against the persisted `FEATURES` list — rather than hand-reconstructing the schema — guarantees the inference vector matches training schema **even after the multicollinearity filter dropped an unpredictable subset of columns**.

---

## Repository Structure

```
.
├── datasets/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── plots/                        # generated EDA & evaluation figures
├── capstone3_pipeline.py         # 24-block training pipeline (1,233 lines)
├── app.py                        # NeuroSense v3 Streamlit dashboard (1,222 lines)
├── requirements.txt              # 13 pinned dependencies
├── .python-version               # pins 3.10 for Streamlit Cloud
│
├── champion_model.pkl/.keras     # ─┐
├── scaler.pkl                    #  │
├── encoder.pkl                   #  ├─ generated by Block 24
├── pipeline_metadata.pkl         #  │
├── best_tree_model.pkl           # ─┘
│
└── README.md
```

---

## Installation & Usage

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Train the pipeline

**Google Colab (recommended):**

```python
!pip install catboost lightgbm xgboost shap
```

Then run Blocks 1–24 sequentially. Block 2 auto-detects whether your CSVs are in `datasets/` or the working directory, so either layout works.

**Locally:**

```bash
python capstone3_pipeline.py
```

Blocks 13–14 (hyperparameter search) are the runtime bottleneck — three `RandomizedSearchCV` runs at 20 iterations × 3 folds each.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires all five artefacts from Block 24 in the working directory.

### Dependencies

```
streamlit>=1.30.0      numpy>=1.24.0        pandas>=2.0.0
scikit-learn>=1.3.0    tensorflow-cpu>=2.14.0
lightgbm>=4.0.0        catboost>=1.2.0      xgboost>=1.7.0
shap>=0.42.0           matplotlib>=3.7.0    seaborn>=0.12.0
plotly>=5.18.0         scipy>=1.10.0
```

---

## Deployment Guide

**Streamlit Community Cloud** — zero-infrastructure, auto-redeploys on push to `main`.

**Files to upload:**

```
app.py
requirements.txt
.python-version
champion_model.pkl (or .keras)
scaler.pkl
encoder.pkl
pipeline_metadata.pkl
best_tree_model.pkl
```

**Steps:** connect the repo at [share.streamlit.io](https://share.streamlit.io) → set main file to `app.py` → deploy. Python 3.10 is auto-detected from `.python-version`.

### Three non-negotiable deployment precautions

1. **`ChampionModelWrapper` must be defined at the top of `app.py`, before any `pickle.load()`.** Unpickling an object whose class is absent from the importing namespace raises `AttributeError` even though the pickle itself is valid.
2. **TensorFlow must be imported lazily** — inside `load_artifacts()`, and only when `model_deployment_type == 'Keras'`. Zero import cost when a tree model wins.
3. **Use `tensorflow-cpu`, never full `tensorflow`.** The full package bundles GPU kernels exceeding 500 MB and will overrun Streamlit Cloud's 1 GB RAM ceiling.

---

## Engineering Challenges Solved

| Bug | Root Cause | Fix |
|---|---|---|
| **SHAP 3-D array crash** | Newer SHAP returns `(samples, features, classes)`; `pd.Series()` on it throws `Data must be 1-dimensional` | Explicit `ndim == 3` branch → `mean(axis=(0,2))` |
| **Pickle `AttributeError`** | Wrapper class absent from `app.py` namespace at load time | Class defined at module top, before `pickle.load()` |
| **Streamlit Cloud OOM** | Full `tensorflow` ≈ 500 MB+ of GPU bloat vs. a 1 GB ceiling | `tensorflow-cpu` + lazy import |
| **Python 3.12 build failure** | `tensorflow` / `shap` / `catboost` lacked 3.12 wheels | `.python-version` pinning `3.10` |
| **`LabelEncoder` crash** | `"Agression"` typo is a class the encoder never fit on | `LABEL_TYPO_MAP` applied *before* encoding |
| **Inconsistent pickle contract** | Tree / ensemble / Keras expose three different interfaces | `ChampionModelWrapper` normalises all three |
| **Hardcoded Colab paths** | `datasets/train.csv` fails when CWD ≠ project root | Dynamic `DATA_DIR` resolution with descriptive fallback error |

---

## Limitations

Stated plainly, because a model that hides its limits is harder to trust than one that names them.

- **Label provenance is undocumented.** The source dataset doesn't specify whether `Dominant_Emotion` is self-reported, externally annotated, or synthetically generated. Every accuracy claim rests on this unverified assumption.
- **No temporal modelling.** Each row is treated as an i.i.d. user-day snapshot. A user's emotional *trajectory* across consecutive days — arguably the more meaningful signal — isn't captured.
- **Possible user-level leakage.** `User_ID` is dropped before this can be tested. If the same user appears across partitions, some leakage can't be ruled out from the schema alone.
- **Single-dataset validity.** Trained and evaluated exclusively on one Kaggle corpus. Generalisation to other platforms, cultures, age cohorts, or genuine production telemetry is unverified.
- **SHAP is a surrogate when the champion isn't a tree.** If a network or the ensemble wins, explanations come from `best_tree_model.pkl` — an approximation of, not identical to, the deployed model's logic.
- **No probability calibration.** `predict_proba()` outputs aren't passed through Platt scaling or conformal prediction. The displayed "confidence" is a relative model score, not a calibrated probability.
- **Correlation, not causation.** The cross-sectional design cannot establish whether usage drives emotion, emotion drives usage, or a shared confound drives both. **No causal claim is made or supported.**

---

## Ethical Considerations

Inferring emotional state from behavioural metadata is more sensitive than most applied-ML tasks and deserves more than a footnote.

- **Affective computing is a higher-risk category** under several emerging regulatory frameworks, especially when informing consequential decisions — employment screening, insurance, credit, or advertising targeted at detected emotional vulnerability. NeuroSense is an academic dashboard with no such integration, and extending it toward consequential use should require formal ethics review.
- **Manipulation risk.** A system that detects Anxiety, Sadness, or Boredom from engagement patterns could, in a different product context, be misused to target vulnerable users with attention-maximising content. Naming this risk is part of building it responsibly, even though this implementation doesn't act on it.
- **Demographics as direct inputs.** `Age` and `Gender` feed the model directly. A strong learned Gender→Emotion association risks *encoding* stereotypes about which emotions are "expected" from whom rather than detecting genuine behavioural signal. A fairness audit (equalised odds / demographic parity across Gender and Age bands) is recommended before any deployment beyond academic use.
- **Data provenance.** Publicly released, de-identified Kaggle corpus. `User_ID` is an anonymous anchor, dropped before modelling; no PII persists in any artefact.
- **Output is not a diagnosis.** The Predict tab shows an emotion label and confidence score. That's a probabilistic pattern-match against a training corpus — not a clinical or authoritative statement about anyone's mental state.

---

## Roadmap

- [ ] **Temporal sequence modelling** — reformulate as per-user time series (LSTM / GRU / Transformer encoder) to capture mood trajectories
- [ ] **Multimodal fusion** — add post text via transformer sentence embeddings instead of relying on engagement counts as an indirect proxy
- [ ] **Probability calibration** — Platt scaling, isotonic regression, or split conformal prediction for statistically valid confidence intervals
- [ ] **Fairness & bias audit** — equalised-odds testing across Gender and Age bands, plus a demographics-excluded ablation
- [ ] **Native explainability for non-tree champions** — DeepSHAP / Integrated Gradients for Keras, KernelExplainer for the ensemble
- [ ] **Automated retraining & drift monitoring** — scheduled rerun promoting a challenger only on statistically significant weighted-F1 improvement
- [ ] **REST microservice** — FastAPI `POST /predict` to decouple inference from the UI
- [ ] **Non-diagnostic disclaimer & consent layer** on the Predict and Batch tabs

---

## References

1. Bulut, E. (2024). *Social Media Usage and Emotional Well-Being* [Data set]. Kaggle.
2. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
3. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*.
4. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
6. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
7. Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML 32*.
8. Srivastava, N., et al. (2014). Dropout. *JMLR, 15*(1), 1929–1958.
9. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. *arXiv:1710.05941*.
10. Kingma, D. P., & Ba, J. (2014). Adam. *arXiv:1412.6980*.
11. Pedregosa, F., et al. (2011). Scikit-learn. *JMLR, 12*, 2825–2830.
12. Russell, J. A. (1980). A circumplex model of affect. *JPSP, 39*(6), 1161–1178.
13. Festinger, L. (1954). A theory of social comparison processes. *Human Relations, 7*(2), 117–140.

---

## Author

**Ashutosh Kaleron**
Applied Data Science, Machine Learning and AI — E&ICT Academy, IIT Guwahati

Capstone Project 3 · Classification using ML/DL

---

<div align="center">

**[▶ Live Demo](https://sentinet-social-11.streamlit.app/)** · **[📊 Dataset](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being)**

*Built with scikit-learn · CatBoost · LightGBM · XGBoost · TensorFlow · SHAP · Streamlit · Plotly*

</div>

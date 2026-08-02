# 🌲 Customer Behavior Pattern Analysis — Classification

> Supervised Machine Learning stage that trains a classifier to automatically recognize a customer's behavioral segment — segments that were originally discovered through unsupervised clustering.

![Dicoding Status](https://img.shields.io/badge/Submission-ACCEPTED-brightgreen?style=for-the-badge&logo=dicoding)
![Grade](https://img.shields.io/badge/Grade-ADVANCED-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)

*Final Submission for Dicoding Indonesia — "Belajar Machine Learning untuk Pemula" (Build Machine Learning Project)*

---

## ⚡ TL;DR — 30-Second Summary

- **Task:** Predict which behavioral segment a bank customer belongs to (Established/Passive vs. Young/Active), using the labels produced by an earlier clustering stage.
- **Approach:** One-Hot Encode the categorical features → baseline `DecisionTreeClassifier` → benchmarked against `RandomForestClassifier` → optimized with `GridSearchCV` (5-fold CV, exhaustive search over 27 combinations).
- **Result:** All three models hit a perfect 1.00 across accuracy, precision, recall, and F1 on a held-out 389-row test set.
- **The differentiator:** rather than stopping at "100% accuracy," this write-up explains *why* that number is mathematically expected here — and what it would mean (and not mean) in a real production setting. See **["Reading the Results Correctly"](#-reading-the-results-correctly)**.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Data Preparation & Feature Encoding** | One-Hot Encoding (`drop_first=True`) on 5 categorical columns, correctly reasoned to avoid the Dummy Variable Trap, before any model is touched |
| **Model Selection** | Comparing a simple baseline (Decision Tree) against an ensemble method (Random Forest) before committing to a final model |
| **Hyperparameter Tuning** | `GridSearchCV` with a deliberately justified search strategy (exhaustive vs. randomized) |
| **Reproducible ML Engineering** | Fixed `random_state=42` throughout, `stratify=y` on the split, and every trained model serialized with `joblib` for reuse without retraining |
| **Rigorous Evaluation** | Full `classification_report` (precision, recall, F1) rather than accuracy alone |
| **Statistical Reasoning** | Correctly diagnosing *why* a perfect score occurred — a Voronoi-partitioned target label from K-Means — instead of reporting it uncritically |
| **Production-Mindedness** | Flagging real deployment risks (feature-alignment on new data, concept drift) before they become bugs — see [Engineering Notes](#-6-engineering-notes-productionizing-this-pipeline) |

---

## 📖 1. Project Overview

This project is the **second stage** of a two-part Machine Learning pipeline. In the first stage (unsupervised clustering), bank customers were grouped into behavioral segments with no pre-existing labels. This second stage takes those segment labels and asks a different question:

> **"Can a model learn to predict which segment a customer belongs to, just from their transaction attributes — without ever running the clustering algorithm again?"**

This is a classic **cluster-then-classify** pattern used in production systems: clustering is run once (offline, on historical data) to *define* meaningful customer segments, and a lightweight classifier is then trained to *assign* new, incoming customers to those segments instantly — no need to re-cluster the entire dataset every time a new transaction comes in.

**Note:** this is a **customer behavior segmentation** task, not a fraud detection task. The goal is to understand and predict *spending/engagement style*, not to flag suspicious activity.

---

## 📂 2. Dataset Information & Target Classes

- **Source file:** `data_clustering_inverse.csv`
- **Size:** 1,945 rows (after outlier cleaning in the prior clustering stage)
- **Target column:** `Target` — the cluster label produced by the K-Means model from the clustering stage

### 🏷️ Target Classes

| Class | Persona | Key Characteristics |
|---|---|---|
| **0** | 🟦 Established / Passive Customer | Mature age profile, a stable resting account balance, and longer, more deliberate transaction durations |
| **1** | 🟨 Young / Active Customer | Younger age profile, higher transaction frequency, noticeably faster / more impulsive transaction durations, and a comparatively lower resting balance |

> 📎 These two classes are the same segments described in full statistical detail in the [Clustering stage README](../Clustering/README.md) — worth reading alongside this one, since the class definitions here originate entirely from that unsupervised analysis.

---

## 🧹 3. Preprocessing Steps

### Step 1 — Load the Clustering Output
The notebook loads `data_clustering_inverse.csv` directly — the **business-readable** version of the dataset (real ages, real currency values, real category names) rather than the scaled version, since this file already carries the `Target` labels needed for supervised learning.

### Step 2 — One-Hot Encoding
All categorical columns (`TransactionType`, `Location`, `Channel`, `CustomerOccupation`, `AgeGroup`) are converted into numerical form using **One-Hot Encoding** (`pd.get_dummies`) with **`drop_first=True`**.

- `drop_first=True` removes one redundant column per category to avoid the **Dummy Variable Trap** (multicollinearity caused by encoded columns being perfectly predictable from one another).
- Because `Location` alone has 43 distinct cities, this expansion is the main driver behind the final feature count: the dataset grows from **5 numeric + 5 categorical columns** into **55 total predictor features**.

### Step 3 — Data Splitting
The dataset is split using `train_test_split` with:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| Split | Rows | Share |
|---|---|---|
| Training set | 1,556 | 80% |
| Test set | 389 | 20% |

`stratify=y` guarantees the class ratio (Class 0 vs. Class 1) is preserved identically in both the training and test sets, which matters here since a naive random split could accidentally over- or under-represent one segment in the test set.

---

## 🤖 4. Modeling & Hyperparameter Tuning

Three models were built and saved in sequence, each one building on the last:

### 4.1 Baseline — Decision Tree
A `DecisionTreeClassifier(random_state=42)` was trained first as the baseline model — simple, fast, and easy to interpret, making it the natural reference point before trying anything more complex.
📦 Saved as: **`decision_tree_model.h5`**

### 4.2 Ensemble Model — Random Forest
A `RandomForestClassifier(random_state=42)` was trained next to see whether an ensemble of decision trees (which typically generalizes better and is less prone to overfitting than a single tree) could improve on the baseline.
📦 Saved as: **`explore_RandomForest_classification.h5`**

### 4.3 Hyperparameter Tuning — GridSearchCV
The Random Forest model was then optimized using **`GridSearchCV`** with 5-fold cross-validation:

```python
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

new_model_tuned = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring='accuracy'
)
```

**Best combination found:** `n_estimators=50`, `max_depth=None`, `min_samples_split=2`
📦 Saved as: **`tuning_classification.h5`**

#### 🤔 Why GridSearchCV instead of RandomizedSearchCV?

This project deliberately chose **`GridSearchCV`** (exhaustive search) over `RandomizedSearchCV` (random sampling) for a specific engineering reason: the parameter grid above only has **3 × 3 × 3 = 27 combinations**, and the dataset itself is small (under 2,000 rows). At this scale, an exhaustive search across every single combination — 135 total model fits across 5 folds — is computationally cheap and guarantees finding the *actual best* combination in the grid, with no need to trade accuracy for speed. `RandomizedSearchCV` earns its keep on much larger search spaces or bigger datasets, where checking every combination would be too expensive; that trade-off simply wasn't necessary here.

---

## 📊 5. Evaluation Results

All three models were evaluated on the held-out 389-row test set using `classification_report` (accuracy, precision, recall, F1-score):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (baseline) | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest (ensemble) | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest (tuned via GridSearchCV) | 1.00 | 1.00 | 1.00 | 1.00 |

*(Class-level breakdown: 196 test samples for Class 0, 193 for Class 1 — both classes scored a perfect 1.00 across every metric.)*

### 🔍 Reading the Results Correctly

A 100% score across every model and every metric should always raise an eyebrow — and it's worth explaining honestly rather than presenting it as an unqualified win.

The reason is structural, not a modeling trick: the `Target` label wasn't collected independently from the real world (like an actual fraud outcome or an actual churn event) — it was **generated directly by the K-Means clustering algorithm** in the prior stage. K-Means assigns every point to a cluster by finding its **nearest centroid**, which mathematically partitions the feature space into clean, non-overlapping regions (a Voronoi partition) with **zero ambiguity by construction**. When a classifier is then trained on that same (or closely related) feature space, it isn't predicting an uncertain real-world outcome — it's **reconstructing a deterministic boundary that was already perfectly defined** by the clustering step.

In short: this is an **expected characteristic of a cluster-then-classify pipeline on a clean, well-separated dataset**, not evidence of model leakage or a bug. It's a useful sanity check that the full pipeline (raw data → clustering → classification) is wired together correctly end-to-end. However, this 100% figure **should not be read the same way as 100% accuracy on an independently-labeled, real-world target** (e.g., actual fraud, actual churn) — those tasks involve genuine label noise and overlapping classes, and a perfect score there would be a red flag worth investigating rather than expecting.

---

## 🔧 6. Engineering Notes: Productionizing This Pipeline

A few things worth thinking through before this moves beyond a notebook into an actual serving system:

- **Feature alignment at inference time.** One-hot encoding is fit on whatever categories happen to appear in the training batch. A brand-new customer from a city not seen during training (or missing a category) would produce a different column set at inference than the model was trained on. In production, the encoder/column schema needs to be persisted (e.g., via the same `joblib` approach used for the models) and applied consistently to new data — not regenerated on the fly.
- **Model choice for deployment.** `GridSearchCV` selected Random Forest with `n_estimators=50` as the best performer, but Decision Tree remains the smaller, faster option if inference latency ever becomes a constraint and the accuracy gap is negligible.
- **Monitoring for drift.** Because the `Target` label is itself derived from clustering, the classifier is only as reliable as the clustering it was trained to imitate. If customer behavior shifts over time, the original K-Means segments may stop reflecting reality — meaning this classifier would need to be retrained against a refreshed clustering run periodically, not treated as a permanently fixed ground truth.

---

## 📁 7. Repository Structure

```text
Classification/
│
├── [Klasifikasi]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb   → Main classification notebook
├── decision_tree_model.h5                                      → Baseline Decision Tree model
├── explore_RandomForest_classification.h5                      → Random Forest ensemble model
├── tuning_classification.h5                                     → Hyperparameter-tuned Random Forest (GridSearchCV)
└── README.md                                                    → You are here
```

All models are serialized with **`joblib`** (`.h5` extension), allowing them to be reloaded instantly for inference or deployment without retraining from scratch.

---

## 🧰 8. Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | `Pandas` |
| Machine Learning | `Scikit-Learn` (`DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, `train_test_split`, `classification_report`) |
| Model Persistence | `Joblib` |
| Environment | Jupyter Notebook / Google Colab |

---

## 🚀 9. How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn joblib jupyter
```

### 2. Make sure the input file is present
Place `data_clustering_inverse.csv` (exported from the Clustering stage) in the same directory as the notebook.

### 3. Launch and run
```bash
jupyter notebook "[Klasifikasi]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb"
```
Run all cells from top to bottom — the notebook will re-generate all three `.h5` model files.

### 4. Reuse a trained model
```python
import joblib

model = joblib.load("tuning_classification.h5")
predicted_segment = model.predict(new_customer_data)
```

---

## 👤 Author

**Roihan Saputra**
*Machine Learning Engineer Enthusiast*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.

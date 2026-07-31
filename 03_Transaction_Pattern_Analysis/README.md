# 🏦 03 — Transaction Pattern Analysis

> A two-stage Machine Learning project that first **discovers** hidden customer segments in bank transaction data (unsupervised learning), then **teaches a model to recognize** those segments automatically (supervised learning).

![Dicoding Status](https://img.shields.io/badge/Submission-ACCEPTED-brightgreen?style=for-the-badge&logo=dicoding)
![Grade](https://img.shields.io/badge/Grade-ADVANCED-blue?style=for-the-badge)

*Final Submission for Dicoding Indonesia — "Belajar Machine Learning untuk Pemula" (Build Machine Learning Project)*

---

## 📖 What's in This Folder?

This directory contains **one connected project told in two parts**. Both parts work on the same underlying bank transaction dataset, but each answers a different business question:

| Stage | Question it answers | Learning type |
|---|---|---|
| 🧩 **[`Clustering/`](./Clustering)** | *"What natural customer groups already exist in our transaction data — with no labels given?"* | Unsupervised Learning |
| 🌲 **[`Classification/`](./Classification)** | *"Given the segments we just discovered, can a model instantly predict which segment a brand-new customer belongs to?"* | Supervised Learning |

In short: **Clustering finds the pattern. Classification learns to repeat it automatically.** This mirrors a very common real-world ML workflow — segment your customers once with unsupervised learning, then deploy a fast classifier so every new customer gets sorted in real time without re-running the clustering algorithm.

---

## 🏆 Official Assessment

This submission was reviewed by the **Dicoding Indonesia** review team and rated **Advanced** — the highest tier — across all 5 assessment criteria:

| # | Criteria | Result |
|---|---|---|
| 1 | Data Loading & Exploratory Data Analysis (EDA) | ✅ Advanced (4/4) |
| 2 | Data Cleaning & Preprocessing | ✅ Advanced (4/4) |
| 3 | Clustering Model Development | ✅ Advanced (4/4) |
| 4 | Cluster Interpretation | ✅ Advanced (4/4) |
| 5 | Classification Model Development | ✅ Advanced (4/4) |

---

## 📁 Repository Structure

```text
03_Transaction_Pattern_Analysis/
│
├── Clustering/
│   ├── [Clustering]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb   → main clustering notebook
│   ├── data_clustering.csv                                        → scaled, ML-ready dataset + cluster labels
│   ├── data_clustering_inverse.csv                                → business-ready dataset (real-world values) + cluster labels
│   ├── model_clustering.h5                                        → trained K-Means model
│   ├── PCA_model_clustering.h5                                    → K-Means model trained on PCA-reduced features
│   └── README.md                                                  → full clustering write-up
│
├── Classification/
│   ├── [Klasifikasi]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb   → main classification notebook
│   ├── decision_tree_model.h5                                     → baseline Decision Tree model
│   ├── explore_RandomForest_classification.h5                     → alternative Random Forest model
│   ├── tuning_classification.h5                                   → hyperparameter-tuned final model
│   └── README.md                                                  → full classification write-up
│
└── README.md   ← you are here
```

👉 This file is just the **map**. For the full technical breakdown of each stage — preprocessing decisions, model choices, evaluation metrics, and business interpretation — open the `README.md` inside each subfolder.

---

## 🧩 Stage 1: Clustering (Unsupervised)

Segments ~1,945 cleaned bank transaction records into behavioral customer groups using **K-Means**, with the optimal number of clusters chosen via the **Elbow Method** and validated with a **Silhouette Score**. Also applies **PCA** to visualize the clusters in 2D and inverse-transforms the results back into real-world values (age, currency, category labels) so the findings are usable by non-technical stakeholders.

**📂 [→ See the full Clustering README](./Clustering/README.md)**

---

## 🌲 Stage 2: Classification (Supervised)

Takes the cluster labels produced in Stage 1 as the prediction target, then trains and compares multiple classification algorithms:

- **Baseline:** `DecisionTreeClassifier`
- **Alternative model explored:** `RandomForestClassifier`
- **Hyperparameter tuning:** `GridSearchCV` (5-fold cross-validation) tuning `n_estimators`, `max_depth`, and `min_samples_split`

Each model is evaluated with **accuracy, precision, recall, and F1-score** via `classification_report`, and the best-performing tuned model is saved for reuse.

**📂 [→ See the full Classification README](./Classification/README.md)**

---

## 🔗 How the Two Stages Connect

```text
┌─────────────────────┐        ┌──────────────────────────┐        ┌────────────────────────┐
│   Raw Transaction    │  --->  │   Clustering (K-Means)    │  --->  │   Classification Model   │
│   Data (unlabeled)   │        │   assigns a "Target"      │        │   learns to predict the  │
│                      │        │   cluster label            │        │   "Target" for new data  │
└─────────────────────┘        └──────────────────────────┘        └────────────────────────┘
                                data_clustering_inverse.csv  --------------->  used as training input
```

The clustering notebook exports `data_clustering_inverse.csv` with a `Target` column attached. The classification notebook reads that exact file, encodes it, and trains on it — making the two notebooks a single, reproducible end-to-end pipeline rather than two disconnected exercises.

---

## 🧰 Shared Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Environment | Jupyter Notebook / Google Colab |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (`StandardScaler`, `LabelEncoder`, `KMeans`, `PCA`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`) |
| Cluster Selection | `yellowbrick` (`KElbowVisualizer`) |
| Model Persistence | `joblib` |

---

## 🚀 Where to Go Next

- New here? Start with **[`Clustering/README.md`](./Clustering/README.md)** to see how the customer segments were discovered.
- Want the modeling results? Head to **[`Classification/README.md`](./Classification/README.md)** for accuracy, precision/recall breakdowns, and the tuning process.

---

## 👤 Author

**Roihan Saputra**
*Machine Learning Engineer Enthusiast*
GitHub: [https://github.com/Roihan21](https://github.com/Roihan21)

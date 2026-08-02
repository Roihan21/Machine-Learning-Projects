# 🏦 Transaction Pattern Analysis — Customer Segmentation & Predictive Classification

> A two-stage Machine Learning case study: first **discovering** hidden customer behavior segments in raw bank transaction data with no labels given, then **training a model to instantly recognize** those segments for every new customer.

![Dicoding Status](https://img.shields.io/badge/Submission-ACCEPTED-brightgreen?style=for-the-badge&logo=dicoding)
![Grade](https://img.shields.io/badge/Grade-ADVANCED-blue?style=for-the-badge)

*Final Submission for Dicoding Indonesia — "Belajar Machine Learning untuk Pemula" (Build Machine Learning Project)*

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** A bank has thousands of transactions but no existing way to group customers by behavior.
- **Approach:** Unsupervised clustering (K-Means) to *discover* natural segments → Supervised classification (Decision Tree / Random Forest, tuned via GridSearchCV) to *predict* segment membership for new customers.
- **Result:** Rated **Advanced** (highest tier) across all 5 official assessment criteria; 1,945 records processed end-to-end, from raw data to a deployable, saved model.
- **What makes it stand out:** the write-up doesn't just report scores — it explains *why* they came out the way they did, including calling out where results should be read with caution. See **["What Sets This Apart"](#-what-sets-this-apart)** below.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Data Cleaning & EDA** | Handling missing values, duplicates, and outliers (IQR method) on a 2,500+ row raw dataset |
| **Feature Engineering** | Label Encoding, One-Hot Encoding, binning, and scaling — with a clear rationale for each choice |
| **Unsupervised Learning** | K-Means clustering, optimal-K selection via Elbow Method, validated with Silhouette Score |
| **Dimensionality Reduction** | PCA, used both for visualization and as a comparison model |
| **Supervised Learning** | Decision Tree & Random Forest classification, hyperparameter tuning with GridSearchCV |
| **Model Evaluation** | Precision, Recall, F1-score, and honest interpretation of what those numbers actually mean |
| **Business Translation** | Converting scaled/statistical output back into real-world terms (age, currency, category) that a non-technical stakeholder can act on |
| **Technical Communication** | This documentation itself — structured, audience-aware writing across three linked README files |

---

## 🎯 Why This Project Matters

Segmenting customers is a real, recurring business need — it's the foundation for targeted marketing, personalized product offers, and smarter resource allocation. This project doesn't stop at "build a model that works." It follows the full lifecycle a data team actually goes through: clean the data, justify every preprocessing decision, validate the clustering statistically, translate results into business language, then operationalize those results with a classifier that can run on new customers in real time — no manual re-analysis required.

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

👉 This file is the **map**. For the full technical breakdown of each stage — preprocessing decisions, model choices, evaluation metrics, and business interpretation — open the `README.md` inside each subfolder.

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

## 🧠 What Sets This Apart

Both sub-project READMEs include something a lot of portfolio projects skip: **an honest account of where the results are strong and where they need context.** For example, the classification models score a perfect 1.00 on every metric — and instead of presenting that at face value, the write-up explains the mathematical reason why (the target label was generated by the clustering step itself, creating a perfectly separable boundary by construction) and is explicit about why that shouldn't be read the same way as 100% accuracy on an independently-collected, real-world label.

This kind of self-checking is a deliberate part of the process here — treating a good-looking number as a starting question rather than a finish line.

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
GitHub: [https://github.com/Roihan21](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.

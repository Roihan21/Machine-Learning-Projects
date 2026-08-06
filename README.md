<div align="center">

# 📊 Classic Machine Learning & Data Science Vault
### From Raw Tabular Data to Validated, Production-Ready Models

**Regression · Classification · Clustering · Feature Engineering · Model Evaluation**
*No neural nets, no hype — just solid statistical learning, done properly.*

![Status](https://img.shields.io/badge/status-actively--maintained-brightgreen?style=for-the-badge)
![Focus](https://img.shields.io/badge/focus-Classic%20ML%20%26%20Data%20Science-blue?style=for-the-badge)
![Made with](https://img.shields.io/badge/made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

---

### 📖 Table of Contents
- [About This Repository](#-about-this-repository)
- [About Me](#-about-me)
- [Data Science Workflow](#️-data-science-workflow-covered-here)
- [Tech Stack & Tools](#️-tech-stack--tools)
- [Projects](#-projects)
- [Repository Structure](#-repository-structure)
- [Roadmap](#-roadmap)
- [GitHub Stats](#-github-stats)
- [Connect With Me](#-connect-with-me)

---

## 📌 About This Repository

This is a focused collection of my **classic Machine Learning and Data Science** projects — deliberately scoped to statistical learning on tabular data rather than trying to cover every corner of AI at once. Every project here follows the same discipline: understand the data first, justify every preprocessing decision, validate results properly, and be explicit about what a model can and can't actually claim.

> No deep learning, no LLMs here — just regression, classification, and clustering, built and explained the way a data team would actually do it.

---

## 👋 About Me

Hi, I'm **Roihan** — an Informatics/IT student at **Politeknik Negeri Semarang (Polines)** and an aspiring **Data Scientist / Machine Learning Engineer**.

I care less about collecting algorithm names and more about the fundamentals underneath them: is the data actually clean, is the split actually fair, does the metric actually mean what I think it means. This repository is where I practice that discipline in public, project by project.

---

## ⚙️ Data Science Workflow Covered Here

| Stage | Process | Key Techniques & Methods |
|---|---|---|
| 🧹 **Data Preprocessing** | Cleaning & preparation | Missing values, outlier detection (IQR), encoding, scaling |
| 🔍 **Exploratory Data Analysis** | Insights & patterns | Feature distributions, correlation analysis |
| 🛠️ **Feature Engineering** | Value addition | Feature selection, dimensionality reduction (PCA), binning |
| 🤖 **Predictive Modeling** | Algorithm selection | Regression, Classification, Clustering (supervised & unsupervised) |
| 📐 **Model Evaluation** | Validation | Train/test splits, cross-validation, Silhouette Score, Precision/Recall/F1 |

---

## 🛠️ Tech Stack & Tools

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

| Category | Tools |
|---|---|
| **Core Language** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn (Linear Regression, Decision Trees, Random Forest, K-Means, PCA) |
| **Model Persistence** | Joblib |
| **Environment** | Jupyter Notebook, Google Colab |

---

## 📂 Projects

| # | Project | Category | Status |
|---|---|---|---|
| 01 | [**Molecular Solubility Prediction**](./01_Molecular_Solubility_Prediction) | Regression | ![Done](https://img.shields.io/badge/-Completed-2ea44f?style=flat-square) |
| 02 | [**Data Preprocessing**](./02_Data_Preprocessing) | Data Cleaning & Preprocessing | ![Done](https://img.shields.io/badge/-Completed-2ea44f?style=flat-square) |
| 03 | [**Transaction Pattern Analysis**](./03_Transaction_Pattern_Analysis) | Clustering & Classification | ![Done](https://img.shields.io/badge/-Completed-2ea44f?style=flat-square) |
| 04 | [**Building Classification Model**](./04_Building_Classification_Model) | Classification | ![Done](https://img.shields.io/badge/-Completed-2ea44f?style=flat-square) |
| 05 | [**Comparing Classifiers for Building Classification Model**](./05_Comparing_Classifiers_for_Building_Classification_Model) | Classification | ![Done](https://img.shields.io/badge/-Completed-2ea44f?style=flat-square) |

**A closer look at each:**

- **01 — Molecular Solubility Prediction:** Predicts aqueous solubility (logS) of molecules from their chemical descriptors using the Delaney (ESOL) dataset, comparing a Linear Regression baseline against a tuned Random Forest model.
- **02 — Data Preprocessing:** A dedicated preprocessing exercise on the UCI Wine dataset — cleaning, scaling, and encoding tabular data properly before it ever reaches a model.
- **03 — Transaction Pattern Analysis:** A two-stage pipeline on bank transaction data — K-Means clustering to discover customer segments, then a tuned classifier to predict segment membership. Rated **Advanced** by Dicoding Indonesia's review team.
- **04 — Building Classification Model:** A foundational Random Forest classifier on the Iris dataset, built specifically to demonstrate and then fix a data leakage problem — comparing an overfit model against a properly validated one.
- **05 — Comparing Classifiers for Building Classification Model:** Benchmarks multiple classification algorithms against each other on a shared classification task — a direct follow-up to the "try other classifiers" idea flagged as future work in Project 04.

> 📎 Every project folder has its own README with the full technical write-up. Project 05's description above is based on its file naming; happy to sharpen it further once its notebook is shared for a detailed review.

---

## 📁 Repository Structure

```text
Machine-Learning-Projects/
│
├── 01_Molecular_Solubility_Prediction/
│   ├── images/
│   │   ├── linear_regression.png
│   │   ├── model_comparison.png
│   │   ├── random_forest.png
│   │   └── tuned_random_forest.png
│   ├── Molecular_Solubility.ipynb
│   ├── delaney_solubility_with_descriptors.csv
│   ├── model_rf_logS_best_estimator.pkl
│   └── README.md
│
├── 02_Data_Preprocessing/
│   ├── Data_Preprocessing.ipynb
│   ├── wine.data
│   └── README.md
│
├── 03_Transaction_Pattern_Analysis/
│   ├── Classification/
│   ├── Clustering/
│   └── README.md
│
├── 04_Building_Classification_Model/
│   ├── iris_classification_random_forest.ipynb
│   └── README.md
│
├── 05_Comparing_Classifiers_for_Building_Classification_Model/
│   ├── Comparing_Classifiers_for_Building_Classification_Model.ipynb
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## 🚀 Roadmap

**Phase 1 — Data & ML Fundamentals**
![Progress](https://img.shields.io/badge/progress-100%25-2ea44f?style=flat-square)
- [x] Exploratory Data Analysis (EDA) & data preprocessing
- [x] Supervised learning (Linear Regression, Decision Trees, Random Forest)
- [x] Unsupervised learning (K-Means clustering, PCA)
- [x] Comparing multiple classification algorithms head-to-head

**Phase 2 — Model Optimization** 🔥 *current focus*
- [ ] Systematic hyperparameter tuning (GridSearchCV / RandomizedSearchCV) across more projects
- [ ] Ensemble methods beyond Random Forest (Gradient Boosting)
- [ ] Deeper statistical validation (confidence intervals, significance testing on model comparisons)

**Phase 3 — Applied Extensions** ⏳ *up next*
- [ ] Classical time series forecasting (e.g., ARIMA, engineered lag features) on sequential/sensor data
- [ ] Packaging a model behind a simple API for real inference, not just notebook predictions
- [ ] Light MLOps: versioning models and tracking experiments properly instead of ad hoc `.pkl` files

---

## 📊 GitHub Stats

<div align="center">

![Repo Stars](https://img.shields.io/github/stars/RoihansLab/Machine-Learning-Projects?label=Stars&style=for-the-badge&color=2c5364)
![Last Commit](https://img.shields.io/github/last-commit/RoihansLab/Machine-Learning-Projects?label=Last%20Commit&style=for-the-badge&color=2ea44f)
![Top Language](https://img.shields.io/github/languages/top/RoihansLab/Machine-Learning-Projects?label=Top%20Language&style=for-the-badge&color=F7931E)
![Repo Size](https://img.shields.io/github/repo-size/RoihansLab/Machine-Learning-Projects?label=Repo%20Size&style=for-the-badge&color=8E75B2)

</div>

---

## 🤝 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RoihansLab)

</div>

<div align="center">

### 💡 *"A model is only as trustworthy as the validation behind it — the metric on the last cell means nothing without the process that got there."*

⭐️ **If this repository helps you, consider giving it a star.**

</div>

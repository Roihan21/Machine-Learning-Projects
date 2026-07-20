# 🧪 Predicting Molecular Solubility using Machine Learning

*My first end-to-end Machine Learning project — learning how to build, evaluate, and improve a regression model beyond just following a tutorial.*

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

## 📑 Table of Contents

- [Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Results](#-results)
- [Current Limitations](#-current-limitations)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Beyond the Original Tutorial](#-beyond-the-original-tutorial)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [Learning Resources](#-learning-resources)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 📖 Project Overview

This project predicts the aqueous solubility (`logS`) of molecules using Machine Learning regression models.

Molecular solubility is an important physicochemical property in chemistry and pharmaceutical research — it helps determine whether a molecule has potential as a drug candidate, since poorly soluble compounds are harder to formulate and test.

This project marks an important milestone in my Machine Learning journey. Rather than just training a model and stopping, my goal was to understand the full workflow of a regression problem — from loading the dataset to evaluating and improving model performance.

## 🎯 Objectives

- Understand the end-to-end Machine Learning workflow for regression problems.
- Compare how different regression models perform on the same dataset.
- Evaluate model performance using standard regression metrics.
- Improve model performance through Hyperparameter Tuning.
- Practice documenting a Machine Learning project as part of my portfolio.

## 📂 Dataset

**Dataset:** [ESOL / Delaney Solubility Dataset](https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv) — 1,144 molecules with precomputed descriptors.

**Target variable**
- `logS` — aqueous solubility

**Input features**

| Feature | Meaning |
|---|---|
| `MolLogP` | Octanol-water partition coefficient (lipophilicity) |
| `MolWt` | Molecular weight |
| `NumRotatableBonds` | Molecular flexibility |
| `AromaticProportion` | Proportion of aromatic atoms in the molecule |

## 🔄 Project Workflow

```text
Load Dataset
      │
      ▼
Data Separation (X, y)
      │
      ▼
Train-Test Split
      │
      ▼
Train Linear Regression ──► Evaluate
      │
      ▼
Train Random Forest (baseline) ──► Evaluate
      │
      ▼
Hyperparameter Tuning
(RandomizedSearchCV)
      │
      ▼
Train Tuned Random Forest ──► Evaluate
      │
      ▼
Model Comparison
      │
      ▼
Prediction Visualization
```

*Note: this version does not yet include a dedicated EDA/data-cleaning step — see [Current Limitations](#-current-limitations).*

## 🤖 Models Used

| Model | Description |
|---|---|
| Linear Regression | Simple baseline regression model |
| Random Forest (baseline) | Ensemble model, shallow depth (`max_depth=2`) |
| Random Forest (Tuned) | Same model, optimized via `RandomizedSearchCV` |

## 📊 Results

| Model | Train MSE | Train R² | Test MSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.008 | 0.765 | 1.021 | 0.789 |
| Random Forest (baseline) | 1.028 | 0.760 | 1.408 | 0.709 |
| **Random Forest (Tuned)** | **0.303** | **0.929** | **0.676** | **0.860** |

### Key Findings

- Linear Regression provided a solid, simple baseline.
- The shallow-depth baseline Random Forest actually **underperformed** Linear Regression on the test set — a reminder that Random Forest isn't automatically better without proper tuning.
- After applying `RandomizedSearchCV` (5-fold CV, 50 iterations), the tuned Random Forest achieved the best performance by a clear margin (Test R² 0.860 vs. 0.789).
- **Takeaway:** hyperparameter tuning mattered more here than the choice of algorithm itself.

## ⚠️ Current Limitations

Being upfront about where this project stands right now:

- No EDA yet — missing values, feature distributions, and correlations between features (e.g. `MolLogP` vs `MolWt`) haven't been checked.
- No data cleaning/preprocessing step; the dataset is used as-is.
- Model comparison relies on a single 80/20 train-test split rather than cross-validated scores across all models.
- No baseline (dummy regressor) to show how much better these models are than simply guessing the average.
- The Random Forest "baseline" uses a shallow depth rather than scikit-learn's true default, so it isn't a fully fair pre-tuning comparison.

I'm treating these as the next things to fix — see [Future Improvements](#-future-improvements).

## 📉 Visualizations

Prediction plots (predicted vs. experimental logS) for each model are available in the `images/` folder:

- Linear Regression prediction plot
- Random Forest prediction plot
- Tuned Random Forest prediction plot
- Model performance comparison

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib
- Jupyter Notebook

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## 🗂️ Project Structure

```text
├── notebook.ipynb          # Main analysis and modeling notebook
├── images/                 # Saved prediction plots
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Beyond the Original Tutorial

This project was initially developed by following the tutorial below, up to the **Prediction Visualization** stage.

To deepen my understanding, I continued experimenting independently by adding:

- `RandomizedSearchCV` for hyperparameter tuning
- Tuned model evaluation and comparison
- Updated results table and documentation

These additions helped me understand *why* model optimization improves performance, not just *that* it does.

## 💡 What I Learned

- How to build regression models with scikit-learn.
- How to compare multiple regression algorithms fairly.
- How to evaluate models using MSE and R².
- How to apply hyperparameter tuning with `RandomizedSearchCV`.
- How to document a Machine Learning project for a GitHub portfolio.

More importantly, this project taught me that building a model isn't just about chasing better metrics — it's about understanding *why* a model performs the way it does, and being honest about what's still missing.

## 🔮 Future Improvements

- [ ] Exploratory Data Analysis (distributions, correlations, outliers)
- [ ] Baseline dummy regressor for context
- [ ] Fair default-parameter Random Forest baseline
- [ ] Feature importance analysis
- [ ] Residual analysis
- [ ] Additional models: XGBoost, CatBoost
- [ ] Explainable AI (SHAP)
- [ ] Model persistence + simple inference example

## 📚 Learning Resources

This project was inspired by educational content from **Data Professor**:

- 📺 [Build your first Machine Learning model in Python (YouTube)](https://youtu.be/29ZQ3TDGgRQ?si=bWJpY_14MABtvXVg)
- 📺 [Data Professor YouTube channel](https://www.youtube.com/@DataProfessor)
- 💻 [Data Professor GitHub](https://github.com/dataprofessor)
- 📂 [Dataset source](https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv)

## 🙏 Acknowledgements

Sincere thanks to **Data Professor** for creating high-quality educational content and making Machine Learning more accessible to beginners. This project follows that tutorial as a starting point, then extends it independently with hyperparameter tuning, additional evaluation, and improved documentation as part of my own learning journey.

## 👨‍💻 Author

**Roihan**
Aspiring Machine Learning Engineer

*"Learning by building, improving through experimentation."* 🚀

# 🧪 01_Predicting Molecular Solubility using Machine Learning

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
- [Model Persistence & Inference](#-model-persistence--inference)
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
Data Preparation
      ├── Feature Selection (X)
      ├── Target Selection (y)
      └── Train-Test Split
      │
      ▼
Linear Regression
      ├── Train Model
      ├── Make Predictions
      └── Evaluate Performance (MSE & R²)
      │
      ▼
Random Forest (Baseline)
      ├── Train Model
      ├── Make Predictions
      └── Evaluate Performance (MSE & R²)
      │
      ▼
Baseline Model Comparison
      └── Prediction Visualization
      │
      ▼
Hyperparameter Tuning
      └── RandomizedSearchCV (5-Fold Cross Validation)
      │
      ▼
Train Tuned Random Forest
      ├── Best Estimator
      ├── Make Predictions
      └── Evaluate Performance (MSE & R²)
      │
      ▼
Updated Model Comparison
      ├── Compare All Models
      └── Visualize Prediction Results
      │
      ▼
Save Best Model
      └── model_rf_logS_best_estimator.pkl
      │
      ▼
Load Saved Model
      └── pickle.load()
      │
      ▼
Model Inference
      └── Predict on unseen test samples

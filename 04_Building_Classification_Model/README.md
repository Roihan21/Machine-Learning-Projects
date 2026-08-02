# 🌸 04 — Iris Flower Classification using Machine Learning

> My foundational Machine Learning classification project — exploring decision boundaries, preventing data leakage, and analyzing model probability.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** Classify Iris flowers into one of 3 species from 4 physical measurements — the classic "Hello World" of Machine Learning.
- **Approach:** Random Forest Classifier, deliberately demonstrating what goes wrong with a naive (leaked) train/test setup before fixing it with a proper stratified split.
- **Result:** ~96.67% accuracy on a held-out test set, with `predict_proba` used to show the model reasons in probabilities rather than blind guesses.
- **What makes it more than a copy-pasted tutorial:** an explicit, hands-on demonstration of **data leakage** — showing exactly what changes (and why it matters) between an overfit model and a properly validated one. See [Beyond the Original Tutorial](#-beyond-the-original-tutorial).

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Core ML Workflow** | End-to-end pipeline: load → explore → split → train → evaluate → persist |
| **Data Leakage Awareness** | Deliberately trained a model on 100% of the data first to *show* the failure mode, before fixing it |
| **Stratified Sampling** | `train_test_split(..., stratify=y, random_state=42)` to keep class balance fair and results reproducible |
| **Model Interpretability** | Feature importance analysis (petal measurements dominate; sepal measurements barely matter) |
| **Probabilistic Reasoning** | Used `predict_proba` to inspect model confidence, not just final class labels |
| **Model Persistence** | Serialized the trained model with `joblib` for reuse without retraining |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Results & Key Insights](#-results--key-insights)
- [Model Persistence & Inference](#-model-persistence--inference)
- [Current Limitations](#-current-limitations)
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

This project focuses on building a Machine Learning classification model to predict the species of Iris flowers based on their physical measurements.

Often considered the "Hello World" of Machine Learning, the Iris dataset provides a perfect playground to understand the core mechanics of classification algorithms. However, rather than just training a model to get a high accuracy score, this project dives deeper into understanding **how** the model learns, how to prevent **data leakage**, and how to analyze the model's prediction confidence using probability metrics.

---

## 🎯 Objectives

- Understand the end-to-end Machine Learning workflow for classification problems.
- Implement a Random Forest Classifier using scikit-learn.
- Identify and prevent **data leakage** (overfitting) by properly splitting data.
- Ensure balanced class distributions in training/testing data using **stratified sampling**.
- Analyze how the model makes decisions using feature importance and probability predictions (`predict_proba`).
- Serialize the trained model for future deployment.

---

## 📂 Dataset

**Dataset:** The classic Iris dataset, loaded directly via `sklearn.datasets.load_iris()`. It contains 150 instances of Iris flowers evenly distributed across 3 species.

**Target variable (classes):**

| Label | Species |
|---|---|
| `0` | Setosa |
| `1` | Versicolor |
| `2` | Virginica |

**Input features:**

| Feature | Meaning | Importance (RF Model) |
|---|---|---:|
| `sepal length (cm)` | Length of the sepal | ~10.9% |
| `sepal width (cm)` | Width of the sepal | ~3.3% |
| `petal length (cm)` | Length of the petal | **~42.4%** |
| `petal width (cm)` | Width of the petal | **~43.2%** |

---

## 🔄 Project Workflow

```text
Load Iris Dataset
      │
      ▼
Data Exploration
      ├── Feature Names (X)
      ├── Target Names (y)
      └── Data Dimensions (150, 4)
      │
      ▼
Concept Demonstration: Data Leakage (Overfitting)
      ├── Train Model on 100% of Data (model.fit(X, y))
      ├── Evaluate Feature Importance (on this leaky model)
      ├── Predict on known data (Index 0, 50, 100)
      └── Observe 100% false confidence [1. 0. 0.]
      │
      ▼
Proper Data Splitting
      └── train_test_split (80/20, stratify=y, random_state=42)
      │
      ▼
Random Forest Classifier (Rebuilt)
      ├── Train Model on X_train, y_train
      └── Make Predictions on X_test
      │
      ▼
Model Evaluation & Inference Analysis
      ├── Compare Predictions vs Actual (y_test)
      ├── Analyze model confidence (predict_proba)
      └── Calculate Accuracy Score (~96.67%)
      │
      ▼
Save Best Model
      └── iris_classification_RF (via joblib)
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| **Random Forest Classifier** | An ensemble learning method that operates by constructing a multitude of decision trees at training time. Used with default hyperparameters for this foundational project. |

---

## 📊 Results & Key Insights

- **Accuracy Score:** The properly trained model achieved an accuracy of **~96.67%** on the unseen test set — a highly realistic and near-optimal result for this dataset.
- **Feature importance is the key signal:** The Random Forest model revealed it heavily relies on **petal width (43.2%)** and **petal length (42.4%)** to classify the flowers, while largely ignoring sepal dimensions. *(Note: this importance was read from the initial full-dataset model, computed before the leakage fix — worth re-checking on the properly split/rebuilt model as a future refinement, though Random Forest importance rankings are typically stable across refits on a dataset this clean.)*
- **The danger of data leakage:** When the model was trained on 100% of the dataset, it merely memorized the answers. Testing it on already-seen data produced absolute confidence (`[1. 0. 0.]`). Properly evaluating it on a held-out test set instead revealed the model actually "thinks" in probabilities (e.g., `[0. 0.14 0.86]`) — a healthy sign of genuine uncertainty when facing new data, rather than memorization.

---

## 💾 Model Persistence & Inference

After validating the Random Forest model on the test set, the model was serialized using Python's `joblib` module.

The saved model (`iris_classification_RF`) can be loaded back into memory to perform predictions on new, unseen flower measurements in a production environment — without needing to retrain the algorithm.

```python
import joblib

model = joblib.load("iris_classification_RF")
predicted_species = model.predict(new_flower_measurements)
```

---

## ⚠️ Current Limitations

- This is a foundational project meant for learning core mechanics; hyperparameter tuning (e.g., `GridSearchCV` or `RandomizedSearchCV`) was not performed here.
- The dataset is extremely clean and small (150 rows). It does not reflect the messy nature of real-world data, which typically requires extensive handling of missing values or outliers.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | `scikit-learn` |
| Data Handling | `numpy` |
| Environment | Jupyter Notebook / Google Colab |
| Model Persistence | `joblib` |

---

## ▶️ How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Open `iris_classification_random_forest.ipynb` in Google Colab.
2. Run all notebook cells from top to bottom.
3. The notebook automatically imports the dataset from scikit-learn and executes the entire workflow — no manual data download needed.

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
```

**2. Navigate to the project folder**
```bash
cd Machine-Learning-Projects/04_Building_Classification_Model
```

**3. Install the required Python libraries**
```bash
pip install numpy scikit-learn joblib jupyter
```

**4. Launch Jupyter Notebook**
```bash
jupyter notebook iris_classification_random_forest.ipynb
```

Then open the notebook and run all cells sequentially.

---

## 🗂️ Project Structure

```text
04_Building_Classification_Model/
│
├── README.md                                  → Project documentation
├── iris_classification_random_forest.ipynb     → End-to-end classification workflow: data exploration,
│                                                  data leakage demonstration, stratified data splitting,
│                                                  model training, probability analysis, and model persistence
└── iris_classification_RF                      → Serialized Random Forest model (joblib)
```

---

## 🚀 Beyond the Original Tutorial

This project was initially developed by following the tutorial referenced below. However, to deepen my understanding and avoid "tutorial hell," I continued experimenting independently by adding:

- **Data leakage proof:** Explicitly testing specific row indices (0, 50, 100) on an overfitted model to visualize how a machine "memorizes" data.
- **Stratified sampling:** Implemented `stratify=y` inside `train_test_split` to ensure mathematically fair class proportions across both training and test sets.
- **Model saving:** Extended the original tutorial by saving the model using `joblib`.

---

## 💡 What I Learned

Throughout this project, I learned:

- How to build and train a Random Forest Classifier.
- The critical importance of the train-test split, and why feeding 100% of the data to `fit()` ruins model integrity.
- How `random_state` ensures reproducibility in data shuffling.
- How to interpret `predict_proba` to understand multi-class confidence levels.
- That understanding the *why* behind the code is far more valuable than simply writing the code itself.

---

## 🔮 Future Improvements

- [ ] Apply hyperparameter tuning to observe if a pruned Random Forest performs differently on this small dataset.
- [ ] Experiment with other classification algorithms like Support Vector Machines (SVM) or Logistic Regression.
- [ ] Visualize the decision boundaries of the Random Forest model using Matplotlib.

---

## 📚 Learning Resources

This project was inspired by educational content from **Data Professor**:

- 📺 [Machine Learning in Python: Building a Classification Model](https://youtu.be/XmSlFPDjKdc) (YouTube)
- 📺 [Data Professor YouTube channel](https://www.youtube.com/@DataProfessor)

---

## 🙏 Acknowledgements

Sincere thanks to **Data Professor** for creating straightforward and accessible educational content for Machine Learning beginners. This project uses his tutorial as the baseline structure, which I then expanded upon to solidify my own understanding of ML evaluation metrics and proper data splitting.

---

## 👤 Author

**Roihan Saputra**
*Aspiring Machine Learning Engineer*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

*"Learning by building, improving through experimentation."* 🚀

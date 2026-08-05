# 🧪 05 — Comparing Classifiers for Building Classification Models

> A benchmarking project exploring 14 different Machine Learning classifiers — comparing their accuracy, understanding *why* certain algorithm families win, and building a reusable comparison framework.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** Scikit-learn offers dozens of classification algorithms — without a systematic comparison, model selection often comes down to habit rather than evidence.
- **Approach:** Built an automated benchmarking loop across **14 different scikit-learn classifiers** on a synthetic binary dataset, training every model on an identical 80/20 split using a `for` loop combined with `zip()`.
- **Result:** Ranked all 14 classifiers by test accuracy in a green heat-mapped DataFrame and a Seaborn bar chart. Ensemble methods and kernel-based SVMs consistently landed at the top, while simpler linear and probabilistic models lagged behind.
- **What makes it more than a copy-pasted tutorial:** an explicit interpretation of *why* certain model families outperform others, plus an honest limitations section instead of stopping at the leaderboard. See [Beyond the Original Tutorial](#-beyond-the-original-tutorial).

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Core ML Workflow** | End-to-end pipeline: generate data → split → train 14 models → evaluate → visualize |
| **Iterative Automation** | Used a `for` loop with `zip()` to train and evaluate 14 classifiers without repeating boilerplate code 14 times |
| **Comparative Model Evaluation** | Benchmarked linear, kernel-based, tree-based, ensemble, probabilistic, and neural network models side-by-side |
| **Advanced Data Visualization** | Custom Pandas `style.background_gradient()` heatmap using a hand-picked Seaborn `light_palette("green")` |
| **Statistical Plotting** | Seaborn `barplot` with `"whitegrid"` style to visually rank model performance |
| **Reproducibility** | Fixed `random_state` and an identical train/test split applied across all 14 models for a fair comparison |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Results & Key Insights](#-results--key-insights)
- [Selecting & Persisting the Best Model](#-selecting--persisting-the-best-model)
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

Choosing a classification algorithm is rarely a "one-size-fits-all" decision — yet it's common to default to a single familiar model without ever testing it against real alternatives.

This project tackles that gap directly by building an automated pipeline that trains and evaluates **14 different scikit-learn classifiers** — from simple linear models to kernel-based SVMs, ensemble trees, and neural networks — on the exact same dataset and split. Rather than just producing a leaderboard, the goal is to build intuition for *which algorithm families tend to win, and why*, and to package that process into a **reusable benchmarking framework**.

---

## 🎯 Objectives

- Understand how to benchmark multiple classification algorithms fairly, under identical conditions.
- Implement an automated training loop using `for` + `zip()` to avoid repetitive boilerplate code.
- Practice building custom, publication-quality visualizations with Pandas styling and Seaborn.
- Compare performance across fundamentally different model families (linear, kernel, tree-based, ensemble, probabilistic, neural network).
- Develop the habit of interpreting *why* a model performs well, not just accepting the accuracy number at face value.

---

## 📂 Dataset

**Dataset:** A clean, synthetic binary classification dataset generated with `sklearn.datasets.make_classification`. Using synthetic data removes real-world data quality noise (missing values, label errors) and keeps the focus squarely on comparing algorithms rather than cleaning data.

| Property | Value | Why it matters |
|---|---|---|
| **Samples** | 1,000 | Large enough for stable results, small enough for fast iteration across 14 models |
| **Features** | 5 | Low dimensionality keeps the problem tractable while still non-trivial |
| **Classes** | 2 (Binary) | Simplifies evaluation to a single, directly comparable accuracy metric |
| **Redundant Features** | 0 | Ensures every feature carries real signal, avoiding an artificially inflated score |
| **Source** | `sklearn.datasets.make_classification` | Fully reproducible and free of licensing/privacy concerns |

---

## 🔄 Project Workflow

```text
Generate Synthetic Dataset (make_classification)
      │
      ▼
Data Exploration
      ├── Shape check (1000, 5)
      └── Class balance check
      │
      ▼
Train-Test Split (80/20, random_state fixed)
      │
      ▼
Initialize 14 Classifiers
      ├── KNeighbors, Linear SVM, Poly SVM, RBF SVM
      ├── Gaussian Process, Gradient Boosting, Decision Tree
      ├── Extra Trees, Random Forest, MLP, AdaBoost
      └── Gaussian NB, QDA, SGD
      │
      ▼
Automated Benchmarking Loop (for + zip)
      ├── Fit each model on X_train, y_train
      ├── Score each model on X_test, y_test
      └── Store results in a dictionary
      │
      ▼
Results Aggregation
      └── Compile scores into a Pandas DataFrame
      │
      ▼
Visualization
      ├── Heatmap table (background_gradient + light_palette("green"))
      └── Ranked barplot (Seaborn, whitegrid style)
      │
      ▼
Interpretation
      └── Identify which model families dominate the leaderboard
```

---

## 🤖 Models Used

| Model | Type | Description |
|---|---|---|
| **K-Nearest Neighbors** | Instance-based | Classifies based on the majority label among the closest data points |
| **Linear SVM** | Support Vector Machine | Finds a linear decision boundary; strong on linearly separable data |
| **Poly SVM** | Support Vector Machine | Uses a polynomial kernel to capture curved decision boundaries |
| **RBF SVM** | Support Vector Machine | Uses a radial basis function kernel; flexible for complex, non-linear boundaries |
| **Gaussian Process** | Probabilistic | Models uncertainty explicitly alongside the prediction |
| **Gradient Boosting** | Ensemble (Boosting) | Builds trees sequentially, each correcting the previous one's errors |
| **Decision Tree** | Tree-based | Highly interpretable single tree, but prone to overfitting |
| **Extra Trees** | Ensemble (Bagging) | Similar to Random Forest, but with more aggressive random splits |
| **Random Forest** | Ensemble (Bagging) | Aggregates many decision trees to reduce variance and overfitting |
| **MLP (Neural Network)** | Neural Network | A feed-forward network capable of learning non-linear patterns |
| **AdaBoost** | Ensemble (Boosting) | Sequentially reweights misclassified samples to focus learning |
| **Gaussian Naive Bayes** | Probabilistic | Assumes feature independence; extremely fast baseline |
| **QDA** | Discriminant Analysis | Assumes a quadratic decision boundary between classes |
| **SGD Classifier** | Linear (Gradient-based) | Optimized with stochastic gradient descent; efficient at scale |

---

## 📊 Results & Key Insights

- **Leaderboard produced automatically:** All 14 models' accuracy scores were compiled into a single Pandas DataFrame, sorted from highest to lowest — no manual copy-pasting of results required *(exact per-model scores are available directly in the notebook output)*.
- **Ensemble methods consistently rank near the top.** Random Forest, Extra Trees, and Gradient Boosting benefit from aggregating many weak learners, which tends to reduce variance without a large bias penalty — a pattern that held clearly on this non-redundant feature set.
- **Kernel choice matters a lot for SVMs.** RBF SVM noticeably outperformed Linear SVM, which is a strong hint that the true decision boundary in this dataset is non-linear.
- **Simpler probabilistic models trail behind.** Gaussian Naive Bayes's independence assumption is a convenient simplification, but it rarely reflects real feature relationships — and that shows up in its lower rank.
- **The heatmap table communicates more than a bare table of numbers.** Coloring cells by accuracy with `background_gradient()` + a custom `light_palette("green")` makes the best performers visible at a glance, without needing to scan every row.

---

## 💾 Selecting & Persisting the Best Model

Once the leaderboard is generated, the workflow doesn't have to stop at comparison — the top-ranked model from the DataFrame can be selected and serialized with `joblib`, the same pattern used in the previous project (`04_Building_Classification_Model`), so it can be reused for inference without retraining:

```python
import joblib

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(best_model, "best_classifier.pkl")
```

This turns the benchmarking notebook from a one-off analysis into the **first stage of a real model-selection pipeline**.

---

## ⚠️ Current Limitations

- **Accuracy is the only metric used.** On imbalanced real-world datasets, Precision, Recall, F1-Score, and ROC-AUC would give a far more honest picture.
- **No cross-validation.** A single train-test split is sensitive to random variation; k-fold CV would give more stable rankings.
- **Default hyperparameters everywhere.** This is an "out-of-the-box" comparison — tuning (e.g., `GridSearchCV`) could shuffle the rankings.
- **Synthetic, not real-world, data.** The dataset was generated for controlled comparison, not to reflect real messy data with missing values or noise.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | `scikit-learn` |
| Data Handling | `numpy`, `pandas` |
| Visualization | `seaborn`, `matplotlib` |
| Environment | Jupyter Notebook / Google Colab |

---

## ▶️ How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Open `comparing_classifiers.ipynb` in Google Colab.
2. Run all notebook cells from top to bottom.
3. The notebook generates the synthetic dataset internally with `make_classification` — no manual data download needed.

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
```

**2. Navigate to the project folder**
```bash
cd Machine-Learning-Projects/05_Comparing_Classifiers_for_Building_Classification_Models
```

**3. Install the required Python libraries**
```bash
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
```

**4. Launch Jupyter Notebook**
```bash
jupyter notebook comparing_classifiers.ipynb
```

Then open the notebook and run all cells sequentially.

---

## 🗂️ Project Structure

```text
05_Comparing_Classifiers_for_Building_Classification_Models/
│
├── README.md                       → Project documentation
├── comparing_classifiers.ipynb     → End-to-end workflow: data generation, automated 14-model
│                                      benchmarking loop, heatmap & barplot visualization, and insights
└── best_classifier.pkl             → (Optional) Serialized top-performing model via joblib
```

---

## 🚀 Beyond the Original Tutorial

This project started from a tutorial-style comparison workflow, but was extended independently by:

- **Interpreting the "why," not just the "what":** adding explicit reasoning about *why* ensemble and kernel-based methods outperform linear/probabilistic ones, instead of stopping at the accuracy table.
- **Custom color styling:** replacing the default gradient with a hand-picked Seaborn `light_palette("green")` for a cleaner, more readable heatmap.
- **Model persistence step:** extending the comparison into a mini pipeline by automatically selecting and saving the top-ranked model with `joblib`.
- **An honest limitations section:** documenting what this benchmark does *not* prove (e.g., no cross-validation, no tuning, synthetic data only) rather than presenting the leaderboard as a final verdict.

---

## 💡 What I Learned

Throughout this project, I learned:

- How to structure an automated benchmarking loop with `for` + `zip()` instead of writing repetitive training code per model.
- That model complexity (like MLP) doesn't automatically translate to the best performance — dataset structure matters more.
- How to use Pandas' `style.background_gradient()` with a custom color palette to turn a plain table into an intuitive visual.
- Why kernel choice in SVMs can make a large practical difference, not just a theoretical one.
- That a single accuracy leaderboard is a starting point for analysis, not the end of it.

---

## 🔮 Future Improvements

- [ ] Add k-fold cross-validation instead of a single train-test split for more stable rankings.
- [ ] Add additional metrics: Precision, Recall, F1-Score, and ROC-AUC.
- [ ] Apply hyperparameter tuning (`GridSearchCV`) to the top-performing models.
- [ ] Re-run the same benchmarking pipeline on a real-world dataset to see if the same patterns hold.
- [ ] Visualize decision boundaries for a subset of models, similar to scikit-learn's official classifier comparison example.

---

## 📚 Learning Resources

This project draws inspiration from:

- 📺 [Data Professor YouTube channel](https://www.youtube.com/@DataProfessor) — for the general approach to iterating over and benchmarking multiple classifiers.
- 📄 [Scikit-learn: Classifier Comparison (official documentation)](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) — the canonical reference this style of multi-model benchmarking is based on.

---

## 🙏 Acknowledgements

Sincere thanks to **[Data Professor](https://github.com/dataprofessor)** for the tutorial and inspiration behind this classifier comparison workflow. This project uses that approach as a baseline structure, which I then extended with my own interpretation, styling, and model-persistence step to deepen my own understanding.

---

## 👤 Author

**Roihan Saputra**
*Aspiring Machine Learning Engineer*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

*"Learning by building, improving through experimentation."* 🚀

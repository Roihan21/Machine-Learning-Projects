# 🧪 Predicting Molecular Solubility using Machine Learning

> *My first end-to-end Machine Learning project, where I learned how to build, evaluate, and improve regression models beyond the original tutorial.*

---

# 📖 Project Overview

This project predicts the aqueous solubility (`logS`) of molecules using Machine Learning regression models.

Molecular solubility is an important physicochemical property in chemistry and pharmaceutical research because it helps determine whether a molecule has the potential to become a drug candidate.

This project marks an important milestone in my Machine Learning journey. Rather than simply training a model, my goal was to understand the complete workflow of a regression problem—from loading the dataset to evaluating and improving model performance.

---

# 🎯 Objectives

The objectives of this project are:

- Understand the end-to-end Machine Learning workflow for regression problems.
- Learn how different regression models perform on the same dataset.
- Evaluate model performance using regression metrics.
- Improve model performance through Hyperparameter Tuning.
- Practice documenting a Machine Learning project as part of my portfolio.

---

# 📂 Dataset

**Dataset:** Delaney Solubility Dataset

### Target Variable

- `logS` (Aqueous Solubility)

### Input Features

- MolLogP
- MolWt
- NumRotatableBonds
- AromaticProportion

The dataset contains molecular descriptors that are commonly used to predict the aqueous solubility of chemical compounds.

---

# 🔄 Project Workflow

```text
Load Dataset
      │
      ▼
Data Exploration
      │
      ▼
Train-Test Split
      │
      ▼
Linear Regression
      │
      ▼
Random Forest
      │
      ▼
Hyperparameter Tuning
(RandomizedSearchCV)
      │
      ▼
Model Evaluation
(MSE & R² Score)
      │
      ▼
Model Comparison
      │
      ▼
Prediction Visualization
      │
      ▼
Conclusion
```

---

# 🤖 Models Used

| Model | Description |
|--------|-------------|
| Linear Regression | Baseline regression model |
| Random Forest | Ensemble regression model |
| Random Forest (RandomizedSearchCV) | Tuned Random Forest using RandomizedSearchCV |

---

# 📊 Evaluation Metrics

The models were evaluated using:

- Mean Squared Error (MSE)
- R² Score

These metrics were used to compare model performance before and after Hyperparameter Tuning.

---

# 📈 Results

| Model | Train MSE | Train R² | Test MSE | Test R² |
|--------|----------:|---------:|---------:|---------:|
| Linear Regression | 1.008 | 0.765 | 1.021 | 0.789 |
| Random Forest | 1.028 | 0.760 | 1.408 | 0.709 |
| **Random Forest (RandomizedSearchCV)** | **0.303** | **0.929** | **0.676** | **0.860** |

### Key Findings

- Linear Regression provided a solid baseline.
- Default Random Forest did not outperform Linear Regression on the test dataset.
- After applying **RandomizedSearchCV**, the Random Forest model achieved the best overall performance.
- Hyperparameter Tuning significantly improved the model's predictive ability.

---

# 📉 Visualizations

Project visualizations are available inside the **images/** folder.

Current visualizations include:

- Linear Regression Prediction
- Random Forest Prediction
- Tuned Random Forest Prediction
- Model Performance Comparison

---

# 🚀 Beyond the Original Tutorial

This project was initially developed by following the tutorial below until the **Prediction Visualization** stage.

To deepen my understanding, I continued experimenting independently by adding:

- RandomizedSearchCV
- Hyperparameter Tuning
- Tuned Model Evaluation
- Updated Model Comparison
- Additional Documentation

These additions helped me better understand how model optimization can improve Machine Learning performance.

---

# 📚 Learning Resources

This project was inspired by the educational content created by **Data Professor**.

### 📺 YouTube Tutorial

**Build your first machine learning model in Python**

https://youtu.be/29ZQ3TDGgRQ?si=bWJpY_14MABtvXVg

### 📺 YouTube Channel

Data Professor

https://www.youtube.com/@DataProfessor

### 💻 GitHub Repository

https://github.com/dataprofessor

### 📂 Dataset Source

https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv

---

# 💡 What I Learned

Through this project, I learned how to:

- Build regression models using Scikit-Learn.
- Compare multiple regression algorithms.
- Evaluate models using MSE and R² Score.
- Apply Hyperparameter Tuning with RandomizedSearchCV.
- Compare baseline and optimized models.
- Organize a Machine Learning project using GitHub.

More importantly, this project taught me that building a Machine Learning model is not only about obtaining better metrics, but also about understanding **why** a model performs better and how different techniques affect its performance.

---

# 🔮 Future Improvements

As I continue learning Machine Learning, I plan to improve this project by exploring:

- Feature Engineering
- Feature Importance Analysis
- XGBoost
- CatBoost
- Explainable AI (SHAP)
- Additional Regression Models

---

# 🙏 Acknowledgements

I would like to sincerely thank **Data Professor** for creating high-quality educational content and making Machine Learning more accessible to beginners.

This project was inspired by the tutorial and educational resources provided by Data Professor. After completing the tutorial, I independently extended the project by experimenting with Hyperparameter Tuning, additional model evaluation, and improved documentation as part of my own learning journey.

Thank you for sharing valuable knowledge with the Machine Learning community.

---

# 👨‍💻 Author

**Roihan**

Aspiring Machine Learning Engineer

*"Learning by building, improving through experimentation."* 🚀

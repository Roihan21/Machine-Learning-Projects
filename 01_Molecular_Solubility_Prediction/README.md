# 🧪 Predicting Molecular Solubility using Machine Learning

## 📖 Project Overview

This project aims to predict the aqueous solubility (`logS`) of molecules using Machine Learning regression models.

Molecular solubility is an important physicochemical property in chemistry and pharmaceutical research because it helps determine whether a molecule has the potential to become a drug candidate.

This project is part of my Machine Learning learning journey and focuses on understanding the complete workflow of a regression problem.

---

# 🎯 Objectives

- Learn the Machine Learning workflow for regression problems.
- Compare different regression algorithms.
- Understand model evaluation using regression metrics.
- Explore Hyperparameter Tuning to improve model performance.

---

# 📂 Dataset

Dataset Name:

**Delaney Solubility Dataset**

Target Variable:

- `logS`

Features:

- MolLogP
- MolWt
- NumRotatableBonds
- AromaticProportion

---

# 🏗️ Project Workflow

```text
Load Dataset
        ↓
Data Exploration
        ↓
Train-Test Split
        ↓
Linear Regression
        ↓
Random Forest
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Visualization
        ↓
Conclusion
```

---

# 🤖 Models Used

- Linear Regression
- Random Forest
- Random Forest (RandomizedSearchCV)

---

# 📊 Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score

---

# 📈 Results

| Model | Train MSE | Train R² | Test MSE | Test R² |
|--------|----------:|---------:|---------:|---------:|
| Linear Regression | 1.008 | 0.765 | 1.021 | 0.789 |
| Random Forest | 1.028 | 0.760 | 1.408 | 0.709 |
| Random Forest (Randomized Tuned) | **0.303** | **0.929** | **0.676** | **0.860** |

---

# 📉 Visualizations

Project visualizations can be found inside the **images/** folder.

Examples:

- Actual vs Predicted
- Model Comparison
- Residual Analysis *(Coming Soon)*

---

# 🚀 Improvements

The original tutorial stopped after the prediction visualization step.

To further explore the project, I independently added:

- RandomizedSearchCV
- Hyperparameter Tuning
- Tuned Model Evaluation
- Model Performance Comparison
- Additional Analysis

---

# 📚 Learning Resources

This project was inspired by the following educational resources.

### YouTube Tutorial

**Build your first machine learning model in Python**

https://youtu.be/29ZQ3TDGgRQ?si=bWJpY_14MABtvXVg

### YouTube Channel

Data Professor

https://www.youtube.com/@DataProfessor

### GitHub

https://github.com/dataprofessor

### Dataset

https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv

---

# 💡 What I Learned

Throughout this project I learned:

- Building regression models with Scikit-Learn
- Comparing Linear Regression and Random Forest
- Using MSE and R² for model evaluation
- Applying RandomizedSearchCV
- Understanding how hyperparameter tuning improves model performance
- Comparing baseline and optimized models

---

# 🔮 Future Improvements

Some improvements I would like to explore in the future:

- Feature Engineering
- Cross Validation
- Feature Importance Analysis
- SHAP Explainability
- XGBoost Comparison
- CatBoost Comparison

---

# 🙏 Acknowledgements

I would like to express my sincere gratitude to **Data Professor** for creating high-quality educational content and openly sharing learning resources with the community.
His tutorial provided a strong foundation for this project and helped me understand the end-to-end Machine Learning workflow.
This project extends the original tutorial by adding Hyperparameter Tuning, additional model evaluation, and further experimentation as part of my own learning journey.
Thank you for making Machine Learning education accessible to everyone.

---

# 👨‍💻 Author

**Roihan**

Machine Learning Portfolio

Learning • Building • Improving 🚀

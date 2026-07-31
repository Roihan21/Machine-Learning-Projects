# 🏦 Customer Transaction Segmentation with K-Means Clustering

> An unsupervised machine learning project that segments bank customers into distinct behavioral groups based on their transaction patterns — turning raw transactional data into actionable business personas.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 1. Project Overview

Financial institutions sit on top of huge volumes of transaction data, but raw numbers alone don't tell a story that a marketing or risk team can act on. This project applies **unsupervised learning (K-Means Clustering)** to a bank transaction dataset (2,500+ records) in order to automatically discover **natural customer segments** — without any pre-existing labels.

**Why does this matter for the business?**
Instead of treating every customer the same, this model groups customers by shared behavior (spending habits, account balance, age, occupation, transaction activity). These segments can then power:
- 🎯 Targeted marketing campaigns
- 💳 Personalized product recommendations (savings vs. investment vs. credit)
- 🛡️ Early-stage anomaly / fraud-pattern monitoring
- 📈 Input labels for a downstream supervised classification model

The final deliverable is a trained, reusable clustering model plus two clean datasets — one machine-ready (scaled) and one business-ready (original real-world values) — so both engineers and non-technical stakeholders get what they need.

---

## 🗂️ 2. Repository Structure

| File | Description |
|---|---|
| `[Clustering]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb` | 📓 Main notebook — full end-to-end pipeline from raw data to business interpretation |
| `data_clustering.csv` | ⚙️ Preprocessed & **scaled** dataset (machine-learning-ready format), including the resulting `Target` cluster label |
| `data_clustering_inverse.csv` | 💼 **Business-ready** dataset — scaled values converted back to their original real-world units (age in years, balance in currency, etc.) |
| `model_clustering.h5` | 🤖 Trained K-Means model (main model, fitted on the full preprocessed feature set) |
| `PCA_model_clustering.h5` | 🧭 K-Means model trained on the 2D PCA-reduced feature space (used for benchmarking/visualization) |

---

## 🧰 3. Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Environment | Jupyter Notebook |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (`StandardScaler`, `LabelEncoder`, `KMeans`, `PCA`, `silhouette_score`) |
| Cluster Selection | `yellowbrick` (`KElbowVisualizer`) |
| Model Persistence | `joblib` |

---

## 🔄 4. Machine Learning Pipeline

### Step 1 — Data Understanding
The raw dataset contains **2,537 transaction records** across **16 attributes** (transaction amount, account balance, customer age, occupation, channel, login attempts, etc.), originally designed for transaction-behavior analysis.

### Step 2 — Data Cleaning
- Checked and removed missing values and **21 duplicate rows**
- Dropped identifier-type columns (`TransactionID`, `AccountID`, `DeviceID`, `MerchantID`, `IP Address`) and raw date columns — these are unique per row and carry no *behavioral* signal for clustering
- Removed statistical outliers using the **IQR (Interquartile Range) method**, so a handful of extreme transactions don't distort the cluster shapes
- Result: a clean dataset of **1,945 records** ready for modeling

### Step 3 — Feature Encoding
Categorical columns (`TransactionType`, `Location`, `Channel`, `CustomerOccupation`) were converted into numbers using `LabelEncoder`, since K-Means can only do math on numeric values, not text labels.

### Step 4 — Feature Scaling with `StandardScaler` 📏

**Why is this necessary?** Imagine comparing two runners: one race measured in *kilometers* and one measured in *centimeters*. Even if both runners performed equally well relative to their own race, the "centimeters" numbers will always look astronomically bigger — making comparisons meaningless.

The same problem happens here: `AccountBalance` can be in the thousands, while `LoginAttempts` is a small number like 1–5. Without scaling, K-Means (which measures *distance* between points) would think `AccountBalance` is "more important" simply because its numbers are bigger — not because it's actually more meaningful.

`StandardScaler` fixes this by transforming every numeric feature to have a **mean of 0** and a **standard deviation of 1**, putting all features on the same playing field so the clustering algorithm treats them fairly.

### Step 5 — Dimensionality Reduction with `PCA` 🧭

With 9 features after cleaning, we're technically working in 9-dimensional space — impossible to plot or visually inspect. **PCA (Principal Component Analysis)** compresses this into just 2 new components (`Principal Component 1` and `2`) that capture as much of the original data's variance (spread/pattern) as possible.

Think of it like taking a **flat photograph of a 3D sculpture**: you lose some depth information, but a well-chosen camera angle still lets you clearly see the sculpture's overall shape. PCA does the same thing — it finds the "best camera angle" to flatten high-dimensional data into 2D so we can *visually* confirm that the clusters our model found are actually separated in space.

### Step 6 — Choosing the Right Number of Clusters (K)

Two complementary techniques were used to avoid guessing K arbitrarily:

- **📉 Elbow Method** (`KElbowVisualizer`, tested K = 2 to 10): plots a performance score against different values of K. The "elbow" — the point where adding more clusters stops giving meaningful improvement — pointed clearly to **K = 2**.
- **✅ Silhouette Score**: measures how well-separated and internally consistent the clusters are (scored from -1 to 1, where higher is better). The final model achieved a **Silhouette Score of 0.572**, indicating reasonably strong, well-defined cluster separation — well above the "weak structure" threshold.

### Step 7 — Model Training (K-Means)
A `KMeans(n_clusters=2, random_state=42)` model was trained on the cleaned, scaled feature set. `random_state=42` ensures the results are **reproducible** — anyone re-running the notebook gets identical clusters.

### Step 8 — Model Persistence
Both the main K-Means model and a secondary PCA-based K-Means model were saved with `joblib` as `.h5` files, so they can be reloaded instantly for inference or further analysis without retraining.

---

## 📊 5. Visualization Explained

- **Elbow / Silhouette plot** — shown during Step 6, this chart is how we objectively justified choosing 2 clusters instead of picking a number arbitrarily.
- **2D PCA Scatter Plot** — each point represents one customer, colored by its assigned cluster (0 or 1), with red **✕** markers showing each cluster's **centroid** (its mathematical center/"average customer"). This is the fastest way to visually sanity-check that K-Means actually found two meaningfully distinct, non-overlapping groups rather than an arbitrary split.

---

## 💡 6. Business Insights & Recommendations
*(Derived from `data_clustering_inverse.csv` — see the crucial explanation below)*

### ⚠️ Why We MUST Inverse-Transform Before Business Interpretation

This is the single most important step for turning a data science output into something a business team can actually use — and it's easy to get wrong.

After `StandardScaler`, our data looks like this: `CustomerAge: 0.02`, `AccountBalance: 0.01`. These numbers are **not** ages or currency amounts — they're statistical z-scores describing "how many standard deviations away from average." **A marketing manager cannot design a campaign around "0.02 years old."**

To make the results usable, we apply `scaler.inverse_transform()` to convert scaled numbers back into their real-world units (actual age in years, actual balance in currency), and `encoder.inverse_transform()` to turn encoded categories back into readable labels (e.g., `0` → `"Doctor"`). Only *after* this step do the clusters become something a business stakeholder can read, trust, and act on.

The difference is dramatic — here's the same Cluster 0 described two ways:

| | Before Inverse (Scaled) | After Inverse (Real Values) |
|---|---|---|
| Average Age | `0.02` | **45.06 years** |
| Average Account Balance | `0.01` | **$5,142.17** |
| Average Transaction Amount | `-0.01` | **$255.55** |
| Business readability | ❌ Meaningless to non-technical staff | ✅ Immediately actionable |

### 🧑‍💼 Cluster Profiles (Real-World Values)

**🟦 Cluster 0 — "Established Professionals"**
- **Average Age:** 45.06 years
- **Average Account Balance:** $5,142.17
- **Average Transaction Amount:** $255.55
- **Dominant Profile:** Mature age group (`Dewasa`/Adult bracket), largely represented by high-tier professions such as **Doctors**
- **Behavior:** Maintains a comparatively higher standing balance with slightly lower average transaction spend — consistent with a saver-oriented, financially stable customer base.

**🟨 Cluster 1 — "Young & Active Customers"**
- **Average Age:** 44.33 years
- **Average Account Balance:** $5,058.81
- **Average Transaction Amount:** $258.15
- **Dominant Profile:** Overwhelmingly represented by the "Young" age bracket and **Student** occupations
- **Behavior:** Transacts slightly more frequently/actively and keeps a lower resting balance — consistent with a spender-oriented, transaction-driven customer base.

> 📝 **Interesting nuance:** the two clusters' *average numeric* ages look nearly identical (45.06 vs. 44.33) — if you only looked at the mean, you might conclude age doesn't matter. But the *categorical mode* (most frequent value) tells a completely different, more useful story: one cluster is dominated by Students, the other by Doctors. This is a great example of why business analysis should never rely on averages alone.

### 📢 Business Recommendations

1. **For Cluster 0 (Established Professionals):** Prioritize wealth-management products, fixed deposits, and premium/priority banking offers. This segment values stability over frequent activity — engagement strategies should focus on trust, exclusivity, and long-term relationship building.
2. **For Cluster 1 (Young & Active Customers):** Focus on digital-first engagement — cashback promos, student banking bundles, and micro-savings or financial literacy programs. This segment is more transaction-active and represents strong potential for building long-term loyalty early.
3. **Operational use:** Because the clustering labels (`Target`) are already embedded in `data_clustering.csv`, this dataset can directly feed a **supervised classification model** to automatically predict which segment a *new* customer belongs to in real time.

---

## 🚀 7. How to Run

### Prerequisites
```bash
Python 3.10+
Jupyter Notebook / JupyterLab
```

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Set up environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook "[Clustering]_Submission_Akhir_BMLP_Roihan_Saputra.ipynb"
```

> 💡 The dataset is loaded automatically from a public source directly within the notebook (via `pd.read_csv(url)`) — no manual download required. Simply run all cells from top to bottom.

### 4. Reuse the trained model
```python
import joblib

model = joblib.load("model_clustering.h5")
predicted_cluster = model.predict(new_scaled_customer_data)
```

---

## 📈 8. Model Evaluation Summary

| Metric | Value |
|---|---|
| Optimal Number of Clusters (K) | 2 |
| Method Used to Determine K | Elbow Method (`KElbowVisualizer`) |
| Silhouette Score | **0.572** |
| Final Dataset Size (post-cleaning) | 1,945 records |
| Dimensionality Reduction | PCA (9 features → 2 components) |

---

## 👤 Author

**Roihan Saputra**
*Data Science & Machine Learning Enthusiast*

---
*This project was developed as part of a Machine Learning learning path (Unsupervised Learning / Clustering track), with results structured to support a downstream classification modeling task.*

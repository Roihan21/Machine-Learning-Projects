# 🏦 Customer Transaction Segmentation with K-Means Clustering

> An unsupervised machine learning project that segments bank customers into statistically distinct behavioral groups based on their transaction patterns — turning raw transactional data into a testable, actionable starting point for targeted business strategy.

![Dicoding Status](https://img.shields.io/badge/Submission-ACCEPTED-brightgreen?style=for-the-badge&logo=dicoding)
![Grade](https://img.shields.io/badge/Grade-ADVANCED-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)

*Final Submission for Dicoding Indonesia — "Belajar Machine Learning untuk Pemula" (Build Machine Learning Project)*

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** A bank has thousands of transaction records but no existing way to group customers by behavior.
- **Approach:** Clean & encode the raw data → scale it with `StandardScaler` → reduce it to 2D with `PCA` → find the right number of groups with the Elbow Method → cluster with `KMeans` → validate with a Silhouette Score → inverse-transform the results back into real-world business terms.
- **Result:** 2 statistically well-separated customer segments (Silhouette Score 0.572) discovered from 1,945 cleaned records, with a near-perfect 50/50 split — output feeds directly into a downstream classification model.
- **What makes it stand out:** the write-up doesn't stop at "clusters found" — it explains exactly how strong the persona differences actually are, and flags the modeling choices worth revisiting. See **["Honest Read of the Numbers"](#-honest-read-of-the-numbers--dont-over-sell-the-persona)** below.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Data Cleaning & EDA** | Handling missing values, 21 duplicate rows, and outliers (IQR method) on a 2,500+ row raw dataset |
| **Feature Engineering** | `LabelEncoder` for categorical variables, with a clear rationale for what was dropped and why (IDs, addresses, raw dates carry no behavioral signal) |
| **Feature Scaling** | `StandardScaler`, explained through a concrete analogy of why unscaled distance-based algorithms mislead |
| **Dimensionality Reduction** | `PCA` to 2D, used both to visualize cluster separation and as a benchmark model in its own right |
| **Unsupervised Learning** | `KMeans` clustering, with **K** chosen deliberately via the Elbow Method rather than guessed |
| **Model Validation** | Silhouette Score used to statistically confirm cluster quality, not just visual inspection |
| **Business Translation** | Inverse-transforming scaled output back into real-world units (age, currency, category) so non-technical stakeholders can act on it |
| **Statistical Honesty** | Explicitly quantifying *how large* the persona differences really are, and calling out modeling choices (e.g., a constant feature, a high-cardinality label-encoded column) that deserve a second look |

---

## 📖 1. Project Overview

Financial institutions sit on top of huge volumes of transaction data, but raw numbers alone don't tell a story that a marketing or risk team can act on. This project applies **unsupervised learning (K-Means Clustering)** to a bank transaction dataset (2,500+ records) in order to automatically discover **natural customer segments** — without any pre-existing labels.

**Why does this matter for the business?**
Instead of treating every customer the same, this model groups customers by shared behavior (spending habits, account balance, age, occupation, transaction activity). These segments can then power:
- 🎯 Targeted marketing campaigns
- 💳 Personalized product recommendations (savings vs. investment vs. credit)
- 🛡️ Early-stage anomaly / fraud-pattern monitoring
- 📈 Input labels for a downstream supervised classification model

The final deliverable is a trained, reusable clustering model plus two clean datasets — one machine-ready (scaled) and one business-ready (original real-world values) — so both engineers and non-technical stakeholders get what they need.

---

## 📁 2. Repository Structure

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
*(Computed directly from `data_clustering_inverse.csv` — 1,945 records)*

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

### ⚖️ Cluster Sizes

| Cluster | Customers | Share |
|---|---|---|
| 🟦 Cluster 0 | 980 | 50.4% |
| 🟨 Cluster 1 | 965 | 49.6% |

The two segments are almost perfectly balanced — this isn't a "small niche vs. the majority" situation, it's a genuine 50/50 split of the customer base, meaning both segments deserve equal strategic attention.

### 🧑‍💼 Cluster Profiles (Real-World Values)

| Metric (mean) | 🟦 Cluster 0 | 🟨 Cluster 1 |
|---|---|---|
| Customer Age | 45.06 yrs | 44.33 yrs |
| Account Balance | $5,142.17 | $5,058.81 |
| Transaction Amount | $255.55 | $258.15 |
| Transaction Duration | 121.1 sec | 117.3 sec |
| Most common Age Group | Dewasa (Adult) | Muda (Young) |
| Most common Occupation | Doctor | Student |

**🟦 Cluster 0 — "Balance-Leaning Segment"**: slightly older on average, holds a marginally higher account balance, and its single most common (mode) occupation is Doctor.

**🟨 Cluster 1 — "Activity-Leaning Segment"**: slightly younger on average, transacts marginally more, and its single most common (mode) occupation is Student.

### 📝 Honest Read of the Numbers — Don't Over-Sell the Persona

This is worth being upfront about, because it's exactly the kind of nuance that separates a trustworthy analysis from a misleading one: **the gap between the two clusters on these business features is real but modest.**

- The mean differences (e.g., $83 in balance, 0.7 years in age) are small compared to how spread out each cluster's own data is (e.g., account balance has a standard deviation of ~$3,900 *within* each cluster). In plain terms: a "Doctor" and a "Student" can both easily show up in either cluster — the tilt is directional, not a hard rule.
- On `AgeGroup`, Cluster 0 is 34.0% Dewasa vs. Cluster 1 at 31.7% — a real but narrow lean, not a takeover. Occupation shows a similar pattern (Cluster 0: 27.6% Doctor vs. Cluster 1: 23.9% Doctor).
- One feature, `LoginAttempts`, turned out to be **constant (always 1)** in the cleaned dataset — likely a side effect of the IQR outlier-removal step stripping out all the higher values. It currently adds no discriminating power and is a candidate to drop or revisit in a future iteration.
- `Location` (43 cities) was converted with `LabelEncoder`, which assigns arbitrary sequential numbers (e.g., Boston = 3, Seattle = 31) with no real-world order. K-Means can still treat that arbitrary numeric distance as "meaningful," so part of what's separating the two clusters is likely this artificial structure rather than genuine geographic behavior. **A recommended next step** is re-encoding high-cardinality nominal features like this with one-hot or frequency/target encoding instead.

**Bottom line:** the clustering is statistically sound (Silhouette Score 0.572 shows the two groups *are* geometrically well-separated), but the business story is best framed as **"two behaviorally-tilted segments"** rather than **"two dramatically different customer types."** This is a common and honest outcome in real-world segmentation work — and it still provides a usable starting point for targeted campaigns.

### 📢 Business Recommendations

1. **For Cluster 0 (Balance-Leaning Segment):** Test wealth-management, fixed-deposit, and priority-banking offers here first — this group shows a mild lean toward higher balances and mature professional occupations like Doctors.
2. **For Cluster 1 (Activity-Leaning Segment):** Test digital-first engagement, cashback promos, and student-banking bundles here first — this group shows a mild lean toward younger customers and more frequent transacting.
3. **Treat this as a testable hypothesis, not a certainty.** Since the separation is statistically valid but demographically subtle, the recommended approach is a controlled **A/B campaign test** across both segments to confirm response differences before committing full marketing budget.
4. **Operational use:** Because the clustering labels (`Target`) are already embedded in `data_clustering.csv` / `data_clustering_inverse.csv`, this dataset can directly feed a **supervised classification model** to automatically predict which segment a *new* customer belongs to in real time.

---

## 🔍 7. Key Findings & Model Limitations

Being transparent about a model's limitations is what makes its results trustworthy. A few things worth flagging for anyone extending this project:

| Observation | Why it matters | Suggested next step |
|---|---|---|
| `LoginAttempts` is constant (`= 1`) after outlier removal | Contributes zero separating power to the model | Drop the column, or revisit the IQR outlier-handling threshold |
| `Location` (43 categories) uses `LabelEncoder` | Introduces artificial numeric ordering between unrelated cities | Switch to one-hot, frequency, or target encoding |
| Mean differences between clusters are small relative to within-cluster spread | Segment "personas" are directional tendencies, not sharply distinct customer types | Validate with A/B testing before large-scale campaign rollout; consider adding richer behavioral features (e.g., transaction frequency over time, spending category) to sharpen future segmentation |
| Cluster split is almost exactly 50/50 | Confirms the split isn't just isolating a small group of outliers | — |

---

## 🚀 8. How to Run

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

## 📊 9. Model Evaluation Summary

| Metric | Value |
|---|---|
| Optimal Number of Clusters (K) | 2 |
| Method Used to Determine K | Elbow Method (`KElbowVisualizer`) |
| Silhouette Score | **0.572** |
| Final Dataset Size (post-cleaning) | 1,945 records |
| Cluster Balance | 980 (50.4%) vs. 965 (49.6%) |
| Dimensionality Reduction | PCA (9 features → 2 components) |

---

## 👤 Author

**Roihan Saputra**
*Machine Learning Engineer Enthusiast*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.

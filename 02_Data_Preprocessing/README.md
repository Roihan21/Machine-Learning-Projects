# Data Preprocessing & Feature Selection (Practice)

## 📌 Project Overview
Project ini adalah bentuk latihan (*follow-along practice*) untuk memahami fundamental tahapan Pra-Pemrosesan Data (Data Preprocessing) dalam Machine Learning. Seluruh alur kode dan penjelasan dalam repositori ini diadaptasi penuh dari tutorial **[Data Professor: How to build a machine learning model in Python from scratch](https://www.youtube.com/watch?v=DctmeFx8s_k)**, yang mengambil referensi dari buku *Machine Learning with PyTorch and Scikit-Learn*.

## 🗂️ Datasets Used
Dalam notebook ini, terdapat dua jenis dataset yang digunakan untuk demonstrasi:
1. **Wine Dataset (UCI ML Repository):** Digunakan untuk mendemonstrasikan penanganan data klasifikasi, pemisahan data, dan ekstraksi fitur.
2. **Delaney Solubility Dataset:** Digunakan untuk contoh *pipeline* prediksi molekul.

## 🛠️ Key Concepts Covered
Project ini mencakup beberapa teknik esensial dalam mempersiapkan data sebelum dimasukkan ke dalam model *Machine Learning*, antara lain:
*   **Handling Missing Values:** Mengidentifikasi dan menangani nilai kosong (NaN) pada data tabular menggunakan Pandas.
*   **Categorical Data Encoding:** Melakukan *mapping* untuk fitur ordinal, *Label Encoding* untuk kelas target, dan *One-Hot Encoding* untuk fitur nominal.
*   **Data Partitioning:** Memisahkan data menjadi *Training Set* dan *Test Set* menggunakan `train_test_split`.
*   **Feature Scaling:** Membawa semua fitur ke dalam skala yang sama menggunakan *Min-Max Normalization* dan *Standardization (StandardScaler)*.
*   **Feature Selection:** 
    *   Membangun algoritma **Sequential Backward Selection (SBS)** dari nol (*from scratch*).
    *   Menilai tingkat kepentingan fitur (*Feature Importance*) menggunakan model **Random Forest**.

## 💻 Tech Stack
*   Python 3
*   Pandas & NumPy
*   Scikit-Learn
*   Matplotlib (untuk visualisasi SBS dan Feature Importance)

## 💡 Credits & Acknowledgements
This educational project is 100% based on the amazing tutorial provided by **Chanin Nantasenamat (Data Professor)**. 
*   **YouTube Video:** [(1543) How to build a machine learning model in Python from scratch](https://www.youtube.com/watch?v=DctmeFx8s_k)
*   **Original Code Repo:** [dataprofessor/machine-learning-model-from-scratch](https://github.com/dataprofessor/machine-learning-model-from-scratch)

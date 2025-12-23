## 📘 Statistical & Structural Analysis of the Quranic Text

> **A Multi-Scale Quantitative Study of Uthmani Orthography, Long-Range Dependencies, and Statistical Control.**

---

## 🧾 Project Overview

This project presents a **fully reproducible, multi-stage computational pipeline** to analyze the Quranic text in its **complete Uthmani orthography**. Unlike standard NLP studies, this analysis accounts for every diacritic, pause symbol, and orthographic variant (e.g., Alif Khanjariya).

The core objective is to determine if the **linguistic–statistical regularities** of the Quran can be explained by simple human processes, or if they represent a **highly optimized, multi-scale system** that challenges the limits of conventional text generation.

## 🔬 Key Scientific Findings
* **Zipf’s Law:** Near-perfect fit ($R^2 = 0.9972$), indicating a highly optimized linguistic distribution.
* **Long-Range Dependency:** A high **Hurst Exponent (0.884)**, suggesting a deep structural "memory" across the entire text.
* **Orthographic Integrity:** 5,000+ Null-model simulations (p-value < 0.001) prove that the distribution of diacritics is statistically distinct from random or post-hoc additions.

---

## 🏗️ Project Structure

```text
project_root/
│
├── data/
│   ├── quran.txt
│   └── arial.ttf
│
├── src/
│   ├── step1_preprocess.py
│   ├── step2_advanced_analysis.py
│   ├── step3_structural_analysis.py
│   ├── step4_anomaly_detection.py
│   ├── step5_null_models.py
│   ├── step5b_pvalues_tests.py
│   ├── step6_orthography_analysis.py
│   ├── step6b_orthography_final.py
│
├── final_results/
│   ├── step1_preprocess/
│   ├── step2_advanced_analysis/
│   ├── structural/
│   ├── anomalies/
│   ├── null_models/
│   └── orthography/
│
├── run_pipeline.py
├── requirements.txt
└── README.md


```


## 🔄 Analysis Pipeline

The analysis is implemented across **8 execution steps** for full auditability:

| Stage | Script | Analytical Focus |
| --- | --- | --- |
| **1** | `step1_preprocess.py` | Parsing, Tokenization, and Feature Extraction. |
| **2** | `step2_advanced_analysis.py` | Zipf Law, Entropy, Autocorrelation, Hurst Exponent. |
| **3** | `step3_structural_analysis.py` | Sura similarity, Hierarchical Clustering, Fractals. |
| **4** | `step4_anomaly_detection.py` | Isolation Forest, LOF, PCA-based consistency checks. |
| **5a** | `step5_null_models.py` | 5,000+ Shuffle, Markov, and Poisson simulations. |
| **5b** | `step5b_pvalues_tests.py` | Statistical Significance and Hypothesis testing. |
| **6a** | `step6_orthography_analysis.py` | Diacritics density & Orthographic variant modelling. |
| **6b** | `step6b_orthography_final.py` | Final synthesis of orthographic results. |

---

## 🧠 Scientific Principles

* **Reproducibility:** Every chart and p-value can be regenerated with one command.
* **Zero-Assumption:** The study relies purely on mathematical metrics (No theological bias).
* **Comparative Baselines:** Real data is strictly compared against probabilistic **Null Models**.

---

## ▶️ Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```

### 2️⃣ Run the Pipeline

```bash
python run_pipeline.py

```

*The pipeline will validate inputs, execute all stages, and save visualizations in the `final_results/` folder.*

---

## 🧪 Statistical Methods

The pipeline utilizes a robust suite of methods:

* **Linguistic Laws:** Zipf’s Law, Shannon Entropy.
* **Time-Series Analysis:** Hurst Exponent, Autocorrelation (ACF).
* **Machine Learning:** Isolation Forest, PCA, UMAP, Local Outlier Factor.
* **Inference:** Kolmogorov–Smirnov, Mann–Whitney U, Permutation Tests.

---

## 👤 Author & Contact

* **Researcher:** Amer Alaa El-Din Attia
* **Date:** December 2025
* **License:** Academic Use Only.

---

*Note: This repository is intended for academic research in Quantitative Linguistics and Digital Humanities.*

```





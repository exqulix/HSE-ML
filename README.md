# HSE-ML
# Anti-Money Laundering Detection — HSE Data Mining Project
> **Dataset:** [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

---

## Project Description

This project tackles the real-world problem of **detecting money laundering in financial transactions** using machine learning. The data is sourced from a synthetic IBM dataset designed to simulate banking transaction patterns, including both legitimate and fraudulent flows.

The project covers the full ML pipeline: from business problem formulation and data extraction to EDA, feature engineering, model training, threshold tuning, and comparative evaluation.

---

## Business Problem

Banks and financial institutions lose billions annually to money laundering schemes. Traditional rule-based AML systems produce excessive false positives and miss sophisticated patterns. This project explores whether supervised and unsupervised ML models can more accurately flag suspicious transactions — **minimizing missed fraud (false negatives)** while keeping false alerts manageable.

---

## Dataset

| File | Description |
|------|-------------|
| `HI-Small_Trans.csv` | High-intensity transactions |
| `LI-Small_Trans.csv` | Low-intensity transactions |
| `HI-Small_accounts.csv` | Account metadata (HI) |
| `LI-Small_accounts.csv` | Account metadata (LI) |

- **Total records used:** ~300,000 transactions  
- **Fraud rate:** ~3% (reflects real-world class imbalance)  
- **Target variable:** `Is Laundering` (0 = legitimate, 1 = fraudulent)

---

## Project Pipeline

### 1. Data Extraction
- Kaggle API integration via `kagglehub`
- Chunked reading of large CSV files to manage memory

### 2. Data Preparation
- Combined HI and LI transaction datasets
- Maintained a controlled fraud ratio (~3%)
- Merged transaction data with account metadata

### 3. Exploratory Data Analysis (EDA)
- **Payment format analysis** — Bitcoin has the highest fraud rate (~70%); ACH dominates by volume (~35%)
- **Transaction amounts** — Fraudulent transactions cluster in the $1k–$100k range (log scale)
- **Time of day** — Found a data artifact: 99% of normal transactions are timestamped at midnight → excluded timestamp from modelling
- **Entity type** — Countries and corporations are disproportionately involved in laundering schemes

### 4. Feature Engineering
- `sender_bank_fraud_rate` / `receiver_bank_fraud_rate` — criminal history per bank
- `sender_count` / `receiver_count` — account activity level
- `log_amount` — log-transformed transaction amounts
- One-hot encoding of payment formats and entity types
- `Entity Type` grouping: `Country` + `Direct` → `Other`

### 5. Models Trained

| Model | Type | Key Tuning |
|-------|------|------------|
| Logistic Regression | Supervised | `class_weight=balanced` |
| K-Nearest Neighbors | Supervised | `HalvingGridSearchCV` + threshold optimization |
| Decision Tree | Supervised | `max_depth`, `min_samples_leaf` |
| Random Forest | Supervised (ensemble) | `HalvingGridSearchCV` |
| **CatBoost** | Supervised (boosting) | `scale_pos_weight`, `depth`, `learning_rate` |
| Isolation Forest | **Unsupervised** | `contamination` = fraud rate |
| MLP (Neural Network) | Supervised (extra) | `StratifiedKFold`, early stopping, threshold search |

### 6. Evaluation
- Metrics: **Precision, Recall, F1-score, ROC-AUC**
- Primary focus: **Recall** (minimizing missed fraud)
- ROC curves and Precision-Recall curves for all models
- Confusion matrices per model
- Feature importance analysis (CatBoost)

---

## 📊 Key Results

- **CatBoost** achieved the best overall performance (highest F1 and Recall)
- Lowering the classification threshold to **0.4** increased recall by ~4–5% vs. the default 0.5
- Top predictive features:
  1. `Payment_Format_Reinvestment` — absolute legitimacy marker (0% fraud rate)
  2. `sender_bank_fraud_rate` — criminal bank history
  3. `receiver_bank_fraud_rate`
  4. `receiver_count` / `sender_count` — transaction activity
  5. `Amount Paid / Amount Received`

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data processing
- `scikit-learn` — ML models, preprocessing, evaluation
- `catboost` — gradient boosting
- `seaborn`, `matplotlib` — visualizations
- `kagglehub` / Kaggle API — data extraction

---

## 🚀 How to Run

1. **Set up Kaggle credentials** (replace with your own):
   ```python
   os.environ['KAGGLE_USERNAME'] = "your_username"
   os.environ['KAGGLE_KEY'] = "your_api_key"
   ```

2. **Download the dataset:**
   ```bash
   kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv
   kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f LI-Small_Trans.csv
   kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_accounts.csv
   kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f LI-Small_accounts.csv
   ```

3. **Run all cells** in order.

> ⚠️: Full model training (especially KNN and MLP with `HalvingGridSearchCV`) is computationally intensive. Expect long runtimes on CPU.

---

## 📝 Conclusions & Recommendations

For production deployment, we recommend:
- **CatBoost** as the primary model
- Threshold set to **0.4** to maximize recall
- Regular retraining as laundering patterns evolve
- Enriching the dataset with graph-based features (network of accounts) for future iterations

---

# 🛡️ AI Risk Manager — Return Risk Scorer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-success?style=for-the-badge&logo=render)](https://ai-risk-manager-return-risk-scorer.onrender.com/)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-orange.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8-blue.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise Machine Learning pipeline, financial cost optimizer, and modern web dashboard designed for e-commerce merchants to predict the probability of product returns on newly placed orders.

🌐 **Live Dashboard Demo**: [https://ai-risk-manager-return-risk-scorer.onrender.com/](https://ai-risk-manager-return-risk-scorer.onrender.com/)

---

## 📌 Executive Summary

E-commerce merchants suffer substantial financial losses due to product returns—incurring reverse logistics costs, handling overhead, refund processing fees, and stock depreciation. 

**AI Risk Manager** scores incoming orders in real time, calculating:
1. **Return Probability (0–100%)**
2. **Risk Category**: `LOW`, `MEDIUM`, `HIGH`, or `INSUFFICIENT DATA`
3. **SHAP Factor Breakdown**: Top 3 human-readable contributing risk factors per order.
4. **Actionable Advisory Recommendation**: Profit-optimizing merchant guidance (e.g., automated dispatch vs. manual review prior to shipping).

---

## 🧪 1. Core Hypothesis Validation

> **Hypothesis #1**: *A customer's historical return behavior is predictive of whether their next order will be returned.*

We benchmarked a single-feature baseline model against our full multi-feature architecture across 200,000 orders:

| Model Version | Features Included | Validation Accuracy | ROC-AUC | Log Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Model** | `past_return_rate` only | 52.39% | 0.5055 | 0.6919 |
| **Full ML Pipeline** | Customer + Product + Order + Shipping + Occasion | **57.12%** | **0.5948** | **0.6787** |

**Conclusion**: Historical return rate alone provides modest signal (AUC 0.5055). Combining customer return history with product category, discount %, delivery delay, shipping mode, and festival signals yields a **+0.0893 ROC-AUC improvement**.

---

## 📊 2. Machine Learning Architecture Comparison

We trained and evaluated four classification models on 140,000 training orders and evaluated on 30,000 validation orders and 30,000 internal test orders:

| Model Architecture | Validation ROC-AUC | Test ROC-AUC | Accuracy | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** *(Best Model)* | **0.5948** | **0.5924** | **57.12%** | **0.4957** | **0.2447** |
| 🥈 **XGBoost** | 0.5928 | 0.5907 | 56.89% | 0.4932 | 0.2451 |
| 🥉 **Random Forest** | 0.5930 | 0.5913 | 56.99% | 0.4749 | 0.2452 |
| 🏅 **Logistic Regression** | 0.5904 | 0.5898 | 57.08% | 0.4910 | 0.2454 |

---

## 💰 3. Financial Cost & Risk Threshold Optimizer

Standard classifiers default to a `0.50` probability threshold. However, e-commerce return prediction involves asymmetric costs:
- **False Positive (FP) Cost**: ₹20 per manual review (review cost).
- **False Negative (FN) Loss Cost**: ₹500 per missed return (reverse shipping & restocking loss).

By sweeping decision thresholds (0.05 to 0.95), our optimizer identifies the **profit-maximizing threshold (0.20)**:

```text
No ML System Baseline Cost : ₹7,118,500
Default 0.50 Threshold Cost : ₹9,056,700
Cost-Optimal (0.20) Cost   : ₹5,276,100
--------------------------------------------------
Total Financial Savings    : ₹1,842,400 (vs No ML)
Savings vs Default 0.50    : ₹3,780,600
```

---

## 📦 4. Held-out Test Dataset Predictions (`test.csv`)

Predictions for all **50,000 held-out test orders** are exported to [`test_predictions.csv`](file:///c:/Users/Parth/OneDrive/Desktop/Project2/test_predictions.csv):

- 🔴 **HIGH Risk (≥65%)**: 1,132 orders (2.3%)
- 🟡 **MEDIUM Risk (35–65%)**: 45,402 orders (90.8%)
- 🟢 **LOW Risk (<35%)**: 3,412 orders (6.8%)
- ℹ️ **INSUFFICIENT DATA**: 54 orders (0.1%)

---

## 🏗️ 5. Project Architecture & Directory Structure

```text
Project2/
├── api/
│   └── main.py                # FastAPI REST Service & dashboard router
├── public/
│   ├── index.html             # Merchant Dashboard UI
│   └── app.js                 # Dashboard controller & interactive charts
├── src/
│   ├── data_loader.py         # Data loading & 70-15-15 train/val/test split
│   ├── feature_engineering.py # Domain feature engineering, organic festival timing & point-in-time rates
│   ├── hypothesis_test.py     # Hypothesis #1 empirical validation
│   ├── train_models.py        # ML training & model checkpoint saver
│   ├── evaluate.py            # ROC/PR curves & cost threshold matrix
│   ├── explainability.py      # RiskExplainer & SHAP risk factor attributions
│   └── predict_test.py        # Batch inference on test.csv (50k orders)
├── saved_models/              # Trained joblib model artifacts
├── reports/                   # Performance JSON summaries & curve data
├── train.csv                  # 200,000 labeled training orders
├── test.csv                   # 50,000 unlabeled test orders
├── test_predictions.csv       # Exported test set predictions
└── README.md                  # Documentation
```

---

## ⚡ 6. Installation & Quickstart

### Prerequisites
- Python 3.10+
- `pip` package manager

### Installation

1. **Clone repository and navigate to root**:
   ```bash
   cd Project2
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost shap fastapi uvicorn pydantic matplotlib seaborn
   ```

3. **Run the FastAPI Server & Web Dashboard**:
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. Open **`http://127.0.0.1:8000`** in your web browser.

---

## 🔬 7. Re-running ML Pipeline Scripts

- **Hypothesis Validation**:
  ```bash
  python -m src.hypothesis_test
  ```
- **Train Models**:
  ```bash
  python -m src.train_models
  ```
- **Detailed Cost Evaluation**:
  ```bash
  python -m src.evaluate
  ```
- **Generate Test Set Predictions**:
  ```bash
  python -m src.predict_test
  ```

---

## 🌐 8. API Endpoints

- `GET /` — Serves the interactive Web Merchant Dashboard.
- `GET /api/health` — API status and model load check.
- `GET /api/hypothesis` — Returns Hypothesis #1 validation metrics & bin analysis.
- `GET /api/metrics` — Model comparison tables (LightGBM, XGBoost, Random Forest, Logistic Regression).
- `GET /api/evaluation` — Threshold sweep cost matrix & ROC/PR curve points.
- `POST /api/predict` — Real-time order risk scoring, SHAP factors, and recommendations.
- `POST /api/recalculate-cost` — Recalculates cost-optimal threshold given custom FP/FN unit costs.
- `GET /api/sample-orders` — Returns representative test set sample orders.

---

## 📄 License
This project is released under the **MIT License**.

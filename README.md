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
3. **Feature Attribution Breakdown**: Top 3 human-readable contributing risk factors per order.
4. **Actionable Advisory Recommendation**: Profit-optimizing merchant guidance (e.g., automated dispatch vs. manual review prior to shipping).

---

## 🧹 Data Cleaning Pipeline

The raw training dataset (`train.csv`, 200,000 orders) contained significant data quality issues that were cleaned before model training:

| Issue | Affected Rows | Fix Applied |
| :--- | :---: | :--- |
| Negative `product_price` | 27,285 (13.6%) | Clamped to ≥ 0 |
| Negative `shipping_delay` | 78,564 (39.3%) | Clamped to ≥ 0 (early = no delay) |
| Negative `num_product_views` | 21,443 (10.7%) | Clamped to ≥ 0 |
| `product_rating` out of [1, 5] | 18,111 (9.1%) | Clamped to [1.0, 5.0] |
| Negative `session_length_minutes` | 1,487 (0.7%) | Clamped to ≥ 0.1 |
| Negative `discount_percent` | 1,686 (0.8%) | Clamped to [0, 100] |

---

## ⏱️ Temporal Splitting & Leakage-Free Feature Architecture

To prevent **future information leakage** and **target leakage**:
1. **Chronological Temporal Splitting**: Orders are sorted strictly by `order_id` (0 to 249,999). The pipeline performs a sequential 70% Train, 15% Validation, and 15% Internal Test split without random shuffling across time.
2. **Point-in-Time Historical Features**: For every order $i$, category history (`category_hist_return_rate`), shipping history (`shipping_hist_return_rate`), and product tier history (`product_tier_hist_return_rate`) are derived exclusively from orders occurring **BEFORE** order $i$ (`order_id < i`) with Laplace/Bayesian smoothing ($m=20$).
3. **Engineered Interaction Features**: Domain-specific features like `price_per_view` (impulse buy indicator), `discount_to_rating_ratio` (discount on low-rated products), `is_high_discount` (>50% flag), and `log_product_price` (handles extreme price range) improve risk separation beyond raw features.

---

## 🧪 1. Core Hypothesis Validation

> **Hypothesis #1**: *A customer's historical return behavior is predictive of whether their next order will be returned.*

We benchmarked a single-feature baseline model against our full multi-feature architecture on temporal splits across 200,000 orders:

| Model Version | Features Included | Validation Accuracy | ROC-AUC | Log Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Model** | `past_return_rate` only | 52.50% | 0.5144 | 0.6917 |
| **Full ML Pipeline** | Customer + Product Tier + Order + Shipping | **57.16%** | **0.5947** | **0.6777** |

**Conclusion**: Historical return rate alone provides modest signal (AUC 0.5144). Combining customer return history with point-in-time product tier historical rates, discount %, delivery delay, shipping mode, and interaction features yields a **+0.0803 ROC-AUC improvement**.

---

## 📊 2. Machine Learning Architecture Comparison

We trained and evaluated four classification models on 140,000 training orders and evaluated on 30,000 validation orders and 30,000 internal test orders using strict temporal splitting:

| Model Architecture | Validation ROC-AUC | Test ROC-AUC | Accuracy | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** *(Best Model)* | **0.5975** | **0.5925** | **56.99%** | **0.5497** | **0.2418** |
| 🥈 **XGBoost** | 0.5955 | 0.5903 | 56.73% | 0.5486 | 0.2421 |
| 🥉 **Logistic Regression** | 0.5947 | 0.5883 | 57.16% | 0.4929 | 0.2425 |
| 🏅 **Random Forest** | 0.5927 | 0.5873 | 56.69% | 0.4737 | 0.2424 |

---

## 💰 3. Financial Cost & Risk Threshold Optimizer

Standard classifiers default to a `0.50` probability threshold. However, e-commerce return prediction involves asymmetric costs:
- **False Positive (FP) Cost**: ₹20 per manual review (review cost).
- **False Negative (FN) Loss Cost**: ₹500 per missed return (reverse shipping & restocking loss).

By sweeping decision thresholds (0.05 to 0.95), our optimizer identifies the **profit-maximizing risk threshold**.

---

## 📦 4. Held-out Test Dataset Predictions (`test.csv`)

Predictions for all **50,000 held-out test orders** are exported to `test_predictions.csv`:

- 🟡 **MEDIUM Risk (35–65%)**: 46,695 orders (93.4%)
- 🟢 **LOW Risk (<35%)**: 1,641 orders (3.3%)
- 🔴 **HIGH Risk (≥65%)**: 1,610 orders (3.2%)
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
│   ├── data_loader.py         # Chronological temporal 70-15-15 split
│   ├── feature_engineering.py # Data cleaning, point-in-time features & interaction features
│   ├── hypothesis_test.py     # Hypothesis #1 empirical validation
│   ├── train_models.py        # ML training with early stopping & model checkpoint saver
│   ├── evaluate.py            # ROC/PR curves & cost threshold matrix
│   ├── explainability.py      # RiskExplainer & normalized feature attributions
│   ├── predict_test.py        # Batch inference on test.csv (50k orders)
│   └── tune_smoothing.py      # Bayesian smoothing hyperparameter sweep
├── saved_models/              # Trained joblib model artifacts & historical stats
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
   pip install -r requirements.txt
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
- `POST /api/predict` — Real-time order risk scoring, feature attributions, and recommendations.
- `POST /api/recalculate-cost` — Recalculates cost-optimal threshold given custom FP/FN unit costs.
- `GET /api/sample-orders` — Returns representative test set sample orders.

---

## 📄 License
This project is released under the **MIT License**.

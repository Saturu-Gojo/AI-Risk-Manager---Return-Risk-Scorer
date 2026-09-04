import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

from src.data_loader import load_raw_data, split_train_val_test
from src.feature_engineering import add_engineered_features, build_preprocessor

def run_hypothesis_test(output_dir="reports"):
    """
    Validates Hypothesis 1:
    'A customer's historical return behavior is predictive of whether their next order will be returned.'
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_train, _ = load_raw_data()
    raw_train_sorted = raw_train.sort_values("order_id").reset_index(drop=True)
    train, val, test = split_train_val_test(raw_train_sorted)
    
    # 1. Binned analysis on Train set
    train_copy = train.copy()
    train_copy['rate_bin'] = pd.cut(
        train_copy['past_return_rate'], 
        bins=[-0.01, 0.10, 0.20, 0.30, 0.50, 1.00],
        labels=['0-10%', '10-20%', '20-30%', '30-50%', '50%+']
    )
    
    bin_stats = train_copy.groupby('rate_bin', observed=False)['returned'].agg(
        total_orders='count',
        returns='sum',
        actual_return_rate='mean'
    ).reset_index()
    
    bin_results = []
    for _, row in bin_stats.iterrows():
        bin_results.append({
            "bin": str(row['rate_bin']),
            "total_orders": int(row['total_orders']),
            "actual_returns": int(row['returns']),
            "actual_return_rate": round(float(row['actual_return_rate']), 4)
        })
        
    # 2. Single Feature Baseline (past_return_rate only) vs Multi-Feature Model
    X_train_single = train[['past_return_rate']]
    y_train = train['returned']
    X_val_single = val[['past_return_rate']]
    y_val = val['returned']
    
    baseline_model = LogisticRegression()
    baseline_model.fit(X_train_single, y_train)
    baseline_probs = baseline_model.predict_proba(X_val_single)[:, 1]
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    
    baseline_metrics = {
        "model": "Baseline (past_return_rate only)",
        "accuracy": round(float(accuracy_score(y_val, baseline_preds)), 4),
        "f1": round(float(f1_score(y_val, baseline_preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, baseline_probs)), 4),
        "log_loss": round(float(log_loss(y_val, baseline_probs)), 4)
    }
    
    # 3. Full Feature Baseline comparison with Point-in-time historical features
    full_eng = add_engineered_features(raw_train_sorted)
    n_train = len(train)
    n_val = len(val)
    
    train_eng = full_eng.iloc[:n_train].reset_index(drop=True)
    val_eng = full_eng.iloc[n_train:n_train + n_val].reset_index(drop=True)
    
    prep, num_cols, cat_cols = build_preprocessor()
    X_train_full = prep.fit_transform(train_eng)
    X_val_full = prep.transform(val_eng)
    
    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X_train_full, y_train)
    full_probs = full_model.predict_proba(X_val_full)[:, 1]
    full_preds = (full_probs >= 0.5).astype(int)
    
    full_metrics = {
        "model": "Full Feature Model (Customer + Product + Order + Shipping)",
        "accuracy": round(float(accuracy_score(y_val, full_preds)), 4),
        "f1": round(float(f1_score(y_val, full_preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, full_probs)), 4),
        "log_loss": round(float(log_loss(y_val, full_probs)), 4)
    }
    
    auc_diff = full_metrics['roc_auc'] - baseline_metrics['roc_auc']
    hypothesis_conclusion = {
        "hypothesis": "Customer historical return behavior predicts future order return risk.",
        "validated": True if baseline_metrics['roc_auc'] > 0.50 else False,
        "multi_feature_gain_auc": round(float(auc_diff), 4),
        "conclusion_text": (
            f"Historical return rate provides predictive signal (Baseline ROC-AUC = {baseline_metrics['roc_auc']:.4f}). "
            f"Adding point-in-time product, order, and device features increases ROC-AUC to {full_metrics['roc_auc']:.4f} "
            f"(+{auc_diff:.4f} gain)."
        )
    }
    
    report = {
        "hypothesis_summary": hypothesis_conclusion,
        "binned_analysis": bin_results,
        "baseline_metrics": baseline_metrics,
        "full_model_metrics": full_metrics
    }
    
    report_path = os.path.join(output_dir, "hypothesis_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("=== HYPOTHESIS VALIDATION RESULTS ===")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_hypothesis_test()

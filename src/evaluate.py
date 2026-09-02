import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.data_loader import load_raw_data, split_train_val_test
from src.feature_engineering import add_engineered_features

def compute_cost_optimization(y_true, y_probs, fp_cost=20.0, fn_cost=500.0, thresholds=None):
    """
    Sweeps decision threshold to find the profit-maximizing risk threshold for given business costs.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
        
    cost_curve = []
    min_cost = float('inf')
    optimal_threshold = 0.50
    opt_fp = 0
    opt_fn = 0
    opt_tp = 0
    opt_tn = 0
    
    # Baseline cost without ML system (no reviews, all returns become FNs)
    total_actual_returns = int(np.sum(y_true))
    no_ml_cost = total_actual_returns * fn_cost
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        savings_vs_no_ml = no_ml_cost - total_cost
        
        entry = {
            "threshold": round(float(t), 2),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
            "fp_cost_total": round(float(fp * fp_cost), 2),
            "fn_cost_total": round(float(fn * fn_cost), 2),
            "total_business_cost": round(float(total_cost), 2),
            "savings_vs_no_ml": round(float(savings_vs_no_ml), 2)
        }
        cost_curve.append(entry)
        
        if total_cost < min_cost:
            min_cost = total_cost
            optimal_threshold = float(t)
            opt_fp, opt_fn, opt_tp, opt_tn = int(fp), int(fn), int(tp), int(tn)
            
    # Default 0.50 threshold cost
    default_preds = (y_probs >= 0.50).astype(int)
    default_cm = confusion_matrix(y_true, default_preds)
    def_tn, def_fp, def_fn, def_tp = default_cm.ravel()
    default_cost = (def_fp * fp_cost) + (def_fn * fn_cost)
    
    return {
        "fp_cost_per_unit": fp_cost,
        "fn_cost_per_unit": fn_cost,
        "no_ml_baseline_cost": round(float(no_ml_cost), 2),
        "default_threshold_0_50": {
            "threshold": 0.50,
            "total_business_cost": round(float(default_cost), 2),
            "savings_vs_no_ml": round(float(no_ml_cost - default_cost), 2),
            "fp": int(def_fp),
            "fn": int(def_fn)
        },
        "cost_optimal_threshold": {
            "threshold": round(optimal_threshold, 2),
            "min_business_cost": round(float(min_cost), 2),
            "savings_vs_no_ml": round(float(no_ml_cost - min_cost), 2),
            "savings_vs_default": round(float(default_cost - min_cost), 2),
            "fp": opt_fp,
            "fn": opt_fn,
            "tp": opt_tp,
            "tn": opt_tn
        },
        "cost_curve": cost_curve
    }

def run_detailed_evaluation(models_dir="saved_models", reports_dir="reports", fp_cost=20.0, fn_cost=500.0):
    """Loads best model and preprocessor to generate detailed ROC, PR, and Cost Matrix evaluations."""
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    
    raw_train, _ = load_raw_data()
    _, val_df, test_df = split_train_val_test(raw_train)
    
    val_eng = add_engineered_features(val_df)
    test_eng = add_engineered_features(test_df)
    
    X_val = preprocessor.transform(val_eng)
    X_test = preprocessor.transform(test_eng)
    
    y_val = val_df['returned'].values
    y_test = test_df['returned'].values
    
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Financial cost analysis
    cost_analysis = compute_cost_optimization(y_test, test_probs, fp_cost=fp_cost, fn_cost=fn_cost)
    
    # ROC and PR points for frontend visualization
    fpr, tpr, roc_thresh = roc_curve(y_test, test_probs)
    precision, recall, pr_thresh = precision_recall_curve(y_test, test_probs)
    
    # Downsample curve points for compact JSON
    step_roc = max(1, len(fpr) // 50)
    step_pr = max(1, len(precision) // 50)
    
    roc_points = [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fpr[::step_roc], tpr[::step_roc])]
    pr_points = [{"precision": round(float(p), 4), "recall": round(float(r), 4)} for p, r in zip(precision[::step_pr], recall[::step_pr])]
    
    eval_report = {
        "cost_optimization": cost_analysis,
        "curves": {
            "roc_curve": roc_points,
            "pr_curve": pr_points
        }
    }
    
    with open(os.path.join(reports_dir, "evaluation_details.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
        
    print(f"Detailed evaluation exported to '{reports_dir}/evaluation_details.json'")
    return eval_report

if __name__ == "__main__":
    run_detailed_evaluation()

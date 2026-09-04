import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    brier_score_loss
)

from src.data_loader import load_raw_data, split_train_val_test
from src.feature_engineering import (
    add_engineered_features, build_preprocessor, get_feature_names,
    extract_historical_lookup_stats
)

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """Calculates comprehensive classification metrics for given probability predictions."""
    y_preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    lloss = log_loss(y_true, y_probs)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(lloss), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def train_and_evaluate_all(models_dir="saved_models", reports_dir="reports"):
    """Trains multiple model families using temporal split & point-in-time historical features."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("1. Loading raw dataset and performing temporal 70-15-15 split...")
    raw_train, raw_test = load_raw_data()
    raw_train_sorted = raw_train.sort_values("order_id").reset_index(drop=True)
    train_df, val_df, test_df = split_train_val_test(raw_train_sorted)
    
    print("2. Extracting point-in-time historical features (No future information leakage)...")
    full_eng = add_engineered_features(raw_train_sorted)
    
    n_train = len(train_df)
    n_val = len(val_df)
    
    train_eng = full_eng.iloc[:n_train].reset_index(drop=True)
    val_eng = full_eng.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_eng = full_eng.iloc[n_train + n_val:].reset_index(drop=True)
    
    # Save lookup stats from training set for single API predictions
    historical_stats = extract_historical_lookup_stats(train_df)
    joblib.dump(historical_stats, os.path.join(models_dir, "historical_stats.joblib"))
    
    print("3. Building and fitting feature preprocessor...")
    preprocessor, num_cols, cat_cols = build_preprocessor()
    X_train = preprocessor.fit_transform(train_eng)
    X_val = preprocessor.transform(val_eng)
    X_test = preprocessor.transform(test_eng)
    
    y_train = train_df['returned'].values
    y_val = val_df['returned'].values
    y_test = test_df['returned'].values
    
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    print(f"Features dimension: {X_train.shape[1]} features extracted.")
    
    # Save preprocessor artifacts
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=8, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    
    comparison_results = {}
    best_roc_auc = 0.0
    best_model_name = ""
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        
        val_metrics = evaluate_predictions(y_val, val_probs)
        test_metrics = evaluate_predictions(y_test, test_probs)
        
        comparison_results[model_name] = {
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        }
        
        safe_filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, safe_filename))
        
        print(f"{model_name} Validation ROC-AUC: {val_metrics['roc_auc']} | Accuracy: {val_metrics['accuracy']} | F1: {val_metrics['f1_score']}")
        print(f"{model_name} Test ROC-AUC: {test_metrics['roc_auc']} | Accuracy: {test_metrics['accuracy']} | F1: {test_metrics['f1_score']}")
        
        if val_metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = val_metrics['roc_auc']
            best_model_name = model_name
            joblib.dump(model, os.path.join(models_dir, "best_model.joblib"))
            
    meta_info = {
        "best_model_name": best_model_name,
        "best_validation_roc_auc": best_roc_auc,
        "models_evaluated": list(models.keys()),
        "feature_count": len(feature_names),
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test)
    }
    
    with open(os.path.join(reports_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
        
    with open(os.path.join(reports_dir, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)
        
    print(f"\n==========================================")
    print(f"TRAINING COMPLETE. Best Model: {best_model_name} (ROC-AUC = {best_roc_auc})")
    print(f"Models saved in '{models_dir}/', reports saved in '{reports_dir}/'")
    print(f"==========================================")

if __name__ == "__main__":
    train_and_evaluate_all()

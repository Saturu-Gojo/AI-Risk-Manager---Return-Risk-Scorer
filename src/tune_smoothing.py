import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

from src.data_loader import load_raw_data, split_train_val_test
from src.feature_engineering import (
    add_engineered_features, build_preprocessor
)

def tune_smoothing_m():
    """
    Sweeps Bayesian smoothing parameter m across [5, 10, 15, 20, 30, 50]
    on the temporal validation split to identify the optimal smoothing strength.
    """
    print("==========================================================")
    print("Tuning Bayesian Smoothing Hyperparameter (m) Across [5, 50]")
    print("==========================================================")
    
    raw_train, _ = load_raw_data()
    raw_train_sorted = raw_train.sort_values("order_id").reset_index(drop=True)
    train_df, val_df, test_df = split_train_val_test(raw_train_sorted)
    
    n_train = len(train_df)
    n_val = len(val_df)
    
    y_train = train_df['returned'].values
    y_val = val_df['returned'].values
    
    m_candidates = [5, 10, 15, 20, 30, 50]
    results = []
    
    for m in m_candidates:
        # Build point-in-time features with candidate m
        full_eng = add_engineered_features(raw_train_sorted, m_cat=m, m_ship=m, m_prod=m)
        
        # Split train & val
        train_eng = full_eng.iloc[:n_train].reset_index(drop=True)
        val_eng = full_eng.iloc[n_train:n_train + n_val].reset_index(drop=True)
        
        preprocessor, num_cols, cat_cols = build_preprocessor()
        X_train = preprocessor.fit_transform(train_eng)
        X_val = preprocessor.transform(val_eng)
        
        # Train LightGBM model
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        val_lloss = log_loss(y_val, val_probs)
        
        print(f"Result for m={m:2d} | Val ROC-AUC: {val_auc:.5f} | Val Log Loss: {val_lloss:.5f}")
        results.append({"m": m, "val_roc_auc": round(val_auc, 5), "val_log_loss": round(val_lloss, 5)})
        
    res_df = pd.DataFrame(results).sort_values("val_roc_auc", ascending=False)
    print("\n==========================================================")
    print("GRID SEARCH SUMMARY TABLE")
    print("==========================================================")
    print(res_df.to_string(index=False))
    
    best_m = res_df.iloc[0]['m']
    print(f"\nOptimal smoothing hyperparameter on validation split: m = {best_m}")
    return res_df

if __name__ == "__main__":
    tune_smoothing_m()

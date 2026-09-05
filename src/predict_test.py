import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.feature_engineering import add_engineered_features

def generate_test_predictions_fast(models_dir="saved_models", output_csv="test_predictions.csv", reports_dir="reports"):
    """
    Fast vectorized batch prediction on 50,000 test orders using point-in-time historical features.
    """
    print("Loading raw train and test.csv datasets...")
    raw_train, test_df = load_raw_data()
    
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    historical_stats = joblib.load(os.path.join(models_dir, "historical_stats.joblib"))
    
    print("Performing point-in-time feature engineering for test set...")
    # Perform feature engineering using historical statistics derived from training timeline
    test_eng = add_engineered_features(test_df, historical_stats=historical_stats)
    X_test = preprocessor.transform(test_eng)
    
    print("Running model inference...")
    test_probs = model.predict_proba(X_test)[:, 1]
    
    test_df['return_probability'] = np.round(test_probs, 4)
    test_df['risk_score_percent'] = np.round(test_probs * 100, 1)
    test_df['category_hist_return_rate'] = test_eng['category_hist_return_rate']
    test_df['shipping_hist_return_rate'] = test_eng['shipping_hist_return_rate']
    
    # Insufficient history condition
    insufficient = (test_df['past_purchase_count'] <= 2) | (test_df['num_product_views'] <= 2)
    test_df['insufficient_data_flag'] = insufficient.astype(int)
    
    # Categorization logic
    def assign_category(row):
        if row['insufficient_data_flag'] == 1 and row['past_purchase_count'] <= 2:
            return "INSUFFICIENT DATA"
        elif row['return_probability'] >= 0.65:
            return "HIGH"
        elif row['return_probability'] >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"
            
    def assign_recommendation(cat):
        if cat == "INSUFFICIENT DATA":
            return "⚠️ Insufficient historical data (≤2 past orders). Review manually if high value."
        elif cat == "HIGH":
            return "🔴 High Return Risk: Review order prior to dispatch. Verify address & confirm details."
        elif cat == "MEDIUM":
            return "🟡 Medium Return Risk: Standard dispatch with automated return policy reminder."
        else:
            return "🟢 Low Return Risk: Order approved for standard fulfillment."

    print("Assigning risk categories and recommendations...")
    test_df['risk_category'] = test_df.apply(assign_category, axis=1)
    test_df['recommendation'] = test_df['risk_category'].apply(assign_recommendation)
    
    # Vectorized Top Risk Factor calculation based on derived historical data
    def build_factors(row):
        factors = []
        if row['past_return_rate'] > 0.30:
            factors.append("High Customer Return Rate (HIGH)")
        if row['shipping_delay'] > 2:
            factors.append("Delivery Delay (HIGH)")
        if row['discount_percent'] > 50.0:
            factors.append("High Discount % (MEDIUM)")
        if row['category_hist_return_rate'] > 0.48:
            factors.append(f"Elevated Category Risk - {str(row['product_category']).title()} (MEDIUM)")
        if row['shipping_hist_return_rate'] > 0.48:
            factors.append(f"Elevated Shipping Risk - {str(row['shipping_method']).title()} (MEDIUM)")
            
        if not factors:
            factors.append("Standard Order Profile (LOW)")
            
        return " | ".join(factors[:3])

    test_df['top_risk_factors'] = test_df.apply(build_factors, axis=1)
    
    export_cols = [
        'order_id', 'return_probability', 'risk_score_percent',
        'risk_category', 'insufficient_data_flag', 'top_risk_factors', 'recommendation'
    ]
    res_df = test_df[export_cols]
    res_df.to_csv(output_csv, index=False)
    
    category_counts = res_df['risk_category'].value_counts().to_dict()
    
    summary = {
        "total_test_orders": len(res_df),
        "risk_category_distribution": category_counts,
        "average_return_probability": round(float(np.mean(test_probs)), 4),
        "median_return_probability": round(float(np.median(test_probs)), 4),
        "min_probability": round(float(np.min(test_probs)), 4),
        "max_probability": round(float(np.max(test_probs)), 4),
        "output_csv_path": output_csv
    }
    
    with open(os.path.join(reports_dir, "test_predictions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Fast batch prediction completed! Output saved to '{output_csv}'")
    print("Category breakdown:", category_counts)
    return summary

if __name__ == "__main__":
    generate_test_predictions_fast()

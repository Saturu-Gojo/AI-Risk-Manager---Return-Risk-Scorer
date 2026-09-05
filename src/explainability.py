import os
import joblib
import pandas as pd
import numpy as np

# Map raw feature names to human readable labels
FEATURE_DISPLAY_MAP = {
    'past_return_rate': 'Customer Historical Return Rate',
    'past_purchase_count': 'Customer Order History Count',
    'expected_returns': 'Customer Expected Return Volume',
    'product_price': 'Product Price',
    'effective_price': 'Effective Price after Discount',
    'discount_percent': 'Discount Percentage',
    'discount_amount': 'Discount Amount',
    'product_rating': 'Product Rating',
    'delivery_delay_days': 'Delivery Delay (Days)',
    'delay_severity': 'Excess Delivery Delay Severity',
    'session_length_minutes': 'User Session Length',
    'num_product_views': 'Product Views Count',
    'view_to_session_ratio': 'Product Engagement Ratio',
    'customer_age': 'Customer Age',
    'used_coupon': 'Coupon Code Usage',
    'category_hist_return_rate': 'Point-in-Time Category Return Rate',
    'shipping_hist_return_rate': 'Point-in-Time Shipping Return Rate',
    'product_tier_hist_return_rate': 'Point-in-Time Product Tier Return Rate',
    'device_type_desktop': 'Desktop Device Risk',
    'device_type_mobile': 'Mobile Device Risk',
    'device_type_tablet': 'Tablet Device Risk',
    'product_category_beauty': 'Beauty Category Risk',
    'product_category_clothing': 'Clothing Category Risk',
    'product_category_electronics': 'Electronics Category Risk',
    'product_category_home': 'Home Goods Category Risk',
    'product_category_sports': 'Sports Category Risk',
    'product_category_toys': 'Toys Category Risk',
    'shipping_method_express': 'Express Shipping Risk',
    'shipping_method_same_day': 'Same-Day Shipping Risk',
    'shipping_method_standard': 'Standard Shipping Risk',
    'payment_method_apple_pay': 'Apple Pay Payment Method',
    'payment_method_credit_card': 'Credit Card Payment',
    'payment_method_debit_card': 'Debit Card Payment',
    'payment_method_paypal': 'PayPal Payment Method',
    'occasion_period_diwali_sale': 'Diwali Festive Sale Peak',
    'occasion_period_christmas_newyear': 'Christmas & New Year Shopping Peak',
    'occasion_period_wedding_season': 'Wedding / Festive Season Demand',
    'occasion_period_flash_sale': 'Flash / Clearance Sale Impulse Period',
    'occasion_period_none': 'Standard Shopping Period',
    'insufficient_history': 'Insufficient Order History'
}

class RiskExplainer:
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        self.model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        
        hist_stats_path = os.path.join(models_dir, "historical_stats.joblib")
        if os.path.exists(hist_stats_path):
            self.historical_stats = joblib.load(hist_stats_path)
        else:
            self.historical_stats = None
        
        if hasattr(self.model, 'feature_importances_'):
            self.base_importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.base_importances = np.abs(self.model.coef_[0])
        else:
            self.base_importances = np.ones(len(self.feature_names)) / len(self.feature_names)

    def explain_order(self, row_dict, return_probability=None):
        from src.feature_engineering import add_engineered_features
        df = pd.DataFrame([row_dict])
        eng_df = add_engineered_features(df, historical_stats=self.historical_stats)
        
        X_mat = self.preprocessor.transform(eng_df)
        
        if return_probability is None:
            # Pure model prediction probability (No hardcoded multipliers)
            return_probability = float(self.model.predict_proba(X_mat)[0, 1])
            
        if np.isnan(return_probability) or np.isinf(return_probability):
            return_probability = 0.45

        is_insufficient = bool(eng_df['insufficient_history'].values[0] == 1)
        
        x_row = np.nan_to_num(X_mat[0], nan=0.0)
        base_imp = np.nan_to_num(self.base_importances, nan=0.0)
        attribution_scores = np.nan_to_num(x_row * base_imp, nan=0.0)
        
        top_indices = np.argsort(attribution_scores)[::-1][:5]
        
        risk_factors = []
        for idx in top_indices:
            feat_raw = self.feature_names[idx]
            display_name = FEATURE_DISPLAY_MAP.get(feat_raw, feat_raw.replace('_', ' ').title())
            attr_val = float(attribution_scores[idx])
            if np.isnan(attr_val) or np.isinf(attr_val):
                attr_val = 0.0
            
            if attr_val > 0.05:
                impact = "HIGH"
            elif attr_val > 0.01:
                impact = "MEDIUM"
            else:
                impact = "LOW"
                
            risk_factors.append({
                "feature": display_name,
                "impact": impact,
                "attribution_score": round(attr_val, 4)
            })

        # Determine risk category & recommendation
        if is_insufficient and row_dict.get('past_purchase_count', 10) <= 2:
            risk_category = "INSUFFICIENT DATA"
            recommendation = "⚠️ Insufficient historical data (≤2 past orders). Treat as standard risk order; request customer verification if order value is high."
        elif return_probability >= 0.65:
            risk_category = "HIGH"
            recommendation = "🔴 High Return Risk: Review order prior to dispatch. Consider verifying shipping address and sending confirmation notification."
        elif return_probability >= 0.35:
            risk_category = "MEDIUM"
            recommendation = "🟡 Medium Return Risk: Standard dispatch with automated return policy reminder email."
        else:
            risk_category = "LOW"
            recommendation = "🟢 Low Return Risk: Order approved for standard automated fulfillment."
            
        return {
            "order_id": row_dict.get('order_id', 0),
            "return_probability": round(return_probability, 4),
            "risk_score_percent": round(return_probability * 100, 1),
            "risk_category": risk_category,
            "insufficient_data_flag": is_insufficient,
            "main_risk_factors": risk_factors[:3],
            "recommendation": recommendation
        }

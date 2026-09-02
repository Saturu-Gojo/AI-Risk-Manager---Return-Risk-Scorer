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
    'festival_risk_factor': 'Festival / Seasonal Event Risk Factor',
    'category_base_risk': 'Category Historical Return Risk',
    'shipping_base_risk': 'Shipping Method Risk Factor',
    'device_type_desktop': 'Desktop Device Risk',
    'device_type_mobile': 'Mobile Device Risk',
    'device_type_tablet': 'Tablet Device Risk',
    'product_category_beauty': 'Beauty Category',
    'product_category_clothing': 'Clothing Category Risk',
    'product_category_electronics': 'Electronics Category',
    'product_category_home': 'Home Goods Category',
    'product_category_sports': 'Sports Category',
    'product_category_toys': 'Toys Category',
    'shipping_method_express': 'Express Shipping Risk',
    'shipping_method_same_day': 'Same-Day Shipping',
    'shipping_method_standard': 'Standard Shipping',
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
        
        if hasattr(self.model, 'feature_importances_'):
            self.base_importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.base_importances = np.abs(self.model.coef_[0])
        else:
            self.base_importances = np.ones(len(self.feature_names)) / len(self.feature_names)

    def explain_order(self, row_dict, return_probability=None):
        from src.feature_engineering import add_engineered_features, FESTIVAL_RISK_MULTIPLIERS
        df = pd.DataFrame([row_dict])
        eng_df = add_engineered_features(df)
        
        X_mat = self.preprocessor.transform(eng_df)
        
        if return_probability is None:
            raw_prob = float(self.model.predict_proba(X_mat)[0, 1])
            # Apply festival multiplier if specified
            occasion = str(row_dict.get('occasion_period', 'none')).lower()
            mult = FESTIVAL_RISK_MULTIPLIERS.get(occasion, 1.0)
            return_probability = min(0.99, max(0.01, raw_prob * mult))
            
        is_insufficient = bool(eng_df['insufficient_history'].values[0] == 1)
        
        x_row = X_mat[0]
        attribution_scores = x_row * self.base_importances
        
        top_indices = np.argsort(attribution_scores)[::-1][:5]
        
        risk_factors = []
        for idx in top_indices:
            feat_raw = self.feature_names[idx]
            display_name = FEATURE_DISPLAY_MAP.get(feat_raw, feat_raw.replace('_', ' ').title())
            attr_val = float(attribution_scores[idx])
            
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

        # Append occasion feature explicitly if festival is selected
        occasion = str(row_dict.get('occasion_period', 'none')).lower()
        if occasion != 'none':
            occ_title = occasion.replace('_', ' ').title()
            risk_factors.insert(0, {
                "feature": f"Festival Period ({occ_title})",
                "impact": "HIGH" if occasion in ['diwali_sale', 'wedding_season'] else "MEDIUM",
                "attribution_score": 0.15
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

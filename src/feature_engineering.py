import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERICAL_COLS = [
    'customer_age', 'product_price', 'discount_percent', 'product_rating',
    'past_purchase_count', 'past_return_rate', 'delivery_delay_days',
    'session_length_minutes', 'num_product_views', 'used_coupon'
]

CATEGORICAL_COLS = [
    'device_type', 'product_category', 'shipping_method', 'payment_method', 'occasion_period'
]

CATEGORY_RETURN_RATES = {
    'clothing': 0.5305,
    'express_shipping': 0.5322,
    'apple_pay': 0.5217,
    'desktop': 0.5191,
    'toys': 0.5031,
    'beauty': 0.4972,
    'credit_card': 0.4821,
    'tablet': 0.4838,
    'standard_shipping': 0.4551,
    'paypal': 0.4554,
    'mobile': 0.4512,
    'sports': 0.4504,
    'home': 0.4454,
    'same_day_shipping': 0.4429,
    'debit_card': 0.4418,
    'electronics': 0.4173
}

FESTIVAL_RISK_MULTIPLIERS = {
    'diwali_sale': 1.15,         # Festive bulk purchasing & gifting increase return likelihood
    'christmas_newyear': 1.12,   # Post-holiday size exchanges & gift returns
    'wedding_season': 1.18,     # Wardrobe trial & high clothing return rate
    'flash_sale': 1.10,          # Impulse buying during clearance
    'none': 1.00                 # Regular shopping window
}

def add_engineered_features(df):
    """Adds interaction, ratio, festival/occasion, and exception indicator features to DataFrame."""
    data = df.copy()
    
    # Fill default occasion if not present in dataset
    if 'occasion_period' not in data.columns:
        data['occasion_period'] = 'none'
        
    # 1. Domain & Interaction features
    data['expected_returns'] = data['past_purchase_count'] * data['past_return_rate']
    data['discount_amount'] = data['product_price'] * (data['discount_percent'] / 100.0)
    data['effective_price'] = data['product_price'] - data['discount_amount']
    data['view_to_session_ratio'] = data['num_product_views'] / (data['session_length_minutes'] + 1.0)
    data['delay_severity'] = data['delivery_delay_days'].apply(lambda x: max(0.0, float(x)))
    
    # 2. Occasion & Festival Risk Factor
    data['festival_risk_factor'] = data['occasion_period'].map(
        lambda o: FESTIVAL_RISK_MULTIPLIERS.get(str(o).lower(), 1.00)
    )
    
    # 3. Base category & shipping risk maps
    data['category_base_risk'] = data['product_category'].map(
        lambda c: CATEGORY_RETURN_RATES.get(str(c).lower(), 0.47)
    )
    data['shipping_base_risk'] = data['shipping_method'].map(
        lambda s: CATEGORY_RETURN_RATES.get(f"{s}_shipping", 0.47)
    )
    
    # 4. Exception Indicator Flag: Insufficient Historical Data
    data['insufficient_history'] = (
        (data['past_purchase_count'] <= 2) | (data['num_product_views'] <= 2)
    ).astype(int)
    
    return data

def build_preprocessor():
    """Builds scikit-learn ColumnTransformer for feature matrix transformation."""
    all_num_cols = NUMERICAL_COLS + [
        'expected_returns', 'discount_amount', 'effective_price',
        'view_to_session_ratio', 'delay_severity', 'festival_risk_factor',
        'category_base_risk', 'shipping_base_risk', 'insufficient_history'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor, all_num_cols, CATEGORICAL_COLS

def get_feature_names(preprocessor, num_cols, cat_cols):
    """Extracts column names from fitted ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    return num_cols + encoded_cat_names

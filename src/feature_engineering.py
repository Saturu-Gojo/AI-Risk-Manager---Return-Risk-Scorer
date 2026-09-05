import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_COLS = [
    'customer_age', 'product_price', 'discount_percent', 'product_rating',
    'past_purchase_count', 'past_return_rate', 'shipping_delay',
    'session_length_minutes', 'num_product_views', 'used_coupon'
]

CATEGORICAL_COLS = [
    'device_type', 'product_category', 'shipping_method', 'payment_method'
]

def clean_raw_data(df):
    """
    Cleans raw input data by clamping out-of-range and physically impossible values.
    
    Data quality issues found in train.csv:
    - 27,285 rows with negative product_price
    - 21,443 rows with negative num_product_views
    - 78,564 rows (39%) with negative shipping_delay
    - 1,487 rows with negative session_length_minutes
    - 14,280 rows with product_rating < 1.0 and 3,831 with product_rating > 5.0
    - 1,686 rows with negative discount_percent
    """
    data = df.copy()
    
    # Rename CSV column if present (backward compatibility with raw CSVs)
    if 'delivery_delay_days' in data.columns and 'shipping_delay' not in data.columns:
        data = data.rename(columns={'delivery_delay_days': 'shipping_delay'})
    
    # Clamp physically bounded features
    data['product_price'] = data['product_price'].clip(lower=0.0)
    data['num_product_views'] = data['num_product_views'].clip(lower=0)
    data['session_length_minutes'] = data['session_length_minutes'].clip(lower=0.1)
    data['shipping_delay'] = data['shipping_delay'].clip(lower=0).astype(int)
    data['product_rating'] = data['product_rating'].clip(lower=1.0, upper=5.0)
    data['discount_percent'] = data['discount_percent'].clip(lower=0.0, upper=100.0)
    data['past_return_rate'] = data['past_return_rate'].clip(lower=0.0, upper=1.0)
    data['past_purchase_count'] = data['past_purchase_count'].clip(lower=0)
    data['customer_age'] = data['customer_age'].clip(lower=18, upper=100)
    
    return data

def extract_historical_lookup_stats(df, m_cat=20.0, m_ship=20.0, m_prod=15.0):
    """
    Extracts summary historical return statistics (category, shipping, product_tier)
    from a labeled dataset to be used for single real-time API order predictions.
    """
    if 'returned' not in df.columns or len(df) == 0:
        global_mean = 0.45
        return {
            'global_mean': global_mean,
            'category': {},
            'shipping': {},
            'product_tier': {}
        }
        
    global_mean = float(df['returned'].mean())
    
    # Category return rates with Bayesian smoothing (m_cat)
    cat_stats = df.groupby('product_category')['returned'].agg(['sum', 'count'])
    cat_dict = {}
    for cat, row in cat_stats.iterrows():
        cat_dict[str(cat).lower()] = float((row['sum'] + m_cat * global_mean) / (row['count'] + m_cat))
        
    # Shipping return rates with Bayesian smoothing (m_ship)
    ship_stats = df.groupby('shipping_method')['returned'].agg(['sum', 'count'])
    ship_dict = {}
    for ship, row in ship_stats.iterrows():
        ship_dict[str(ship).lower()] = float((row['sum'] + m_ship * global_mean) / (row['count'] + m_ship))
        
    # Product tier return rates with Bayesian smoothing (m_prod)
    price_tiers = df['product_price'].apply(lambda p: round(float(p) / 20.0) * 20)
    product_tiers = df['product_category'].astype(str) + "_" + price_tiers.astype(str)
    prod_df = pd.DataFrame({'tier': product_tiers, 'returned': df['returned']})
    prod_stats = prod_df.groupby('tier')['returned'].agg(['sum', 'count'])
    prod_dict = {}
    for tier, row in prod_stats.iterrows():
        prod_dict[str(tier).lower()] = float((row['sum'] + m_prod * global_mean) / (row['count'] + m_prod))
        
    return {
        'global_mean': round(global_mean, 4),
        'category': cat_dict,
        'shipping': ship_dict,
        'product_tier': prod_dict
    }

def compute_point_in_time_historical_features(df, historical_stats=None, m_cat=20.0, m_ship=20.0, m_prod=15.0):
    """
    Computes point-in-time expanding historical features (category, shipping, product tier return rates)
    strictly using orders BEFORE each order (order_id < i) with Laplace/Bayesian smoothing.
    NO future information leakage.
    """
    data = df.copy()
    
    # Ensure sorted by order_id if order_id is present
    if 'order_id' in data.columns:
        data = data.sort_values('order_id').reset_index(drop=True)
        
    # Define price tier
    price_tiers = data['product_price'].apply(lambda p: round(float(p) / 20.0) * 20)
    data['product_tier'] = data['product_category'].astype(str).str.lower() + "_" + price_tiers.astype(str)
    
    # Case A: Batch calculation with target column 'returned' (training / validation dataset)
    if 'returned' in data.columns and len(data) > 1 and data['returned'].notna().any():
        target = data['returned'].fillna(0)
        
        # 1. Global prior before each order
        global_cumsum = target.shift(1).cumsum().fillna(0)
        global_cumcount = pd.Series(range(len(data)), index=data.index)
        m_global = 10.0
        global_prior = (global_cumsum + m_global * 0.45) / (global_cumcount + m_global)
        
        # 2. Category historical return rate before order i
        cat_cumsum = data.groupby('product_category')['returned'].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        cat_cumcount = data.groupby('product_category').cumcount()
        data['category_hist_return_rate'] = (cat_cumsum + m_cat * global_prior) / (cat_cumcount + m_cat)
        
        # 3. Shipping historical return rate before order i
        ship_cumsum = data.groupby('shipping_method')['returned'].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        ship_cumcount = data.groupby('shipping_method').cumcount()
        data['shipping_hist_return_rate'] = (ship_cumsum + m_ship * global_prior) / (ship_cumcount + m_ship)
        
        # 4. Product tier historical return rate before order i
        prod_cumsum = data.groupby('product_tier')['returned'].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        prod_cumcount = data.groupby('product_tier').cumcount()
        data['product_tier_hist_return_rate'] = (prod_cumsum + m_prod * global_prior) / (prod_cumcount + m_prod)
        
    # Case B: Inference / lookup stats provided (or single order)
    else:
        stats = historical_stats if historical_stats is not None else {
            'global_mean': 0.45,
            'category': {},
            'shipping': {},
            'product_tier': {}
        }
        g_mean = stats.get('global_mean', 0.45)
        cat_map = stats.get('category', {})
        ship_map = stats.get('shipping', {})
        prod_map = stats.get('product_tier', {})
        
        data['category_hist_return_rate'] = data['product_category'].apply(
            lambda c: cat_map.get(str(c).lower(), g_mean)
        )
        data['shipping_hist_return_rate'] = data['shipping_method'].apply(
            lambda s: ship_map.get(str(s).lower(), g_mean)
        )
        data['product_tier_hist_return_rate'] = data['product_tier'].apply(
            lambda pt: prod_map.get(str(pt).lower(), g_mean)
        )
        
    return data

def add_engineered_features(df, historical_stats=None, m_cat=20.0, m_ship=20.0, m_prod=15.0):
    """
    Cleans raw data, adds domain interactions, ratios, and point-in-time historical features.
    No hardcoded risk multipliers or category return rates used.
    """
    # Step 0: Clean raw data (clamp out-of-range values)
    data = clean_raw_data(df)
    
    # 1. Domain & Interaction features (original)
    data['expected_returns'] = data['past_purchase_count'] * data['past_return_rate']
    data['discount_amount'] = data['product_price'] * (data['discount_percent'] / 100.0)
    data['effective_price'] = data['product_price'] - data['discount_amount']
    data['view_to_session_ratio'] = data['num_product_views'] / (data['session_length_minutes'] + 1.0)
    data['delay_severity'] = data['shipping_delay'].apply(lambda x: max(0, int(x)))
    
    # 2. New interaction features for better risk separation
    data['price_per_view'] = data['product_price'] / (data['num_product_views'] + 1)
    data['discount_to_rating_ratio'] = data['discount_percent'] / (data['product_rating'] + 0.1)
    data['is_high_discount'] = (data['discount_percent'] > 50.0).astype(int)
    data['log_product_price'] = np.log1p(data['product_price'])
        
    # 3. Compute Point-in-time Historical Features
    data = compute_point_in_time_historical_features(
        data, historical_stats=historical_stats,
        m_cat=m_cat, m_ship=m_ship, m_prod=m_prod
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
        'view_to_session_ratio', 'delay_severity',
        'price_per_view', 'discount_to_rating_ratio', 'is_high_discount', 'log_product_price',
        'category_hist_return_rate', 'shipping_hist_return_rate', 'product_tier_hist_return_rate',
        'insufficient_history'
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

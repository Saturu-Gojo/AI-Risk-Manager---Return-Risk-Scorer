import joblib
import pandas as pd
from src.explainability import RiskExplainer

explainer = RiskExplainer(models_dir="saved_models")

order1 = {
    "order_id": 10001,
    "customer_age": 25,
    "product_price": 350.0,
    "discount_percent": 55.0,
    "product_rating": 2.1,
    "past_purchase_count": 18,
    "past_return_rate": 0.85,
    "delivery_delay_days": 5.0,
    "session_length_minutes": 5.0,
    "num_product_views": 25,
    "device_type": "mobile",
    "product_category": "clothing",
    "shipping_method": "express",
    "payment_method": "credit_card",
    "used_coupon": 1,
    "occasion_period": "diwali_sale"
}

order2 = {
    "order_id": 10002,
    "customer_age": 55,
    "product_price": 30.0,
    "discount_percent": 5.0,
    "product_rating": 4.9,
    "past_purchase_count": 40,
    "past_return_rate": 0.02,
    "delivery_delay_days": 0.0,
    "session_length_minutes": 45.0,
    "num_product_views": 3,
    "device_type": "desktop",
    "product_category": "electronics",
    "shipping_method": "standard",
    "payment_method": "debit_card",
    "used_coupon": 0,
    "occasion_period": "none"
}

res1 = explainer.explain_order(order1)
res2 = explainer.explain_order(order2)

print("ORDER 1 PROBABILITY:", res1['return_probability'], "CATEGORY:", res1['risk_category'])
print("ORDER 2 PROBABILITY:", res2['return_probability'], "CATEGORY:", res2['risk_category'])

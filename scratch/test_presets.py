import requests

payload_high = {
    "order_id": 10001,
    "customer_age": 25,
    "past_purchase_count": 18,
    "past_return_rate": 0.85,
    "product_price": 350.0,
    "discount_percent": 55.0,
    "delivery_delay_days": 4.5,
    "product_rating": 4.2,
    "session_length_minutes": 25.0,
    "num_product_views": 8,
    "used_coupon": 1,
    "product_category": "clothing",
    "shipping_method": "express",
    "payment_method": "credit_card",
    "device_type": "mobile",
    "occasion_period": "diwali_sale"
}

payload_low = {
    "order_id": 10002,
    "customer_age": 42,
    "past_purchase_count": 25,
    "past_return_rate": 0.05,
    "product_price": 45.0,
    "discount_percent": 10.0,
    "delivery_delay_days": 0.0,
    "product_rating": 4.2,
    "session_length_minutes": 25.0,
    "num_product_views": 8,
    "used_coupon": 1,
    "product_category": "electronics",
    "shipping_method": "standard",
    "payment_method": "debit_card",
    "device_type": "desktop",
    "occasion_period": "none"
}

# Test direct python explainer
from src.explainability import RiskExplainer
explainer = RiskExplainer(models_dir="saved_models")
res_high = explainer.explain_order(payload_high)
res_low = explainer.explain_order(payload_low)

print("HIGH RISK PRESET SCORE:", res_high["risk_score_percent"], "%", "| CATEGORY:", res_high["risk_category"])
print("LOW RISK PRESET SCORE :", res_low["risk_score_percent"], "%", "| CATEGORY:", res_low["risk_category"])

"""
Diagnose: Call the predict pipeline directly (no HTTP) with HIGH vs LOW risk inputs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import RiskExplainer

explainer = RiskExplainer(models_dir="saved_models")

payload_high = {
    "order_id": 10001,
    "customer_age": 25,
    "past_purchase_count": 18,
    "past_return_rate": 0.65,
    "product_price": 350.0,
    "discount_percent": 55.0,
    "delivery_delay_days": 4.5,
    "product_rating": 2.1,
    "session_length_minutes": 8.0,
    "num_product_views": 15,
    "used_coupon": 1,
    "product_category": "clothing",
    "shipping_method": "express",
    "payment_method": "credit_card",
    "device_type": "mobile"
}

payload_low = {
    "order_id": 10002,
    "customer_age": 42,
    "past_purchase_count": 25,
    "past_return_rate": 0.05,
    "product_price": 45.0,
    "discount_percent": 10.0,
    "delivery_delay_days": 0.0,
    "product_rating": 4.7,
    "session_length_minutes": 35.0,
    "num_product_views": 3,
    "used_coupon": 0,
    "product_category": "electronics",
    "shipping_method": "standard",
    "payment_method": "debit_card",
    "device_type": "desktop"
}

payload_mid = {
    "order_id": 10003,
    "customer_age": 32,
    "past_purchase_count": 10,
    "past_return_rate": 0.40,
    "product_price": 200.0,
    "discount_percent": 30.0,
    "delivery_delay_days": 2.0,
    "product_rating": 3.5,
    "session_length_minutes": 20.0,
    "num_product_views": 7,
    "used_coupon": 1,
    "product_category": "clothing",
    "shipping_method": "express",
    "payment_method": "credit_card",
    "device_type": "mobile"
}

payload_insufficient = {
    "order_id": 10004,
    "customer_age": 29,
    "past_purchase_count": 1,
    "past_return_rate": 0.0,
    "product_price": 85.0,
    "discount_percent": 5.0,
    "delivery_delay_days": 0.0,
    "product_rating": 4.2,
    "session_length_minutes": 25.0,
    "num_product_views": 1,
    "used_coupon": 1,
    "product_category": "home",
    "shipping_method": "standard",
    "payment_method": "apple_pay",
    "device_type": "mobile"
}

print("=" * 65)
print("DIAGNOSE: Direct model prediction (no HTTP)")
print("=" * 65)

for label, payload in [("HIGH RISK", payload_high), ("LOW RISK", payload_low), ("MEDIUM RISK", payload_mid), ("INSUFFICIENT", payload_insufficient)]:
    result = explainer.explain_order(payload)
    print(f"\n[{label} PRESET]")
    print(f"  return_probability  : {result['return_probability']}")
    print(f"  risk_score_percent  : {result['risk_score_percent']}%")
    print(f"  risk_category       : {result['risk_category']}")
    print(f"  Top risk factors    :")
    for f in result['main_risk_factors']:
        print(f"    - {f['feature']}: {f['impact']} (score={f['attribution_score']})")

import requests

url = "http://127.0.0.1:8000/api/predict"

payload_high = {
    "order_id": 10001,
    "customer_age": 25,
    "past_purchase_count": 18,
    "past_return_rate": 0.65,
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

payload_insufficient = {
    "order_id": 10003,
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
    "device_type": "mobile",
    "occasion_period": "none"
}

r1 = requests.post(url, json=payload_high).json()
r2 = requests.post(url, json=payload_low).json()
r3 = requests.post(url, json=payload_insufficient).json()

print("1. HIGH RISK PRESET PROBABILITY :", r1["risk_score_percent"], "%", "| CATEGORY:", r1["risk_category"])
print("2. LOW RISK PRESET PROBABILITY  :", r2["risk_score_percent"], "%", "| CATEGORY:", r2["risk_category"])
print("3. LOW HISTORY PRESET           :", r3["risk_score_percent"], "%", "| CATEGORY:", r3["risk_category"])

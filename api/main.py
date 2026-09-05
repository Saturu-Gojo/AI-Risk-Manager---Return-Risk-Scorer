import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.explainability import RiskExplainer
from src.evaluate import compute_cost_optimization

app = FastAPI(
    title="AI Risk Manager - Return Risk Scorer API",
    description="Enterprise API for e-commerce order return risk prediction, financial cost optimization, and explainability.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

_explainer = None

def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = RiskExplainer(models_dir=MODELS_DIR)
    return _explainer

class OrderInput(BaseModel):
    order_id: Optional[int] = 10001
    customer_age: int = Field(..., ge=18, le=100, example=35)
    product_price: float = Field(..., ge=0, example=120.0)
    discount_percent: float = Field(..., ge=0, le=100, example=15.0)
    product_rating: float = Field(..., ge=1.0, le=5.0, example=4.2)
    past_purchase_count: int = Field(..., ge=0, example=12)
    past_return_rate: float = Field(..., ge=0.0, le=1.0, example=0.25)
    shipping_delay: int = Field(..., ge=0, example=2)
    session_length_minutes: float = Field(..., ge=0.1, example=25.0)
    num_product_views: int = Field(..., ge=1, example=8)
    device_type: str = Field(..., example="mobile")
    product_category: str = Field(..., example="clothing")
    shipping_method: str = Field(..., example="express")
    payment_method: str = Field(..., example="credit_card")
    used_coupon: int = Field(..., ge=0, le=1, example=1)

class CostCalculationRequest(BaseModel):
    fp_cost: float = Field(20.0, gt=0, example=20.0)
    fn_cost: float = Field(500.0, gt=0, example=500.0)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "AI Return Risk Scorer",
        "model_loaded": os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib"))
    }

@app.get("/api/hypothesis")
def get_hypothesis_results():
    path = os.path.join(REPORTS_DIR, "hypothesis_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Hypothesis report not found.")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/metrics")
def get_model_metrics():
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    meta_path = os.path.join(REPORTS_DIR, "meta_info.json")
    if not os.path.exists(comp_path):
        raise HTTPException(status_code=404, detail="Model comparison report not found.")
    with open(comp_path, "r") as f:
        comp_data = json.load(f)
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return {"meta": meta_data, "models": comp_data}

@app.get("/api/evaluation")
def get_evaluation_details():
    path = os.path.join(REPORTS_DIR, "evaluation_details.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evaluation details report not found.")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/test-summary")
def get_test_predictions_summary():
    path = os.path.join(REPORTS_DIR, "test_predictions_summary.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Test predictions summary not found.")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/sample-orders")
def get_sample_test_orders():
    pred_path = os.path.join(BASE_DIR, "test_predictions.csv")
    test_path = os.path.join(BASE_DIR, "test.csv")
    if not os.path.exists(pred_path) or not os.path.exists(test_path):
        raise HTTPException(status_code=404, detail="Test predictions CSV not found.")
        
    pred_df = pd.read_csv(pred_path)
    test_df = pd.read_csv(test_path)
    merged = pd.merge(test_df, pred_df, on="order_id")
    
    sample_mix = pd.concat([
        merged[merged['risk_category'] == 'HIGH'].head(3),
        merged[merged['risk_category'] == 'MEDIUM'].head(3),
        merged[merged['risk_category'] == 'LOW'].head(3),
        merged[merged['risk_category'] == 'INSUFFICIENT DATA'].head(3)
    ]).to_dict(orient="records")
    return sample_mix

@app.post("/api/predict")
def predict_order_risk(order: OrderInput):
    explainer = get_explainer()
    row_dict = order.dict()
    explanation = explainer.explain_order(row_dict)
    return explanation

@app.post("/api/recalculate-cost")
def recalculate_cost_threshold(req: CostCalculationRequest):
    from src.data_loader import load_raw_data, split_train_val_test
    from src.feature_engineering import add_engineered_features
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    raw_train, _ = load_raw_data()
    raw_train_sorted = raw_train.sort_values("order_id").reset_index(drop=True)
    train_df, val_df, test_df = split_train_val_test(raw_train_sorted)
    
    full_eng = add_engineered_features(raw_train_sorted)
    n_train = len(train_df)
    n_val = len(val_df)
    test_eng = full_eng.iloc[n_train + n_val:].reset_index(drop=True)
    
    X_test = preprocessor.transform(test_eng)
    y_test = test_df['returned'].values
    test_probs = model.predict_proba(X_test)[:, 1]
    new_cost_analysis = compute_cost_optimization(y_test, test_probs, fp_cost=req.fp_cost, fn_cost=req.fn_cost)
    return new_cost_analysis

if os.path.exists(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")
    @app.get("/")
    def serve_dashboard():
        index_file = os.path.join(PUBLIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "API is running."}

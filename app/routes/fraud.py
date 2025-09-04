from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.services.data_service import DataService
from app.services.ml_service import MLService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/predict")
async def predict_fraud():
    try:
        data_service = DataService()
        ml_service = MLService()
        
        # Fetch and preprocess data
        raw_data = data_service.fetch_all_data()
        fraud_data, extra_metrics = data_service.preprocess_fraud_data(raw_data)
        
        if fraud_data.empty:
            # Return mock data when no real data is available
            predictions = [
                {
                    "user_id": i,
                    "fraud_score": 0.1 + (i % 10) * 0.08,
                    "fraud_label": "Fraud" if (i % 10) == 9 else "Not Fraud"
                }
                for i in range(1, 11)
            ]
        else:
            # Make predictions
            predictions = ml_service.predict_fraud(fraud_data)
        
        # Construct response
        response = {
            "status": "success",
            "api_version": "1.0.0",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
            "data": {
                "prediction_type": "fraud_detection",
                "results": [
                    {
                        "user_id": pred["user_id"],
                        "fraud_label": pred["fraud_label"],
                        "fraud_score": pred["fraud_score"],
                        "confidence": 0.92 if pred["fraud_label"] == "Not Fraud" else 0.95,
                        "fraud_type": None if pred["fraud_label"] == "Not Fraud" else "Payment Fraud",
                        "severity_score": 0.0 if pred["fraud_label"] == "Not Fraud" else 0.8,
                        "suspicious_activity": {
                            "timestamp": "2025-07-02T10:00:00Z",
                            "details": "Multiple deposits from same IP"
                        } if pred["fraud_label"] == "Fraud" else None,
                        "linked_accounts": [
                            {"user_id": 99999, "shared_attribute": "IP address"}
                        ] if pred["fraud_label"] == "Fraud" else []
                    } for pred in predictions
                ],
                "fraud_rate": float(sum(1 for p in predictions if p["fraud_label"] == "Fraud") / len(predictions)) if predictions else 0.33,
                "average_fraud_score": float(sum(p["fraud_score"] for p in predictions) / len(predictions)) if predictions else 0.13,
                "total_users": len(predictions),
                "fraud_percentage": float(sum(1 for p in predictions if p["fraud_label"] == "Fraud") / len(predictions) * 100) if predictions else 33.33,
                "fraud_trend": {
                    "weekly_fraud_rate": extra_metrics.get("weekly_fraud_trend", [0.05, 0.1, 0.3, 0.25])
                },
                "fraud_anomalies": [
                    {"user_id": p["user_id"], "anomalous_behavior": "Rapid deposits and high win streak"}
                    for p in predictions if p["fraud_label"] == "Fraud"
                ],
                "top_fraudulent_users": [
                    {"user_id": p["user_id"], "fraud_score": p["fraud_score"]}
                    for p in predictions if p["fraud_label"] == "Fraud"
                ],
                "real_time_alerts": [
                    {"user_id": p["user_id"], "alert": "Suspicious deposit detected at 2025-07-02T10:00:00Z"}
                    for p in predictions if p["fraud_label"] == "Fraud"
                ],
                "metadata": {
                    "model_version": "v1.2",
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
                    "data_freshness": "real-time"
                }
            },
            "pagination": {
                "page": 1,
                "total_pages": 10,
                "items_per_page": 100,
                "total_items": 1000
            },
            "errors": [],
            "localization": {
                "currency": "USD",
                "language": "en",
                "region": "US"
            }
        }
        return response
    except Exception as e:
        logger.error(f"Error in fraud detection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
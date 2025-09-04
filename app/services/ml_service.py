import joblib
import pandas as pd
from app.config.settings import settings
import logging
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model_dir = settings.MODEL_DIR
        self.churn_model = self.load_model("churn_model.pkl")

    def load_model(self, model_name):
        try:
            model_path = f"{self.model_dir}/{model_name}"
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None

    def predict_churn(self, data):
        try:
            if data.empty:
                return []
            
            # Create mock predictions if model is not available
            user_ids = data["user_id"] if "user_id" in data.columns else range(len(data))
            predictions = []
            
            for user_id in user_ids:
                # Mock probability
                prob = np.random.random()
                predictions.append({
                    "user_id": user_id,
                    "churn_probability": float(prob),
                    "churn_label": "Churn" if prob > 0.5 else "Not Churn",
                    "confidence": 0.95 if prob > 0.5 else 0.9,
                    "priority_score": float(prob * 1.0588235294117647),
                    "retention_recommendation": "Offer free spins and VIP support outreach" if prob > 0.5 else "Send loyalty email with 10% bonus",
                    "feature_importance": {
                        "recency": 0.5 if prob > 0.5 else 0.4,
                        "frequency": 0.3,
                        "monetary": 0.15 if prob > 0.5 else 0.2
                    },
                    "churn_impact": {
                        "user_id": user_id,
                        "estimated_impact": self.estimate_churn_impact(user_id, data)
                    } if prob > 0.5 else {},
                    "high_risk_groups": [
                        {"group": "high frequency, low monetary", "risk": 0.8}
                    ] if prob > 0.5 else [],
                    "campaign_eligibility": "VIP rewards" if prob > 0.5 else "None",
                    "preferred_game": "Unknown",
                    "preferred_payment_method": "Unknown"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting churn: {str(e)}")
            return []

    def predict_ltv(self, data):
        try:
            if data.empty:
                return []
            
            user_ids = data["user_id"] if "user_id" in data.columns else range(len(data))
            predictions = []
            
            for user_id in user_ids:
                # Mock LTV prediction
                ltv = np.random.uniform(300, 1500)
                predictions.append({
                    "user_id": user_id,
                    "predicted_ltv": float(ltv)
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting LTV: {str(e)}")
            return []

    def predict_fraud(self, data):
        try:
            if data.empty:
                return []
            
            user_ids = data["user_id"] if "user_id" in data.columns else range(len(data))
            predictions = []
            
            for user_id in user_ids:
                # Mock fraud prediction
                fraud_score = np.random.random()
                predictions.append({
                    "user_id": user_id,
                    "fraud_score": float(fraud_score),
                    "fraud_label": "Fraud" if fraud_score > 0.8 else "Not Fraud"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting fraud: {str(e)}")
            return []

    def predict_engagement(self, data):
        try:
            if data.empty:
                return []
            
            user_ids = data["user_id"] if "user_id" in data.columns else range(len(data))
            predictions = []
            
            for user_id in user_ids:
                # Mock engagement prediction
                engagement_score = np.random.random()
                predictions.append({
                    "user_id": user_id,
                    "engagement_score": float(engagement_score),
                    "engagement_prediction": "Engaged" if engagement_score > 0.5 else "Not Engaged"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting engagement: {str(e)}")
            return []

    def predict_segmentation(self, data):
        try:
            if data.empty:
                return [], []
            
            user_ids = data["user_id"] if "user_id" in data.columns else range(len(data))
            predictions = []
            
            # Mock segmentation
            segments = [0, 1, 2, 3]  # 4 segments
            
            for user_id in user_ids:
                segment = np.random.choice(segments)
                predictions.append({
                    "user_id": user_id,
                    "segment": segment
                })
            
            return predictions, segments
        except Exception as e:
            logger.error(f"Error predicting segmentation: {str(e)}")
            return [], []

    def estimate_churn_impact(self, user_id, data):
        try:
            # Mock impact estimation
            return float(np.random.uniform(100, 1000))
        except Exception as e:
            logger.error(f"Error estimating churn impact for user {user_id}: {str(e)}")
            return 0.0
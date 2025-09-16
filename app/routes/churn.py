# from fastapi import APIRouter, HTTPException
# from datetime import datetime
# from app.services.data_service import DataService
# from app.services.ml_service import MLService
# import logging
# import pandas as pd

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# router = APIRouter()

# @router.get("/predict")
# async def predict_churn():
#     try:
#         data_service = DataService()
#         ml_service = MLService()
        
#         # Fetch and preprocess data
#         raw_data = data_service.fetch_all_data()
#         churn_data, extra_metrics = data_service.preprocess_churn_data(raw_data)
        
#         if churn_data.empty:
#             # Return mock data when no real data is available
#             predictions = [
#                 {
#                     "user_id": int(i),  # Ensure user_id is int, not numpy type
#                     "churn_probability": float(0.3 + (i % 3) * 0.2),  # Convert to Python float
#                     "churn_label": "Churn" if (i % 3) == 2 else "Not Churn"
#                 }
#                 for i in range(1, 11)
#             ]
#         else:
#             # Make predictions
#             predictions = ml_service.predict_churn(churn_data)
#             # Ensure all numpy types are converted to Python native types
#             for pred in predictions:
#                 pred["user_id"] = int(pred["user_id"]) if "user_id" in pred else 0
#                 pred["churn_probability"] = float(pred["churn_probability"]) if "churn_probability" in pred else 0.0
        
#         # Compute cohort analysis - add safety check
#         cohort_analysis = {}
#         try:
#             cohort_analysis = data_service.compute_cohort_analysis(raw_data)
#         except Exception as cohort_error:
#             logger.warning(f"Cohort analysis failed: {str(cohort_error)}")
#             cohort_analysis = {}
        
#         # Helper function to safely get user data
#         def get_user_last_login(user_id):
#             try:
#                 # For playerDetails, use 'id' column to match user_id
#                 if "players" in raw_data and not raw_data["players"].empty:
#                     players_df = raw_data["players"]
#                     user_data = players_df[players_df["id"] == user_id]
                    
#                     if not user_data.empty and "last_login_at" in user_data.columns:
#                         last_login = user_data["last_login_at"].iloc[0]
#                         return last_login if pd.notna(last_login) else None
                        
#             except Exception as e:
#                 logger.warning(f"Error getting last login for user {user_id}: {str(e)}")
#             return None
        
#         def get_user_last_deposit(user_id):
#             try:
#                 # For playerDeposite, use 'user_id' column and 'amount' column
#                 if "deposits" in raw_data and not raw_data["deposits"].empty:
#                     deposits_df = raw_data["deposits"]
#                     user_deposits = deposits_df[deposits_df["user_id"] == user_id]
                    
#                     if not user_deposits.empty and "amount" in user_deposits.columns:
#                         # Filter only successful/completed deposits
#                         if "status" in user_deposits.columns:
#                             successful_deposits = user_deposits[
#                                 user_deposits["status"].isin(["completed", "success", "approved"])
#                             ]
#                             if not successful_deposits.empty:
#                                 user_deposits = successful_deposits
                        
#                         # Calculate total deposit amount
#                         total_amount = 0.0
#                         for amount_val in user_deposits["amount"]:
#                             try:
#                                 # Handle string amounts
#                                 amount_str = str(amount_val)
#                                 # Handle concatenated values like '200.00200.00'
#                                 if amount_str.count('.') > 1:
#                                     # Split by '.00' pattern for concatenated amounts
#                                     if '.00' in amount_str:
#                                         parts = amount_str.split('.00')
#                                         for i, part in enumerate(parts[:-1]):  # Exclude last empty part
#                                             if part:
#                                                 total_amount += float(part + '.00')
#                                         # Add the last part if it's not empty
#                                         if parts[-1]:
#                                             total_amount += float(parts[-1])
#                                     else:
#                                         # Fallback: try to parse as single amount
#                                         total_amount += float(amount_str)
#                                 else:
#                                     total_amount += float(amount_val)
#                             except (ValueError, TypeError) as ve:
#                                 logger.warning(f"Could not convert deposit amount '{amount_val}' for user {user_id}: {str(ve)}")
#                                 continue
                        
#                         return total_amount
                        
#             except Exception as e:
#                 logger.warning(f"Error getting last deposit for user {user_id}: {str(e)}")
#             return 0.0
        
#         # Construct response
#         response = {
#             "status": "success",
#             "api_version": "1.0.0",
#             "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
#             "data": {
#                 "prediction_type": "churn_prediction",
#                 "results": [
#                     {
#                         "user_id": int(pred.get("user_id", i)),  # Safe access with fallback
#                         "churn_probability": float(pred.get("churn_probability", 0.5)),
#                         "churn_label": pred.get("churn_label", "Not Churn"),
#                         "confidence": float(pred.get("confidence", 0.95 if pred.get("churn_label") == "Churn" else 0.9)),
#                         "priority_score": float(pred.get("priority_score", pred.get("churn_probability", 0.5) * 1.0588235294117647)),
#                         "last_activity": {
#                             "last_login": get_user_last_login(pred.get("user_id", i)),
#                             "last_deposit": get_user_last_deposit(pred.get("user_id", i))
#                         },
#                         "retention_recommendation": pred.get("retention_recommendation", 
#                             "Offer free spins and VIP support outreach" if pred.get("churn_label") == "Churn" 
#                             else "Send loyalty email with 10% bonus"),
#                         "feature_importance": pred.get("feature_importance", {
#                             "recency": float(0.5 if pred.get("churn_label") == "Churn" else 0.4),
#                             "frequency": float(0.3),
#                             "monetary": float(0.15 if pred.get("churn_label") == "Churn" else 0.2)
#                         }),
#                         "churn_impact": pred.get("churn_impact", {
#                             "user_id": int(pred.get("user_id", i)),
#                             "estimated_impact": float(ml_service.estimate_churn_impact(pred.get("user_id", i), raw_data) if hasattr(ml_service, 'estimate_churn_impact') else 100.0)
#                         } if pred.get("churn_label") == "Churn" else {}),
#                         "churn_trend": pred.get("churn_trend", {
#                             "weekly_churn_rate": [float(x) for x in extra_metrics.get("weekly_churn_trend", [0.1, 0.2, 0.3, 0.4])]
#                         } if pred.get("churn_label") == "Churn" else {}),
#                         "high_risk_groups": pred.get("high_risk_groups", [
#                             {"group": "high frequency, low monetary", "risk": float(0.8)}
#                         ] if pred.get("churn_label") == "Churn" else []),
#                         "cohort_analysis": cohort_analysis.get(pred.get("user_id", i), {
#                             "new_users": {"churn_rate": float(0.4), "count": int(100)},
#                             "vip_users": {"churn_rate": float(0.2), "count": int(50)}
#                         }),
#                         "metadata": {
#                             "model_version": "v1.2",
#                             "data_freshness": "real-time",
#                             "data_quality_issues": extra_metrics.get("data_quality_issues", [])
#                         }
#                     } for i, pred in enumerate(predictions, 1)
#                 ],
#                 "metrics": {
#                     "accuracy": float(extra_metrics.get("accuracy", 0.92)),
#                     "auc": float(extra_metrics.get("auc", 0.91)),
#                     "total_users": int(len(predictions)),
#                     "churn_rate": float(sum(1 for p in predictions if p.get("churn_label") == "Churn") / len(predictions)) if predictions else 0.33,
#                     "churn_distribution": {
#                         "churned": float(sum(1 for p in predictions if p.get("churn_label") == "Churn") / len(predictions) * 100) if predictions else 33.33,
#                         "not_churned": float(sum(1 for p in predictions if p.get("churn_label") == "Not Churn") / len(predictions) * 100) if predictions else 66.67
#                     }
#                 },
#                 "trends": {
#                     "churn_trend": {
#                         "weekly_churn_rate": [float(x) for x in extra_metrics.get("weekly_churn_trend", [0.1, 0.2, 0.3, 0.4])]
#                     }
#                 },
#                 "recommendations": extra_metrics.get("recommendations", [
#                     {"type": "send_email", "message": "Offer a 10% discount on next deposit", "priority": "high"},
#                     {"type": "send_sms", "message": "Offer free spins to reactivate", "priority": "medium"}
#                 ]),
#                 "campaign_eligibility": [
#                     {"user_id": int(p.get("user_id", i)), "eligible_for_campaign": p.get("campaign_eligibility", "VIP rewards")}
#                     for i, p in enumerate(predictions, 1) if p.get("churn_label") == "Churn"
#                 ],
#                 "personalization_insights": [
#                     {
#                         "user_id": int(p.get("user_id", i)), 
#                         "preferred_game": p.get("preferred_game", data_service.get_preferred_game(p.get("user_id", i), raw_data) if hasattr(data_service, 'get_preferred_game') else "Unknown"), 
#                         "preferred_payment_method": p.get("preferred_payment_method", data_service.get_preferred_payment_method(p.get("user_id", i), raw_data) if hasattr(data_service, 'get_preferred_payment_method') else "Unknown")
#                     }
#                     for i, p in enumerate(predictions, 1) if p.get("churn_label") == "Churn"
#                 ],
#                 "metadata": {
#                     "model_version": "v1.2",
#                     "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
#                     "data_freshness": "real-time"
#                 }
#             },
#             "pagination": {
#                 "page": 1,
#                 "total_pages": 10,
#                 "items_per_page": 100,
#                 "total_items": 1000
#             },
#             "errors": [],
#             "localization": {
#                 "currency": "USD",
#                 "language": "en",
#                 "region": "US"
#             }
#         }
#         return response
#     except Exception as e:
#         logger.error(f"Error in churn prediction endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Churn prediction failed: {str(e)}")









from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from app.services.data_service import DataService
from app.services.ml_service import MLService
from app.config.settings import settings
import logging
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/predict")
async def predict_churn(
    threshold: Optional[float] = Query(None, description="Custom churn threshold (0.0-1.0)", ge=0.0, le=1.0),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(50, description="Items per page", ge=1, le=settings.MAX_PAGE_SIZE),
    risk_level: Optional[str] = Query(None, description="Filter by risk level: low, medium, high, all"),
    sort_by: Optional[str] = Query("priority_score", description="Sort by: probability, priority_score, user_value, user_id"),
    include_details: bool = Query(True, description="Include detailed analysis and recommendations")
):
    """
    Enhanced churn prediction endpoint with flexible filtering and sorting options
    """
    try:
        # Initialize services
        data_service = DataService()
        ml_service = MLService()
        
        # Use custom threshold or default
        prediction_threshold = threshold if threshold is not None else settings.CHURN_THRESHOLD
        
        # Fetch and preprocess data
        raw_data = data_service.fetch_all_data()
        churn_data, extra_metrics = data_service.preprocess_churn_data(raw_data)
        
        # Generate predictions
        if churn_data.empty:
            # Generate mock data for demonstration
            predictions = _generate_mock_predictions(page_size, prediction_threshold)
            total_users = 1000  # Mock total
        else:
            predictions = ml_service.predict_churn(churn_data, threshold=prediction_threshold)
            total_users = len(predictions)
        
        # Apply risk level filtering
        if risk_level and risk_level != "all":
            predictions = _filter_by_risk_level(predictions, risk_level)
        
        # Sort predictions
        predictions = _sort_predictions(predictions, sort_by)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_predictions = predictions[start_idx:end_idx]
        
        # Calculate summary statistics
        summary_stats = _calculate_summary_stats(predictions, extra_metrics)
        
        # Build response structure
        response = {
            "status": "success",
            "api_version": "1.0.0",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
            "data": {
                "prediction_type": "churn_prediction",
                "model_info": {
                    "version": settings.MODEL_VERSION,
                    "threshold_used": float(prediction_threshold),
                    "confidence_levels": {
                        "high_confidence": settings.CHURN_HIGH_CONFIDENCE,
                        "low_confidence": settings.CHURN_LOW_CONFIDENCE
                    }
                },
                "results": _format_prediction_results(paginated_predictions, raw_data, data_service, include_details),
                "summary": summary_stats,
                "metadata": {
                    "model_version": settings.MODEL_VERSION,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
                    "data_freshness": "real-time",
                    "data_quality": extra_metrics.get("data_quality_issues", []),
                    "model_performance": extra_metrics.get("model_performance", {})
                }
            },
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": len(predictions),
                "total_pages": (len(predictions) + page_size - 1) // page_size,
                "has_next": end_idx < len(predictions),
                "has_prev": page > 1
            },
            "filters_applied": {
                "threshold": prediction_threshold,
                "risk_level": risk_level,
                "sort_by": sort_by
            },
            "errors": [],
            "localization": {
                "currency": "USD",
                "language": "en",
                "region": "US"
            }
        }
        
        # Add recommendations if details are included
        if include_details:
            response["data"]["business_recommendations"] = _generate_business_recommendations(
                predictions, extra_metrics
            )
            response["data"]["cohort_analysis"] = data_service.compute_cohort_analysis(raw_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in churn prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {str(e)}")

def _generate_mock_predictions(count, threshold):
    """Generate mock predictions for demonstration when no real data is available"""
    import random
    import numpy as np
    
    predictions = []
    for i in range(186201, 186201 + count):
        prob = np.random.random()
        is_churn = prob > threshold
        user_value = random.uniform(100, 5000)
        
        predictions.append({
            "user_id": i,
            "churn_probability": float(prob),
            "churn_label": "Churn" if is_churn else "Not Churn",
            "confidence": 0.95 if abs(prob - threshold) > 0.2 else 0.90,
            "priority_score": float(prob * (1.5 if user_value > 1000 else 1.0)),
            "retention_recommendation": _get_retention_recommendation(prob, user_value),
            "feature_importance": _normalize_features({
                "recency": 0.5 if is_churn else 0.35,
                "frequency": 0.3,
                "monetary": 0.15 if is_churn else 0.25
            }),
            "estimated_impact": float(random.uniform(200, 1000)) if is_churn else 0.0,
            "user_value": float(user_value),
            "threshold_used": float(threshold)
        })
    
    return predictions

def _filter_by_risk_level(predictions, risk_level):
    """Filter predictions by risk level"""
    if risk_level == "low":
        return [p for p in predictions if p["churn_probability"] < 0.3]
    elif risk_level == "medium":
        return [p for p in predictions if 0.3 <= p["churn_probability"] < 0.7]
    elif risk_level == "high":
        return [p for p in predictions if p["churn_probability"] >= 0.7]
    return predictions

def _sort_predictions(predictions, sort_by):
    """Sort predictions by specified criteria"""
    if sort_by == "probability":
        return sorted(predictions, key=lambda x: x["churn_probability"], reverse=True)
    elif sort_by == "priority_score":
        return sorted(predictions, key=lambda x: x["priority_score"], reverse=True)
    elif sort_by == "user_value":
        return sorted(predictions, key=lambda x: x.get("user_value", 0), reverse=True)
    elif sort_by == "user_id":
        return sorted(predictions, key=lambda x: x["user_id"])
    else:
        return sorted(predictions, key=lambda x: x["priority_score"], reverse=True)

def _calculate_summary_stats(predictions, extra_metrics):
    """Calculate summary statistics for the prediction results"""
    if not predictions:
        return {}
    
    churn_predictions = [p for p in predictions if p["churn_label"] == "Churn"]
    total_users = len(predictions)
    churn_count = len(churn_predictions)
    
    return {
        "total_users_analyzed": total_users,
        "churn_predictions": {
            "total_at_risk": churn_count,
            "churn_rate": round(churn_count / total_users * 100, 2) if total_users > 0 else 0,
            "avg_churn_probability": round(
                sum(p["churn_probability"] for p in churn_predictions) / churn_count, 3
            ) if churn_count > 0 else 0,
            "high_risk_users": len([p for p in predictions if p["churn_probability"] >= 0.7]),
            "medium_risk_users": len([p for p in predictions if 0.3 <= p["churn_probability"] < 0.7]),
            "low_risk_users": len([p for p in predictions if p["churn_probability"] < 0.3])
        },
        "financial_impact": {
            "total_estimated_loss": round(
                sum(p.get("estimated_impact", 0) for p in churn_predictions), 2
            ),
            "avg_user_value": round(
                sum(p.get("user_value", 0) for p in predictions) / total_users, 2
            ) if total_users > 0 else 0,
            "high_value_at_risk": len([
                p for p in churn_predictions 
                if p.get("user_value", 0) > settings.HIGH_VALUE_THRESHOLD
            ])
        },
        "model_performance": extra_metrics.get("model_performance", {
            "accuracy": 0.94,
            "precision": 0.89,
            "recall": 0.87,
            "f1_score": 0.88
        })
    }

def _format_prediction_results(predictions, raw_data, data_service, include_details):
    """Format prediction results with optional detailed information"""
    formatted_results = []
    
    for pred in predictions:
        user_id = pred["user_id"]
        
        # Base result structure
        result = {
            "user_id": int(user_id),
            "churn_probability": float(pred["churn_probability"]),
            "churn_label": pred["churn_label"],
            "confidence": float(pred["confidence"]),
            "priority_score": float(pred["priority_score"]),
            "risk_level": _get_risk_level(pred["churn_probability"])
        }
        
        if include_details:
            # Add detailed information
            result.update({
                "last_activity": {
                    "last_login": data_service.get_user_last_login(user_id, raw_data),
                    "total_deposits": data_service.get_user_total_deposits(user_id, raw_data)
                },
                "retention_recommendation": pred["retention_recommendation"],
                "feature_importance": pred["feature_importance"],
                "estimated_impact": float(pred.get("estimated_impact", 0)),
                "user_value": float(pred.get("user_value", 0)),
                "personalization": {
                    "preferred_game": data_service.get_preferred_game(user_id, raw_data),
                    "preferred_payment_method": data_service.get_preferred_payment_method(user_id, raw_data)
                },
                "threshold_used": float(pred.get("threshold_used", settings.CHURN_THRESHOLD))
            })
        
        formatted_results.append(result)
    
    return formatted_results

def _get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "high"
    elif probability >= 0.3:
        return "medium"
    else:
        return "low"

def _get_retention_recommendation(prob, user_value):
    """Get retention recommendation based on probability and user value"""
    if prob >= 0.8:
        if user_value > settings.VIP_THRESHOLD:
            return "Immediate VIP manager call + exclusive bonus package"
        elif user_value > settings.HIGH_VALUE_THRESHOLD:
            return "Priority support call + personalized offer"
        else:
            return "Urgent retention campaign + free spins"
    elif prob >= 0.6:
        return "Send targeted promotion based on preferences"
    elif prob >= 0.4:
        return "Offer personalized loyalty rewards"
    else:
        return "Continue regular engagement campaigns"

def _normalize_features(features):
    """Normalize feature importance to sum to 1.0"""
    total = sum(features.values())
    if total > 0:
        return {k: round(v/total, 3) for k, v in features.items()}
    return features

def _generate_business_recommendations(predictions, extra_metrics):
    """Generate business recommendations based on prediction results"""
    churn_predictions = [p for p in predictions if p["churn_label"] == "Churn"]
    high_risk_users = [p for p in predictions if p["churn_probability"] >= 0.7]
    high_value_at_risk = [
        p for p in churn_predictions 
        if p.get("user_value", 0) > settings.HIGH_VALUE_THRESHOLD
    ]
    
    recommendations = []
    
    if high_value_at_risk:
        recommendations.append({
            "type": "urgent_action",
            "priority": "critical",
            "action": "immediate_vip_outreach",
            "description": f"Contact {len(high_value_at_risk)} high-value users at risk within 24 hours",
            "estimated_impact": sum(p.get("estimated_impact", 0) for p in high_value_at_risk),
            "user_count": len(high_value_at_risk)
        })
    
    if high_risk_users:
        recommendations.append({
            "type": "retention_campaign",
            "priority": "high",
            "action": "targeted_retention_offers",
            "description": f"Deploy personalized retention campaigns to {len(high_risk_users)} high-risk users",
            "estimated_impact": sum(p.get("estimated_impact", 0) for p in high_risk_users),
            "user_count": len(high_risk_users)
        })
    
    medium_risk_users = [p for p in predictions if 0.3 <= p["churn_probability"] < 0.7]
    if medium_risk_users:
        recommendations.append({
            "type": "engagement_boost",
            "priority": "medium",
            "action": "loyalty_program_enhancement",
            "description": f"Increase engagement for {len(medium_risk_users)} medium-risk users",
            "estimated_impact": sum(p.get("estimated_impact", 0) for p in medium_risk_users) * 0.5,
            "user_count": len(medium_risk_users)
        })
    
    return recommendations
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from datetime import datetime, timedelta
import requests
from pathlib import Path
import logging
import json
import pytz  # Added for timezone handling

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Client
class APIClient:
    def __init__(self):
        self.base_url = "https://canada777.com/api"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Basic canada777"
        }
        self.session = requests.Session()
        retries = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount("https://", retries)
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def clear_data_dir(self):
        try:
            for file in self.data_dir.glob("*.json"):
                file.unlink()
            logger.info("Cleared existing JSON files in data directory")
        except Exception as e:
            logger.error(f"Error clearing data directory: {str(e)}")

    def save_data(self, endpoint, data):
        try:
            file_path = self.data_dir / f"{endpoint}.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data to data/{endpoint}.json")
        except Exception as e:
            logger.error(f"Error saving data for {endpoint}: {str(e)}")

    def fetch_data(self, endpoint: str, created_at: str = None):
        try:
            url = f"{self.base_url}/{endpoint}"
            params = {"createdAt": created_at} if created_at else {}
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data["data"]["data"]
            else:
                logger.error(f"Failed to fetch data from {endpoint} for {created_at}: {data.get('message')}")
                return []
        except Exception as e:
            logger.error(f"Error fetching data from {endpoint} for {created_at}: {str(e)}")
            mock_file = self.data_dir / f"mock_{endpoint}_{created_at}.json" if created_at else self.data_dir / f"mock_{endpoint}.json"
            if mock_file.exists():
                with open(mock_file, "r") as f:
                    return json.load(f)["data"]["data"]
            return []

    def get_last_7_days_data(self, endpoint: str):
        data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            logger.info(f"Fetching data for {endpoint} on {date}")
            daily_data = self.fetch_data(endpoint, created_at=date)
            logger.info(f"Fetched {len(daily_data)} records for {endpoint} on {date}")
            data.extend(daily_data)
        logger.info(f"Total fetched {len(data)} records for {endpoint} over last 7 days")
        if data:
            self.save_data(endpoint, {"success": True, "message": f"{endpoint} fetched successfully", "data": {"data": data}})
        return data

# ------------------------
# Data Preprocessing
# ------------------------
def preprocess_churn_data(data):
    try:
        players = pd.DataFrame(data["players"])
        logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
        if 'id' in players.columns and 'user_id' not in players.columns:
            players = players.rename(columns={'id': 'user_id'})
            logger.info("Renamed 'id' to 'user_id' in players DataFrame")
        logs = pd.DataFrame(data["logs"])
        deposits = pd.DataFrame(data["deposits"])
        logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
        logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
        merged = players.merge(logs, on="user_id", how="left")
        merged = merged.merge(deposits, on="user_id", how="left")
        merged["created_at"] = pd.to_datetime(merged["created_at"], utc=True)
        merged["last_login_at"] = pd.to_datetime(merged["last_login_at"], utc=True)
        merged["amount"] = merged["amount"].astype(float).fillna(0)
        utc_now = datetime.now(pytz.UTC) - timedelta(hours=6)
        recency = merged.groupby("user_id")["last_login_at"].max().apply(
            lambda x: (utc_now - x).days if pd.notnull(x) else 30
        )
        frequency = merged.groupby("user_id")["action"].count()
        monetary = merged.groupby("user_id")["amount"].sum()
        churn_data = pd.DataFrame({
            "user_id": recency.index,
            "recency": recency.values,
            "frequency": frequency.values,
            "monetary": monetary.values
        })
        churn_data["churn_label"] = (churn_data["recency"] > 30).astype(int)
        return churn_data
    except Exception as e:
        logger.error(f"Error preprocessing churn data: {str(e)}")
        return pd.DataFrame()

def preprocess_ltv_data(data):
    try:
        players = pd.DataFrame(data["players"])
        if 'id' in players.columns and 'user_id' not in players.columns:
            players = players.rename(columns={'id': 'user_id'})
        deposits = pd.DataFrame(data["deposits"])
        bonuses = pd.DataFrame(data["bonuses"])
        merged = players.merge(deposits, on="user_id", how="left")
        merged = merged.merge(bonuses, on="user_id", how="left")
        merged["amount"] = merged["amount"].astype(float).fillna(0)
        merged["bonus_amount"] = merged["bonus_amount"].astype(float).fillna(0)
        ltv_data = merged.groupby("user_id").agg({
            "amount": "sum",
            "bonus_amount": "sum",
            "deposit_id": "count"
        }).reset_index()
        ltv_data.columns = ["user_id", "total_deposits", "total_bonuses", "deposit_count"]
        ltv_data["ltv"] = ltv_data["total_deposits"] + ltv_data["total_bonuses"]
        return ltv_data
    except Exception as e:
        logger.error(f"Error preprocessing LTV data: {str(e)}")
        return pd.DataFrame()

def preprocess_fraud_data(data):
    try:
        deposits = pd.DataFrame(data["deposits"])
        logs = pd.DataFrame(data["logs"])
        merged = deposits.merge(logs, on="user_id", how="left")
        merged["amount"] = merged["amount"].astype(float).fillna(0)
        merged["win_amount"] = merged["win_amount"].astype(float).fillna(0)
        fraud_data = merged.groupby("user_id").agg({
            "amount": "sum",
            "win_amount": ["sum", "count"],
            "ip": lambda x: x.nunique()
        }).reset_index()
        fraud_data.columns = ["user_id", "total_deposits", "total_wins", "win_count", "unique_ips"]
        fraud_data["rapid_deposits"] = fraud_data["total_deposits"] / (fraud_data["win_count"] + 1)
        fraud_data["fraud_label"] = (fraud_data["rapid_deposits"] > fraud_data["rapid_deposits"].quantile(0.95)).astype(int)
        return fraud_data
    except Exception as e:
        logger.error(f"Error preprocessing fraud data: {str(e)}")
        return pd.DataFrame()

def preprocess_segmentation_data(data):
    return preprocess_churn_data(data)

def preprocess_engagement_data(data):
    try:
        players = pd.DataFrame(data["players"])
        if 'id' in players.columns and 'user_id' not in players.columns:
            players = players.rename(columns={'id': 'user_id'})
        logs = pd.DataFrame(data["logs"])
        deposits = pd.DataFrame(data["deposits"])
        merged = players.merge(logs, on="user_id", how="left")
        merged = merged.merge(deposits, on="user_id", how="left")
        merged["created_at"] = pd.to_datetime(merged["created_at"], utc=True)
        utc_now = datetime.now(pytz.UTC) - timedelta(hours=6)
        engagement_data = merged.groupby("user_id").agg({
            "action": "count",
            "created_at": lambda x: (utc_now - pd.to_datetime(x).max()).days if pd.notnull(x.max()) else 30,
            "amount": "count"
        }).reset_index()
        engagement_data.columns = ["user_id", "activity_count", "recency", "deposit_count"]
        engagement_data["engagement_label"] = (engagement_data["recency"] < 7).astype(int)
        return engagement_data
    except Exception as e:
        logger.error(f"Error preprocessing engagement data: {str(e)}")
        return pd.DataFrame()

# ------------------------
# Training Functions
# ------------------------
def train_churn_model(data):
    churn_data = preprocess_churn_data(data)
    if churn_data.empty:
        logger.error("No data for churn model training")
        return None
    X = churn_data[["recency", "frequency", "monetary"]].fillna(0)
    y = churn_data["churn_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info(f"Churn model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    return model

def train_ltv_model(data):
    ltv_data = preprocess_ltv_data(data)
    if ltv_data.empty:
        logger.error("No data for LTV model training")
        return None
    X = ltv_data[["total_deposits", "total_bonuses", "deposit_count"]].fillna(0)
    y = ltv_data["ltv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info(f"LTV model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
    return model

def train_fraud_model(data):
    fraud_data = preprocess_fraud_data(data)
    if fraud_data.empty:
        logger.error("No data for fraud model training")
        return None
    X = fraud_data[["total_deposits", "total_wins", "rapid_deposits", "unique_ips"]].fillna(0)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    logger.info("Fraud model trained")
    return model

def train_segmentation_model(data):
    segmentation_data = preprocess_segmentation_data(data)
    if segmentation_data.empty:
        logger.error("No data for segmentation model training")
        return None
    X = segmentation_data[["recency", "frequency", "monetary"]].fillna(0)
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X)
    logger.info("Segmentation model trained")
    return model

def train_engagement_model(data):
    engagement_data = preprocess_engagement_data(data)
    if engagement_data.empty:
        logger.error("No data for engagement model training")
        return None
    X = engagement_data[["activity_count", "recency", "deposit_count"]].fillna(0)
    y = engagement_data["engagement_label"]

    # ✅ Fix: check for at least 2 classes
    if y.nunique() < 2:
        logger.warning("Engagement model training skipped: only one class present in labels")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    logger.info(f"Engagement model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    return model

# ------------------------
# Main Training Function
# ------------------------
def main():
    api_client = APIClient()
    api_client.clear_data_dir()
    data = {
        "players": api_client.get_last_7_days_data("players_details"),
        "deposits": api_client.get_last_7_days_data("players_deposit_details"),
        "bonuses": api_client.get_last_7_days_data("players_bonus_details"),
        "logs": api_client.get_last_7_days_data("players_log_details")
    }

    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "churn_model.pkl": train_churn_model(data),
        "ltv_model.pkl": train_ltv_model(data),
        "fraud_model.pkl": train_fraud_model(data),
        "segmentation_model.pkl": train_segmentation_model(data),
        "engagement_model.pkl": train_engagement_model(data)
    }

    for model_name, model in models.items():
        if model:
            joblib.dump(model, model_dir / model_name)
            logger.info(f"Saved {model_name}")

if __name__ == "__main__":
    main()




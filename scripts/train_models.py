# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error
# from datetime import datetime, timedelta
# import requests
# from pathlib import Path
# import logging
# import json

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # API Client
# class APIClient:
#     def __init__(self):
#         self.base_url = "https://canada777.com/api"
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Basic canada777"
#         }
#         self.session = requests.Session()
#         retries = requests.adapters.HTTPAdapter(max_retries=3)
#         self.session.mount("https://", retries)
#         self.data_dir = Path("data")
#         self.data_dir.mkdir(parents=True, exist_ok=True)

#     def save_data(self, endpoint, data):
#         try:
#             with open(self.data_dir / f"{endpoint}.json", "w") as f:
#                 json.dump(data, f, indent=2)
#             logger.info(f"Saved data to data/{endpoint}.json")
#         except Exception as e:
#             logger.error(f"Error saving data for {endpoint}: {str(e)}")

#     def fetch_data(self, endpoint: str, created_at: str = None):
#         try:
#             url = f"{self.base_url}/{endpoint}"
#             params = {"createdAt": created_at} if created_at else {}
#             response = self.session.get(url, headers=self.headers, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if data.get("success"):
#                 self.save_data(endpoint, data)
#                 return data["data"]["data"]
#             else:
#                 logger.error(f"Failed to fetch data from {endpoint}: {data.get('message')}")
#                 return []
#         except Exception as e:
#             logger.error(f"Error fetching data from {endpoint}: {str(e)}")
#             mock_file = self.data_dir / f"mock_{endpoint}.json"
#             if mock_file.exists():
#                 with open(mock_file, "r") as f:
#                     return json.load(f)["data"]["data"]
#             return []

#     def get_last_7_days_data(self, endpoint: str):
#         end_date = datetime.now().strftime("%Y-%m-%d")
#         start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
#         return self.fetch_data(endpoint, created_at=start_date)

# # Data Preprocessing
# def preprocess_churn_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         merged = players.merge(logs, on="user_id", how="left")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         merged["last_login_at"] = pd.to_datetime(merged["last_login_at"])
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         recency = (datetime.now() - merged.groupby("user_id")["last_login_at"].max()).dt.days
#         frequency = merged.groupby("user_id")["action"].count()
#         monetary = merged.groupby("user_id")["amount"].sum()
#         churn_data = pd.DataFrame({
#             "user_id": recency.index,
#             "recency": recency.values,
#             "frequency": frequency.values,
#             "monetary": monetary.values
#         })
#         churn_data["churn_label"] = (churn_data["recency"] > 30).astype(int)
#         return churn_data
#     except Exception as e:
#         logger.error(f"Error preprocessing churn data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_ltv_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         deposits = pd.DataFrame(data["deposits"])
#         bonuses = pd.DataFrame(data["bonuses"])
#         merged = players.merge(deposits, on="user_id", how="left")
#         merged = merged.merge(bonuses, on="user_id", how="left")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["bonus_amount"] = merged["bonus_amount"].astype(float).fillna(0)
#         ltv_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "bonus_amount": "sum",
#             "deposit_id": "count"
#         }).reset_index()
#         ltv_data.columns = ["user_id", "total_deposits", "total_bonuses", "deposit_count"]
#         ltv_data["ltv"] = ltv_data["total_deposits"] + ltv_data["total_bonuses"]
#         return ltv_data
#     except Exception as e:
#         logger.error(f"Error preprocessing LTV data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_fraud_data(data):
#     try:
#         deposits = pd.DataFrame(data["deposits"])
#         logs = pd.DataFrame(data["logs"])
#         merged = deposits.merge(logs, on="user_id", how="left")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["win_amount"] = merged["win_amount"].astype(float).fillna(0)
#         fraud_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "win_amount": ["sum", "count"],
#             "ip": lambda x: x.nunique()
#         }).reset_index()
#         fraud_data.columns = ["user_id", "total_deposits", "total_wins", "win_count", "unique_ips"]
#         fraud_data["rapid_deposits"] = fraud_data["total_deposits"] / (fraud_data["win_count"] + 1)
#         fraud_data["fraud_label"] = (fraud_data["rapid_deposits"] > fraud_data["rapid_deposits"].quantile(0.95)).astype(int)
#         return fraud_data
#     except Exception as e:
#         logger.error(f"Error preprocessing fraud data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_segmentation_data(data):
#     return preprocess_churn_data(data)

# def preprocess_engagement_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         merged = players.merge(logs, on="user_id", how="left")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         engagement_data = merged.groupby("user_id").agg({
#             "action": "count",
#             "created_at": lambda x: (datetime.now() - pd.to_datetime(x).max()).dt.days,
#             "amount": "count"
#         }).reset_index()
#         engagement_data.columns = ["user_id", "activity_count", "recency", "deposit_count"]
#         engagement_data["engagement_label"] = (engagement_data["recency"] < 7).astype(int)
#         return engagement_data
#     except Exception as e:
#         logger.error(f"Error preprocessing engagement data: {str(e)}")
#         return pd.DataFrame()

# # Training Functions
# def train_churn_model(data):
#     churn_data = preprocess_churn_data(data)
#     if churn_data.empty:
#         logger.error("No data for churn model training")
#         return None
#     X = churn_data[["recency", "frequency", "monetary"]].fillna(0)
#     y = churn_data["churn_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Churn model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_ltv_model(data):
#     ltv_data = preprocess_ltv_data(data)
#     if ltv_data.empty:
#         logger.error("No data for LTV model training")
#         return None
#     X = ltv_data[["total_deposits", "total_bonuses", "deposit_count"]].fillna(0)
#     y = ltv_data["ltv"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"LTV model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_fraud_model(data):
#     fraud_data = preprocess_fraud_data(data)
#     if fraud_data.empty:
#         logger.error("No data for fraud model training")
#         return None
#     X = fraud_data[["total_deposits", "total_wins", "rapid_deposits", "unique_ips"]].fillna(0)
#     model = IsolationForest(contamination=0.1, random_state=42)
#     model.fit(X)
#     logger.info("Fraud model trained")
#     return model

# def train_segmentation_model(data):
#     segmentation_data = preprocess_segmentation_data(data)
#     if segmentation_data.empty:
#         logger.error("No data for segmentation model training")
#         return None
#     X = segmentation_data[["recency", "frequency", "monetary"]].fillna(0)
#     model = KMeans(n_clusters=4, random_state=42)
#     model.fit(X)
#     logger.info("Segmentation model trained")
#     return model

# def train_engagement_model(data):
#     engagement_data = preprocess_engagement_data(data)
#     if engagement_data.empty:
#         logger.error("No data for engagement model training")
#         return None
#     X = engagement_data[["activity_count", "recency", "deposit_count"]].fillna(0)
#     y = engagement_data["engagement_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Engagement model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# # Main Training Function
# def main():
#     api_client = APIClient()
#     data = {
#         "players": api_client.get_last_7_days_data("players_details"),
#         "deposits": api_client.get_last_7_days_data("players_deposit_details"),
#         "bonuses": api_client.get_last_7_days_data("players_bonus_details"),
#         "logs": api_client.get_last_7_days_data("players_log_details")
#     }

#     model_dir = Path("data/models")
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Train and save models
#     models = {
#         "churn_model.pkl": train_churn_model(data),
#         "ltv_model.pkl": train_ltv_model(data),
#         "fraud_model.pkl": train_fraud_model(data),
#         "segmentation_model.pkl": train_segmentation_model(data),
#         "engagement_model.pkl": train_engagement_model(data)
#     }

#     for model_name, model in models.items():
#         if model:
#             joblib.dump(model, model_dir / model_name)
#             logger.info(f"Saved {model_name}")

# if __name__ == "__main__":
#     main()





# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error
# from datetime import datetime, timedelta
# import requests
# from pathlib import Path
# import logging
# import json
# import shutil

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # API Client
# class APIClient:
#     def __init__(self):
#         self.base_url = "https://canada777.com/api"
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Basic canada777"
#         }
#         self.session = requests.Session()
#         retries = requests.adapters.HTTPAdapter(max_retries=3)
#         self.session.mount("https://", retries)
#         self.data_dir = Path("data")
#         self.data_dir.mkdir(parents=True, exist_ok=True)

#     def clear_data_dir(self):
#         try:
#             for file in self.data_dir.glob("*.json"):
#                 file.unlink()
#             logger.info("Cleared existing JSON files in data directory")
#         except Exception as e:
#             logger.error(f"Error clearing data directory: {str(e)}")

#     def save_data(self, endpoint, data):
#         try:
#             file_path = self.data_dir / f"{endpoint}.json"
#             with open(file_path, "w") as f:
#                 json.dump(data, f, indent=2)
#             logger.info(f"Saved data to data/{endpoint}.json")
#         except Exception as e:
#             logger.error(f"Error saving data for {endpoint}: {str(e)}")

#     def fetch_data(self, endpoint: str, created_at: str = None):
#         try:
#             url = f"{self.base_url}/{endpoint}"
#             params = {"createdAt": created_at} if created_at else {}
#             response = self.session.get(url, headers=self.headers, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if data.get("success"):
#                 return data["data"]["data"]
#             else:
#                 logger.error(f"Failed to fetch data from {endpoint} for {created_at}: {data.get('message')}")
#                 return []
#         except Exception as e:
#             logger.error(f"Error fetching data from {endpoint} for {created_at}: {str(e)}")
#             mock_file = self.data_dir / f"mock_{endpoint}_{created_at}.json" if created_at else self.data_dir / f"mock_{endpoint}.json"
#             if mock_file.exists():
#                 with open(mock_file, "r") as f:
#                     return json.load(f)["data"]["data"]
#             return []

#     def get_last_7_days_data(self, endpoint: str):
#         data = []
#         for i in range(7):
#             date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
#             logger.info(f"Fetching data for {endpoint} on {date}")
#             daily_data = self.fetch_data(endpoint, created_at=date)
#             logger.info(f"Fetched {len(daily_data)} records for {endpoint} on {date}")
#             data.extend(daily_data)
#         logger.info(f"Total fetched {len(data)} records for {endpoint} over last 7 days")
#         if data:
#             self.save_data(endpoint, {"success": True, "message": f"{endpoint} fetched successfully", "data": {"data": data}})
#         return data

# # Data Preprocessing
# def preprocess_churn_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         merged["last_login_at"] = pd.to_datetime(merged["last_login_at"])
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         recency = (datetime.now() - timedelta(hours=6)).groupby(merged["user_id"])["last_login_at"].max().dt.days
#         frequency = merged.groupby("user_id")["action"].count()
#         monetary = merged.groupby("user_id")["amount"].sum()
#         churn_data = pd.DataFrame({
#             "user_id": recency.index,
#             "recency": recency.values,
#             "frequency": frequency.values,
#             "monetary": monetary.values
#         })
#         churn_data["churn_label"] = (churn_data["recency"] > 30).astype(int)
#         logger.info(f"Churn DataFrame shape: {churn_data.shape}")
#         return churn_data
#     except Exception as e:
#         logger.error(f"Error preprocessing churn data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_ltv_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         deposits = pd.DataFrame(data["deposits"])
#         bonuses = pd.DataFrame(data["bonuses"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Bonuses DataFrame columns: {bonuses.columns.tolist()}")
#         merged = players.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged players and deposits: {merged.shape}")
#         merged = merged.merge(bonuses, on="user_id", how="left")
#         logger.info(f"Merged with bonuses: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["bonus_amount"] = merged["bonus_amount"].astype(float).fillna(0)
#         ltv_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "bonus_amount": "sum",
#             "deposit_id": "count"
#         }).reset_index()
#         ltv_data.columns = ["user_id", "total_deposits", "total_bonuses", "deposit_count"]
#         ltv_data["ltv"] = ltv_data["total_deposits"] + ltv_data["total_bonuses"]
#         logger.info(f"LTV DataFrame shape: {ltv_data.shape}")
#         return ltv_data
#     except Exception as e:
#         logger.error(f"Error preprocessing LTV data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_fraud_data(data):
#     try:
#         deposits = pd.DataFrame(data["deposits"])
#         logs = pd.DataFrame(data["logs"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         merged = deposits.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged deposits and logs: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["win_amount"] = merged["win_amount"].astype(float).fillna(0)
#         fraud_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "win_amount": ["sum", "count"],
#             "ip": lambda x: x.nunique()
#         }).reset_index()
#         fraud_data.columns = ["user_id", "total_deposits", "total_wins", "win_count", "unique_ips"]
#         fraud_data["rapid_deposits"] = fraud_data["total_deposits"] / (fraud_data["win_count"] + 1)
#         fraud_data["fraud_label"] = (fraud_data["rapid_deposits"] > fraud_data["rapid_deposits"].quantile(0.95)).astype(int)
#         logger.info(f"Fraud DataFrame shape: {fraud_data.shape}")
#         return fraud_data
#     except Exception as e:
#         logger.error(f"Error preprocessing fraud data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_segmentation_data(data):
#     return preprocess_churn_data(data)

# def preprocess_engagement_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         engagement_data = merged.groupby("user_id").agg({
#             "action": "count",
#             "created_at": lambda x: (datetime.now() - timedelta(hours=6)).days - pd.to_datetime(x).max().day,
#             "amount": "count"
#         }).reset_index()
#         engagement_data.columns = ["user_id", "activity_count", "recency", "deposit_count"]
#         engagement_data["engagement_label"] = (engagement_data["recency"] < 7).astype(int)
#         logger.info(f"Engagement DataFrame shape: {engagement_data.shape}")
#         return engagement_data
#     except Exception as e:
#         logger.error(f"Error preprocessing engagement data: {str(e)}")
#         return pd.DataFrame()

# # Training Functions
# def train_churn_model(data):
#     churn_data = preprocess_churn_data(data)
#     if churn_data.empty:
#         logger.error("No data for churn model training")
#         return None
#     X = churn_data[["recency", "frequency", "monetary"]].fillna(0)
#     y = churn_data["churn_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Churn model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_ltv_model(data):
#     ltv_data = preprocess_ltv_data(data)
#     if ltv_data.empty:
#         logger.error("No data for LTV model training")
#         return None
#     X = ltv_data[["total_deposits", "total_bonuses", "deposit_count"]].fillna(0)
#     y = ltv_data["ltv"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"LTV model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_fraud_model(data):
#     fraud_data = preprocess_fraud_data(data)
#     if fraud_data.empty:
#         logger.error("No data for fraud model training")
#         return None
#     X = fraud_data[["total_deposits", "total_wins", "rapid_deposits", "unique_ips"]].fillna(0)
#     model = IsolationForest(contamination=0.1, random_state=42)
#     model.fit(X)
#     logger.info("Fraud model trained")
#     return model

# def train_segmentation_model(data):
#     segmentation_data = preprocess_segmentation_data(data)
#     if segmentation_data.empty:
#         logger.error("No data for segmentation model training")
#         return None
#     X = segmentation_data[["recency", "frequency", "monetary"]].fillna(0)
#     model = KMeans(n_clusters=4, random_state=42)
#     model.fit(X)
#     logger.info("Segmentation model trained")
#     return model

# def train_engagement_model(data):
#     engagement_data = preprocess_engagement_data(data)
#     if engagement_data.empty:
#         logger.error("No data for engagement model training")
#         return None
#     X = engagement_data[["activity_count", "recency", "deposit_count"]].fillna(0)
#     y = engagement_data["engagement_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Engagement model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# # Main Training Function
# def main():
#     api_client = APIClient()
#     api_client.clear_data_dir()  # Clear existing JSON files
#     data = {
#         "players": api_client.get_last_7_days_data("players_details"),
#         "deposits": api_client.get_last_7_days_data("players_deposit_details"),
#         "bonuses": api_client.get_last_7_days_data("players_bonus_details"),
#         "logs": api_client.get_last_7_days_data("players_log_details")
#     }

#     model_dir = Path("data/models")
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Train and save models
#     models = {
#         "churn_model.pkl": train_churn_model(data),
#         "ltv_model.pkl": train_ltv_model(data),
#         "fraud_model.pkl": train_fraud_model(data),
#         "segmentation_model.pkl": train_segmentation_model(data),
#         "engagement_model.pkl": train_engagement_model(data)
#     }

#     for model_name, model in models.items():
#         if model:
#             joblib.dump(model, model_dir / model_name)
#             logger.info(f"Saved {model_name}")

# if __name__ == "__main__":
#     main()





# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error
# from datetime import datetime, timedelta
# import requests
# from pathlib import Path
# import logging
# import json
# import shutil

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # API Client
# class APIClient:
#     def __init__(self):
#         self.base_url = "https://canada777.com/api"
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Basic canada777"
#         }
#         self.session = requests.Session()
#         retries = requests.adapters.HTTPAdapter(max_retries=3)
#         self.session.mount("https://", retries)
#         self.data_dir = Path("data")
#         self.data_dir.mkdir(parents=True, exist_ok=True)

#     def clear_data_dir(self):
#         try:
#             for file in self.data_dir.glob("*.json"):
#                 file.unlink()
#             logger.info("Cleared existing JSON files in data directory")
#         except Exception as e:
#             logger.error(f"Error clearing data directory: {str(e)}")

#     def save_data(self, endpoint, data):
#         try:
#             file_path = self.data_dir / f"{endpoint}.json"
#             with open(file_path, "w") as f:
#                 json.dump(data, f, indent=2)
#             logger.info(f"Saved data to data/{endpoint}.json")
#         except Exception as e:
#             logger.error(f"Error saving data for {endpoint}: {str(e)}")

#     def fetch_data(self, endpoint: str, created_at: str = None):
#         try:
#             url = f"{self.base_url}/{endpoint}"
#             params = {"createdAt": created_at} if created_at else {}
#             response = self.session.get(url, headers=self.headers, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if data.get("success"):
#                 return data["data"]["data"]
#             else:
#                 logger.error(f"Failed to fetch data from {endpoint} for {created_at}: {data.get('message')}")
#                 return []
#         except Exception as e:
#             logger.error(f"Error fetching data from {endpoint} for {created_at}: {str(e)}")
#             mock_file = self.data_dir / f"mock_{endpoint}_{created_at}.json" if created_at else self.data_dir / f"mock_{endpoint}.json"
#             if mock_file.exists():
#                 with open(mock_file, "r") as f:
#                     return json.load(f)["data"]["data"]
#             return []

#     def get_last_7_days_data(self, endpoint: str):
#         data = []
#         for i in range(7):
#             date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
#             logger.info(f"Fetching data for {endpoint} on {date}")
#             daily_data = self.fetch_data(endpoint, created_at=date)
#             logger.info(f"Fetched {len(daily_data)} records for {endpoint} on {date}")
#             data.extend(daily_data)
#         logger.info(f"Total fetched {len(data)} records for {endpoint} over last 7 days")
#         if data:
#             self.save_data(endpoint, {"success": True, "message": f"{endpoint} fetched successfully", "data": {"data": data}})
#         return data

# # Data Preprocessing
# def preprocess_churn_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         merged["last_login_at"] = pd.to_datetime(merged["last_login_at"])
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         # Fix: Group by user_id first, then compute recency
#         recency = merged.groupby("user_id")["last_login_at"].max().apply(
#             lambda x: (datetime.now() - timedelta(hours=6) - x).days if pd.notnull(x) else 30
#         )
#         frequency = merged.groupby("user_id")["action"].count()
#         monetary = merged.groupby("user_id")["amount"].sum()
#         churn_data = pd.DataFrame({
#             "user_id": recency.index,
#             "recency": recency.values,
#             "frequency": frequency.values,
#             "monetary": monetary.values
#         })
#         churn_data["churn_label"] = (churn_data["recency"] > 30).astype(int)
#         logger.info(f"Churn DataFrame shape: {churn_data.shape}")
#         return churn_data
#     except Exception as e:
#         logger.error(f"Error preprocessing churn data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_ltv_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         deposits = pd.DataFrame(data["deposits"])
#         bonuses = pd.DataFrame(data["bonuses"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Bonuses DataFrame columns: {bonuses.columns.tolist()}")
#         merged = players.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged players and deposits: {merged.shape}")
#         merged = merged.merge(bonuses, on="user_id", how="left")
#         logger.info(f"Merged with bonuses: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["bonus_amount"] = merged["bonus_amount"].astype(float).fillna(0)
#         ltv_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "bonus_amount": "sum",
#             "deposit_id": "count"
#         }).reset_index()
#         ltv_data.columns = ["user_id", "total_deposits", "total_bonuses", "deposit_count"]
#         ltv_data["ltv"] = ltv_data["total_deposits"] + ltv_data["total_bonuses"]
#         logger.info(f"LTV DataFrame shape: {ltv_data.shape}")
#         return ltv_data
#     except Exception as e:
#         logger.error(f"Error preprocessing LTV data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_fraud_data(data):
#     try:
#         deposits = pd.DataFrame(data["deposits"])
#         logs = pd.DataFrame(data["logs"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         merged = deposits.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged deposits and logs: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["win_amount"] = merged["win_amount"].astype(float).fillna(0)
#         fraud_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "win_amount": ["sum", "count"],
#             "ip": lambda x: x.nunique()
#         }).reset_index()
#         fraud_data.columns = ["user_id", "total_deposits", "total_wins", "win_count", "unique_ips"]
#         fraud_data["rapid_deposits"] = fraud_data["total_deposits"] / (fraud_data["win_count"] + 1)
#         fraud_data["fraud_label"] = (fraud_data["rapid_deposits"] > fraud_data["rapid_deposits"].quantile(0.95)).astype(int)
#         logger.info(f"Fraud DataFrame shape: {fraud_data.shape}")
#         return fraud_data
#     except Exception as e:
#         logger.error(f"Error preprocessing fraud data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_segmentation_data(data):
#     return preprocess_churn_data(data)

# def preprocess_engagement_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"])
#         engagement_data = merged.groupby("user_id").agg({
#             "action": "count",
#             "created_at": lambda x: (datetime.now() - timedelta(hours=6) - pd.to_datetime(x).max()).days if pd.notnull(x.max()) else 30,
#             "amount": "count"
#         }).reset_index()
#         engagement_data.columns = ["user_id", "activity_count", "recency", "deposit_count"]
#         engagement_data["engagement_label"] = (engagement_data["recency"] < 7).astype(int)
#         logger.info(f"Engagement DataFrame shape: {engagement_data.shape}")
#         return engagement_data
#     except Exception as e:
#         logger.error(f"Error preprocessing engagement data: {str(e)}")
#         return pd.DataFrame()

# # Training Functions
# def train_churn_model(data):
#     churn_data = preprocess_churn_data(data)
#     if churn_data.empty:
#         logger.error("No data for churn model training")
#         return None
#     X = churn_data[["recency", "frequency", "monetary"]].fillna(0)
#     y = churn_data["churn_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Churn model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_ltv_model(data):
#     ltv_data = preprocess_ltv_data(data)
#     if ltv_data.empty:
#         logger.error("No data for LTV model training")
#         return None
#     X = ltv_data[["total_deposits", "total_bonuses", "deposit_count"]].fillna(0)
#     y = ltv_data["ltv"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"LTV model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_fraud_model(data):
#     fraud_data = preprocess_fraud_data(data)
#     if fraud_data.empty:
#         logger.error("No data for fraud model training")
#         return None
#     X = fraud_data[["total_deposits", "total_wins", "rapid_deposits", "unique_ips"]].fillna(0)
#     model = IsolationForest(contamination=0.1, random_state=42)
#     model.fit(X)
#     logger.info("Fraud model trained")
#     return model

# def train_segmentation_model(data):
#     segmentation_data = preprocess_segmentation_data(data)
#     if segmentation_data.empty:
#         logger.error("No data for segmentation model training")
#         return None
#     X = segmentation_data[["recency", "frequency", "monetary"]].fillna(0)
#     model = KMeans(n_clusters=4, random_state=42)
#     model.fit(X)
#     logger.info("Segmentation model trained")
#     return model

# def train_engagement_model(data):
#     engagement_data = preprocess_engagement_data(data)
#     if engagement_data.empty:
#         logger.error("No data for engagement model training")
#         return None
#     X = engagement_data[["activity_count", "recency", "deposit_count"]].fillna(0)
#     y = engagement_data["engagement_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Engagement model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# # Main Training Function
# def main():
#     api_client = APIClient()
#     api_client.clear_data_dir()  # Clear existing JSON files
#     data = {
#         "players": api_client.get_last_7_days_data("players_details"),
#         "deposits": api_client.get_last_7_days_data("players_deposit_details"),
#         "bonuses": api_client.get_last_7_days_data("players_bonus_details"),
#         "logs": api_client.get_last_7_days_data("players_log_details")
#     }

#     model_dir = Path("data/models")
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Train and save models
#     models = {
#         "churn_model.pkl": train_churn_model(data),
#         "ltv_model.pkl": train_ltv_model(data),
#         "fraud_model.pkl": train_fraud_model(data),
#         "segmentation_model.pkl": train_segmentation_model(data),
#         "engagement_model.pkl": train_engagement_model(data)
#     }

#     for model_name, model in models.items():
#         if model:
#             joblib.dump(model, model_dir / model_name)
#             logger.info(f"Saved {model_name}")

# if __name__ == "__main__":
#     main()








# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error
# from datetime import datetime, timedelta
# import requests
# from pathlib import Path
# import logging
# import json
# import shutil
# import pytz  # Added for timezone handling

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # API Client
# class APIClient:
#     def __init__(self):
#         self.base_url = "https://canada777.com/api"
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Basic canada777"
#         }
#         self.session = requests.Session()
#         retries = requests.adapters.HTTPAdapter(max_retries=3)
#         self.session.mount("https://", retries)
#         self.data_dir = Path("data")
#         self.data_dir.mkdir(parents=True, exist_ok=True)

#     def clear_data_dir(self):
#         try:
#             for file in self.data_dir.glob("*.json"):
#                 file.unlink()
#             logger.info("Cleared existing JSON files in data directory")
#         except Exception as e:
#             logger.error(f"Error clearing data directory: {str(e)}")

#     def save_data(self, endpoint, data):
#         try:
#             file_path = self.data_dir / f"{endpoint}.json"
#             with open(file_path, "w") as f:
#                 json.dump(data, f, indent=2)
#             logger.info(f"Saved data to data/{endpoint}.json")
#         except Exception as e:
#             logger.error(f"Error saving data for {endpoint}: {str(e)}")

#     def fetch_data(self, endpoint: str, created_at: str = None):
#         try:
#             url = f"{self.base_url}/{endpoint}"
#             params = {"createdAt": created_at} if created_at else {}
#             response = self.session.get(url, headers=self.headers, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if data.get("success"):
#                 return data["data"]["data"]
#             else:
#                 logger.error(f"Failed to fetch data from {endpoint} for {created_at}: {data.get('message')}")
#                 return []
#         except Exception as e:
#             logger.error(f"Error fetching data from {endpoint} for {created_at}: {str(e)}")
#             mock_file = self.data_dir / f"mock_{endpoint}_{created_at}.json" if created_at else self.data_dir / f"mock_{endpoint}.json"
#             if mock_file.exists():
#                 with open(mock_file, "r") as f:
#                     return json.load(f)["data"]["data"]
#             return []

#     def get_last_7_days_data(self, endpoint: str):
#         data = []
#         for i in range(7):
#             date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
#             logger.info(f"Fetching data for {endpoint} on {date}")
#             daily_data = self.fetch_data(endpoint, created_at=date)
#             logger.info(f"Fetched {len(daily_data)} records for {endpoint} on {date}")
#             data.extend(daily_data)
#         logger.info(f"Total fetched {len(data)} records for {endpoint} over last 7 days")
#         if data:
#             self.save_data(endpoint, {"success": True, "message": f"{endpoint} fetched successfully", "data": {"data": data}})
#         return data

# # Data Preprocessing
# def preprocess_churn_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"], utc=True)
#         merged["last_login_at"] = pd.to_datetime(merged["last_login_at"], utc=True)
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         # Fix: Compute recency with timezone-aware datetime
#         utc_now = datetime.now(pytz.UTC) - timedelta(hours=6)
#         recency = merged.groupby("user_id")["last_login_at"].max().apply(
#             lambda x: (utc_now - x).days if pd.notnull(x) else 30
#         )
#         frequency = merged.groupby("user_id")["action"].count()
#         monetary = merged.groupby("user_id")["amount"].sum()
#         churn_data = pd.DataFrame({
#             "user_id": recency.index,
#             "recency": recency.values,
#             "frequency": frequency.values,
#             "monetary": monetary.values
#         })
#         churn_data["churn_label"] = (churn_data["recency"] > 30).astype(int)
#         logger.info(f"Churn DataFrame shape: {churn_data.shape}")
#         return churn_data
#     except Exception as e:
#         logger.error(f"Error preprocessing churn data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_ltv_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         deposits = pd.DataFrame(data["deposits"])
#         bonuses = pd.DataFrame(data["bonuses"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Bonuses DataFrame columns: {bonuses.columns.tolist()}")
#         merged = players.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged players and deposits: {merged.shape}")
#         merged = merged.merge(bonuses, on="user_id", how="left")
#         logger.info(f"Merged with bonuses: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["bonus_amount"] = merged["bonus_amount"].astype(float).fillna(0)
#         ltv_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "bonus_amount": "sum",
#             "deposit_id": "count"
#         }).reset_index()
#         ltv_data.columns = ["user_id", "total_deposits", "total_bonuses", "deposit_count"]
#         ltv_data["ltv"] = ltv_data["total_deposits"] + ltv_data["total_bonuses"]
#         logger.info(f"LTV DataFrame shape: {ltv_data.shape}")
#         return ltv_data
#     except Exception as e:
#         logger.error(f"Error preprocessing LTV data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_fraud_data(data):
#     try:
#         deposits = pd.DataFrame(data["deposits"])
#         logs = pd.DataFrame(data["logs"])
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         merged = deposits.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged deposits and logs: {merged.shape}")
#         merged["amount"] = merged["amount"].astype(float).fillna(0)
#         merged["win_amount"] = merged["win_amount"].astype(float).fillna(0)
#         fraud_data = merged.groupby("user_id").agg({
#             "amount": "sum",
#             "win_amount": ["sum", "count"],
#             "ip": lambda x: x.nunique()
#         }).reset_index()
#         fraud_data.columns = ["user_id", "total_deposits", "total_wins", "win_count", "unique_ips"]
#         fraud_data["rapid_deposits"] = fraud_data["total_deposits"] / (fraud_data["win_count"] + 1)
#         fraud_data["fraud_label"] = (fraud_data["rapid_deposits"] > fraud_data["rapid_deposits"].quantile(0.95)).astype(int)
#         logger.info(f"Fraud DataFrame shape: {fraud_data.shape}")
#         return fraud_data
#     except Exception as e:
#         logger.error(f"Error preprocessing fraud data: {str(e)}")
#         return pd.DataFrame()

# def preprocess_segmentation_data(data):
#     return preprocess_churn_data(data)

# def preprocess_engagement_data(data):
#     try:
#         players = pd.DataFrame(data["players"])
#         logger.info(f"Players DataFrame columns: {players.columns.tolist()}")
#         if 'id' in players.columns and 'user_id' not in players.columns:
#             players = players.rename(columns={'id': 'user_id'})
#             logger.info("Renamed 'id' to 'user_id' in players DataFrame")
#         logs = pd.DataFrame(data["logs"])
#         deposits = pd.DataFrame(data["deposits"])
#         logger.info(f"Logs DataFrame columns: {logs.columns.tolist()}")
#         logger.info(f"Deposits DataFrame columns: {deposits.columns.tolist()}")
#         merged = players.merge(logs, on="user_id", how="left")
#         logger.info(f"Merged players and logs: {merged.shape}")
#         merged = merged.merge(deposits, on="user_id", how="left")
#         logger.info(f"Merged with deposits: {merged.shape}")
#         merged["created_at"] = pd.to_datetime(merged["created_at"], utc=True)
#         # Fix: Compute recency with timezone-aware datetime
#         utc_now = datetime.now(pytz.UTC) - timedelta(hours=6)
#         engagement_data = merged.groupby("user_id").agg({
#             "action": "count",
#             "created_at": lambda x: (utc_now - pd.to_datetime(x).max()).days if pd.notnull(x.max()) else 30,
#             "amount": "count"
#         }).reset_index()
#         engagement_data.columns = ["user_id", "activity_count", "recency", "deposit_count"]
#         engagement_data["engagement_label"] = (engagement_data["recency"] < 7).astype(int)
#         logger.info(f"Engagement DataFrame shape: {engagement_data.shape}")
#         return engagement_data
#     except Exception as e:
#         logger.error(f"Error preprocessing engagement data: {str(e)}")
#         return pd.DataFrame()

# # Training Functions
# def train_churn_model(data):
#     churn_data = preprocess_churn_data(data)
#     if churn_data.empty:
#         logger.error("No data for churn model training")
#         return None
#     X = churn_data[["recency", "frequency", "monetary"]].fillna(0)
#     y = churn_data["churn_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Churn model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_ltv_model(data):
#     ltv_data = preprocess_ltv_data(data)
#     if ltv_data.empty:
#         logger.error("No data for LTV model training")
#         return None
#     X = ltv_data[["total_deposits", "total_bonuses", "deposit_count"]].fillna(0)
#     y = ltv_data["ltv"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"LTV model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
#     return model

# def train_fraud_model(data):
#     fraud_data = preprocess_fraud_data(data)
#     if fraud_data.empty:
#         logger.error("No data for fraud model training")
#         return None
#     X = fraud_data[["total_deposits", "total_wins", "rapid_deposits", "unique_ips"]].fillna(0)
#     model = IsolationForest(contamination=0.1, random_state=42)
#     model.fit(X)
#     logger.info("Fraud model trained")
#     return model

# def train_segmentation_model(data):
#     segmentation_data = preprocess_segmentation_data(data)
#     if segmentation_data.empty:
#         logger.error("No data for segmentation model training")
#         return None
#     X = segmentation_data[["recency", "frequency", "monetary"]].fillna(0)
#     model = KMeans(n_clusters=4, random_state=42)
#     model.fit(X)
#     logger.info("Segmentation model trained")
#     return model

# def train_engagement_model(data):
#     engagement_data = preprocess_engagement_data(data)
#     if engagement_data.empty:
#         logger.error("No data for engagement model training")
#         return None
#     X = engagement_data[["activity_count", "recency", "deposit_count"]].fillna(0)
#     y = engagement_data["engagement_label"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)
#     logger.info(f"Engagement model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
#     return model

# # Main Training Function
# def main():
#     api_client = APIClient()
#     api_client.clear_data_dir()  # Clear existing JSON files
#     data = {
#         "players": api_client.get_last_7_days_data("players_details"),
#         "deposits": api_client.get_last_7_days_data("players_deposit_details"),
#         "bonuses": api_client.get_last_7_days_data("players_bonus_details"),
#         "logs": api_client.get_last_7_days_data("players_log_details")
#     }

#     model_dir = Path("data/models")
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Train and save models
#     models = {
#         "churn_model.pkl": train_churn_model(data),
#         "ltv_model.pkl": train_ltv_model(data),
#         "fraud_model.pkl": train_fraud_model(data),
#         "segmentation_model.pkl": train_segmentation_model(data),
#         "engagement_model.pkl": train_engagement_model(data)
#     }

#     for model_name, model in models.items():
#         if model:
#             joblib.dump(model, model_dir / model_name)
#             logger.info(f"Saved {model_name}")

# if __name__ == "__main__":
#     main()








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






# from fastapi import APIRouter, HTTPException
# from datetime import datetime
# from app.services.data_service import DataService
# from app.services.ml_service import MLService
# import logging
# import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# router = APIRouter()

# def convert_numpy_types(obj):
#     """Convert numpy types to native Python types for JSON serialization"""
#     if isinstance(obj, (np.integer, np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {key: convert_numpy_types(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_types(item) for item in obj]
#     return obj

# @router.get("/predict")
# async def predict_ltv():
#     try:
#         data_service = DataService()
#         ml_service = MLService()
        
#         # Fetch and preprocess data
#         raw_data = data_service.fetch_all_data()
#         ltv_data, extra_metrics = data_service.preprocess_ltv_data(raw_data)
        
#         if ltv_data.empty:
#             # Return mock data when no real data is available
#             predictions = [
#                 {"user_id": int(i), "predicted_ltv": float(500.0 + i * 50)}
#                 for i in range(1, 11)
#             ]
#         else:
#             # Make predictions
#             raw_predictions = ml_service.predict_ltv(ltv_data)
#             # Convert numpy types to native Python types
#             predictions = convert_numpy_types(raw_predictions)
        
#         # Ensure all numeric values are converted to Python native types
#         processed_predictions = []
#         for pred in predictions:
#             processed_pred = {
#                 "user_id": int(pred["user_id"]) if not isinstance(pred["user_id"], int) else pred["user_id"],
#                 "predicted_ltv": float(pred["predicted_ltv"]) if not isinstance(pred["predicted_ltv"], float) else pred["predicted_ltv"]
#             }
#             processed_predictions.append(processed_pred)
        
#         predictions = processed_predictions
        
#         # Calculate averages and ranges with proper type conversion
#         total_predictions = len(predictions)
#         if total_predictions > 0:
#             average_ltv = float(sum(p["predicted_ltv"] for p in predictions) / total_predictions)
#             min_ltv = float(min(p["predicted_ltv"] for p in predictions))
#             max_ltv = float(max(p["predicted_ltv"] for p in predictions))
            
#             # Calculate segment breakdowns
#             low_value_count = sum(1 for p in predictions if p["predicted_ltv"] < 500)
#             medium_value_count = sum(1 for p in predictions if 500 <= p["predicted_ltv"] < 1000)
#             high_value_count = sum(1 for p in predictions if p["predicted_ltv"] >= 1000)
            
#             low_value_pct = float(low_value_count / total_predictions * 100)
#             medium_value_pct = float(medium_value_count / total_predictions * 100)
#             high_value_pct = float(high_value_count / total_predictions * 100)
#         else:
#             average_ltv = 672.08
#             min_ltv = 315.67
#             max_ltv = 1200.45
#             low_value_pct = 33.33
#             medium_value_pct = 33.33
#             high_value_pct = 33.33
        
#         # Construct response with proper type conversion
#         response = {
#             "status": "success",
#             "api_version": "1.0.0",
#             "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+06"),
#             "data": {
#                 "prediction_type": "ltv_prediction",
#                 "results": [
#                     {
#                         "user_id": int(pred["user_id"]),
#                         "predicted_ltv": float(pred["predicted_ltv"]),
#                         "prediction_confidence": 0.85 if pred["predicted_ltv"] < 1000 else 0.92,
#                         "churn_adjusted_ltv": float(pred["predicted_ltv"] * 0.8),
#                         "ltv_confidence_interval": {
#                             "min": float(pred["predicted_ltv"] * 0.9), 
#                             "max": float(pred["predicted_ltv"] * 1.1)
#                         },
#                         "user_preferences": {
#                             "favorite_game": data_service.get_preferred_game(int(pred["user_id"]), raw_data),
#                             "preferred_payment_method": data_service.get_preferred_payment_method(int(pred["user_id"]), raw_data)
#                         },
#                         "cross_sell_opportunity": "Offer premium membership" if pred["predicted_ltv"] < 1000 else "Promote live tournaments"
#                     } for pred in predictions
#                 ],
#                 "average_ltv": average_ltv,
#                 "mae": 50.67,
#                 "ltv_range": {
#                     "min": min_ltv,
#                     "max": max_ltv
#                 },
#                 "total_users": int(total_predictions),
#                 "ltv_forecast": convert_numpy_types(extra_metrics.get("ltv_forecast", {"predicted_ltv_next_month": 800.50})),
#                 "ltv_segment_breakdown": {
#                     "low_value": low_value_pct,
#                     "medium_value": medium_value_pct,
#                     "high_value": high_value_pct
#                 },
#                 "top_high_value_users": [
#                     {
#                         "user_id": int(p["user_id"]), 
#                         "predicted_ltv": float(p["predicted_ltv"])
#                     }
#                     for p in predictions if p["predicted_ltv"] >= 1000
#                 ],
#                 "ltv_growth_potential": [
#                     {
#                         "user_id": int(p["user_id"]), 
#                         "potential_ltv_increase": float(p["predicted_ltv"] * 0.2), 
#                         "action": "Upsell VIP package"
#                     }
#                     for p in predictions
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
        
#         # Final conversion to ensure all numpy types are converted
#         response = convert_numpy_types(response)
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in LTV prediction endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
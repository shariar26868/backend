
import requests
from datetime import datetime, timedelta
from pathlib import Path
from app.config.settings import settings
import logging  # Changed to import logging directly
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Define logger locally

class APIClient:
    def __init__(self):
        self.base_url = settings.API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": settings.API_AUTH
        }
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.data_dir = settings.DATA_DIR
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
            logger.info(f"Saved data to {file_path}")
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
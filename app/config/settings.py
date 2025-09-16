from pydantic import BaseModel, Field
from pathlib import Path

class Settings(BaseModel):
    # API Configuration
    API_URL: str = Field(default="https://canada777.com/api", env="API_URL")
    API_AUTH: str = Field(default="Basic canada777", env="API_AUTH")
    
    # Directory Configuration
    DATA_DIR: Path = Field(default=Path("data"), env="DATA_DIR")
    MODEL_DIR: Path = Field(default=Path("data/models"), env="MODEL_DIR")
    
    # Churn Prediction Configuration
    CHURN_THRESHOLD: float = Field(default=0.5, env="CHURN_THRESHOLD")
    CHURN_HIGH_CONFIDENCE: float = Field(default=0.95, env="CHURN_HIGH_CONFIDENCE")
    CHURN_LOW_CONFIDENCE: float = Field(default=0.90, env="CHURN_LOW_CONFIDENCE")
    
    # Model Configuration
    MODEL_VERSION: str = Field(default="v1.3", env="MODEL_VERSION")
    
    # Business Rules
    HIGH_VALUE_THRESHOLD: float = Field(default=1000.0, env="HIGH_VALUE_THRESHOLD")
    VIP_THRESHOLD: float = Field(default=5000.0, env="VIP_THRESHOLD")
    
    # Response Configuration
    DEFAULT_PAGE_SIZE: int = Field(default=50, env="DEFAULT_PAGE_SIZE")
    MAX_PAGE_SIZE: int = Field(default=100, env="MAX_PAGE_SIZE")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()

from pydantic import BaseModel, Field
from pathlib import Path

class Settings(BaseModel):
    API_URL: str = Field(default="https://canada777.com/api", env="API_URL")
    API_AUTH: str = Field(default="Basic canada777", env="API_AUTH")
    DATA_DIR: Path = Field(default=Path("data"), env="DATA_DIR")
    MODEL_DIR: Path = Field(default=Path("data/models"), env="MODEL_DIR")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()
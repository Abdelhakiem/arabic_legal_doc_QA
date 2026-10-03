from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # Data / artifacts
    data_path: Path = Path("data/green_tripdata_2024-01.parquet")
    model_path: Path = Path("models/baseline.pkl")

    # Training
    train_size: float = 0.8
    min_duration: float = 0.0
    max_duration: float = 120.0
    min_trip_distance: float = 0.0

    # Model
    n_estimators: int = 15
    random_state: int = 42

    # API
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()

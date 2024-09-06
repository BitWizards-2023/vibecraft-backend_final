# app/core/config.py

from pydantic_settings import BaseSettings  # Updated import

class Settings(BaseSettings):
    app_name: str = "Typing Analysis API"
    environment: str = "development"
    port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
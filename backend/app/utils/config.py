from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True
    
    # Database
    database_url: str = "sqlite:///./dengue_ai.db"
    
    # External APIs
    openweathermap_api_key: Optional[str] = None
    google_translate_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = "AIzaSyA43gTE9O9A0yyX24PFkWzyKYooYr8WU88"
    
    # Security
    secret_key: str = "dengue-ai-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # AI Models
    ai_model_path: str = "./models/"
    
    # Gamification
    points_per_verified_report: int = 50
    points_per_photo: int = 30
    points_per_location: int = 20
    
    class Config:
        env_file = ".env"

settings = Settings()

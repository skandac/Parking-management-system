"""
Configuration settings for the AI Parking Management System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Parking Management System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"\]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/parking_db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # ML Models
    YOLO_MODEL_PATH: str = "ml_models/yolov5_parking.pt"
    KNN_MODEL_PATH: str = "ml_models/knn_classifier.pkl"
    HAAR_CASCADE_PATH: str = "ml_models/haarcascade_frontalface_default.xml"
    
    # Computer Vision
    CAMERA_FPS: int = 30
    PROCESSING_INTERVAL: float = 0.1  # seconds
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Storage
    UPLOAD_DIR: str = "uploads/"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # External APIs
    PAYMENT_API_KEY: Optional[str] = None
    NAVIGATION_API_KEY: Optional[str] = None
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

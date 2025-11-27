"""
Configuration management for the cyberattack detection system.
"""
import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017/cyberattack_db", env="MONGODB_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    postgres_url: Optional[str] = Field(default=None, env="POSTGRES_URL")
    
    # API Keys for Threat Intelligence
    virustotal_api_key: Optional[str] = Field(default=None, env="VIRUSTOTAL_API_KEY")
    google_safe_browsing_api_key: Optional[str] = Field(default=None, env="GOOGLE_SAFE_BROWSING_API_KEY")
    abuseipdb_api_key: Optional[str] = Field(default=None, env="ABUSEIPDB_API_KEY")
    shodan_api_key: Optional[str] = Field(default=None, env="SHODAN_API_KEY")
    maxmind_license_key: Optional[str] = Field(default=None, env="MAXMIND_LICENSE_KEY")
    
    # Model Configuration
    model_update_interval: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")
    threat_intelligence_update_interval: int = Field(default=1800, env="THREAT_INTELLIGENCE_UPDATE_INTERVAL")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    learning_rate: float = Field(default=0.001, env="LEARNING_RATE")
    max_epochs: int = Field(default=100, env="MAX_EPOCHS")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    encryption_key: str = Field(default="your-encryption-key-change-in-production", env="ENCRYPTION_KEY")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/cyberattack_detection.log", env="LOG_FILE")
    max_log_size: int = Field(default=10485760, env="MAX_LOG_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Alerting Configuration
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    webhook_url: Optional[str] = Field(default=None, env="WEBHOOK_URL")
    
    # Performance Configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    queue_size: int = Field(default=1000, env="QUEUE_SIZE")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    rate_limit: int = Field(default=1000, env="RATE_LIMIT")
    
    # Feature Flags
    enable_real_time_analysis: bool = Field(default=True, env="ENABLE_REAL_TIME_ANALYSIS")
    enable_threat_intelligence: bool = Field(default=True, env="ENABLE_THREAT_INTELLIGENCE")
    enable_anomaly_detection: bool = Field(default=True, env="ENABLE_ANOMALY_DETECTION")
    enable_explainability: bool = Field(default=True, env="ENABLE_EXPLAINABILITY")
    enable_continuous_learning: bool = Field(default=True, env="ENABLE_CONTINUOUS_LEARNING")
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    metrics_update_interval: int = Field(default=60, env="METRICS_UPDATE_INTERVAL")
    
    # Attack Type Configuration
    attack_types: List[str] = Field(default=[
        "phishing", "malware", "ransomware", "dos", "ddos", 
        "r2l", "u2r", "probe", "spam", "botnet"
    ])
    
    # Model Paths
    model_dir: str = Field(default="data/models", env="MODEL_DIR")
    data_dir: str = Field(default="data", env="DATA_DIR")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Dashboard Configuration
    dashboard_host: str = Field(default="0.0.0.0", env="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8501, env="DASHBOARD_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


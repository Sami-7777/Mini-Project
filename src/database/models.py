"""
Data models for the cyberattack detection system.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from pymongo import IndexModel, ASCENDING, DESCENDING
from bson import ObjectId


class AttackType(str, Enum):
    """Enumeration of supported attack types."""
    PHISHING = "phishing"
    MALWARE = "malware"
    RANSOMWARE = "ransomware"
    DOS = "dos"
    DDOS = "ddos"
    R2L = "r2l"  # Remote to Local
    U2R = "u2r"  # User to Root
    PROBE = "probe"
    SPAM = "spam"
    BOTNET = "botnet"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """Enumeration of severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    """Enumeration of analysis status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class BaseDocument(BaseModel):
    """Base document model with common fields."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class URLFeatures(BaseModel):
    """URL feature extraction model."""
    
    # Lexical features
    url_length: int
    domain_length: int
    path_length: int
    query_length: int
    fragment_length: int
    
    # Character analysis
    digit_count: int
    letter_count: int
    special_char_count: int
    entropy: float
    
    # Domain analysis
    subdomain_count: int
    tld: str
    domain_age_days: Optional[int] = None
    
    # Keyword patterns
    suspicious_keywords: List[str] = Field(default_factory=list)
    brand_keywords: List[str] = Field(default_factory=list)
    
    # URL structure
    has_ip_address: bool
    has_shortener: bool
    has_redirect: bool
    port_number: Optional[int] = None
    
    # Semantic features
    url_similarity_score: Optional[float] = None
    brand_similarity_score: Optional[float] = None


class IPFeatures(BaseModel):
    """IP address feature extraction model."""
    
    # Basic IP info
    ip_address: str
    is_private: bool
    is_reserved: bool
    
    # Geolocation
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    
    # Network analysis
    asn: Optional[int] = None
    asn_organization: Optional[str] = None
    isp: Optional[str] = None
    
    # Reputation
    reputation_score: Optional[float] = None
    abuse_score: Optional[float] = None
    threat_types: List[str] = Field(default_factory=list)
    
    # Behavioral patterns
    request_frequency: Optional[float] = None
    unique_user_agents: Optional[int] = None
    common_ports: List[int] = Field(default_factory=list)


class ThreatIntelligence(BaseModel):
    """Threat intelligence data model."""
    
    source: str  # virustotal, google_safe_browsing, abuseipdb, etc.
    source_id: str
    threat_type: str
    confidence: float
    last_updated: datetime
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class ModelPrediction(BaseModel):
    """ML model prediction result."""
    
    model_name: str
    attack_type: AttackType
    confidence: float
    probability_scores: Dict[str, float] = Field(default_factory=dict)
    features_used: List[str] = Field(default_factory=list)
    prediction_time: datetime = Field(default_factory=datetime.utcnow)


class ExplainabilityResult(BaseModel):
    """Model explainability results."""
    
    method: str  # SHAP, LIME, etc.
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    explanation_text: Optional[str] = None
    visualization_data: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseDocument):
    """Main analysis result document."""
    
    # Target information
    target_type: str  # "url" or "ip"
    target_value: str  # The actual URL or IP
    
    # Analysis metadata
    status: AnalysisStatus = AnalysisStatus.PENDING
    analysis_duration_ms: Optional[int] = None
    
    # Feature data
    url_features: Optional[URLFeatures] = None
    ip_features: Optional[IPFeatures] = None
    
    # Threat intelligence
    threat_intelligence: List[ThreatIntelligence] = Field(default_factory=list)
    
    # ML predictions
    predictions: List[ModelPrediction] = Field(default_factory=list)
    
    # Final assessment
    final_attack_type: Optional[AttackType] = None
    final_confidence: Optional[float] = None
    severity: Optional[SeverityLevel] = None
    risk_score: Optional[float] = None
    
    # Explainability
    explainability: Optional[ExplainabilityResult] = None
    
    # Additional metadata
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    session_id: Optional[str] = None
    
    # Feedback and learning
    user_feedback: Optional[str] = None  # "correct", "incorrect", "partial"
    feedback_timestamp: Optional[datetime] = None
    
    @validator('target_type')
    def validate_target_type(cls, v):
        if v not in ['url', 'ip']:
            raise ValueError('target_type must be either "url" or "ip"')
        return v


class Alert(BaseDocument):
    """Alert model for security notifications."""
    
    analysis_id: PyObjectId
    alert_type: str  # "threat_detected", "anomaly_detected", "system_alert"
    severity: SeverityLevel
    title: str
    description: str
    attack_type: Optional[AttackType] = None
    
    # Alert metadata
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Notification status
    email_sent: bool = False
    sms_sent: bool = False
    webhook_sent: bool = False
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelMetrics(BaseDocument):
    """Model performance metrics."""
    
    model_name: str
    model_version: str
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Training metadata
    training_samples: int
    validation_samples: int
    training_duration_seconds: int
    
    # Model metadata
    feature_count: int
    model_size_mb: float
    
    # Timestamps
    training_start: datetime
    training_end: datetime
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemMetrics(BaseDocument):
    """System performance and health metrics."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance metrics
    requests_per_second: float
    average_response_time_ms: float
    error_rate: float
    
    # Resource usage
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    
    # Model performance
    model_accuracy: Dict[str, float] = Field(default_factory=dict)
    prediction_latency_ms: Dict[str, float] = Field(default_factory=dict)
    
    # Threat statistics
    threats_detected: int
    false_positives: int
    attack_type_distribution: Dict[str, int] = Field(default_factory=dict)


# MongoDB indexes for optimal performance
ANALYSIS_INDEXES = [
    IndexModel([("target_type", ASCENDING), ("created_at", DESCENDING)]),
    IndexModel([("status", ASCENDING)]),
    IndexModel([("final_attack_type", ASCENDING)]),
    IndexModel([("severity", ASCENDING)]),
    IndexModel([("created_at", DESCENDING)]),
    IndexModel([("target_value", ASCENDING)]),
]

ALERT_INDEXES = [
    IndexModel([("severity", ASCENDING), ("created_at", DESCENDING)]),
    IndexModel([("is_acknowledged", ASCENDING)]),
    IndexModel([("alert_type", ASCENDING)]),
]

MODEL_METRICS_INDEXES = [
    IndexModel([("model_name", ASCENDING), ("evaluation_timestamp", DESCENDING)]),
    IndexModel([("training_end", DESCENDING)]),
]

SYSTEM_METRICS_INDEXES = [
    IndexModel([("timestamp", DESCENDING)]),
]


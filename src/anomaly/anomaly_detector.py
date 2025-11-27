"""
Anomaly detection system for cyberattack detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import structlog
from dataclasses import dataclass
from enum import Enum

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..features.feature_engine import feature_engine

logger = structlog.get_logger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies."""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    NETWORK = "network"
    CONTENT = "content"


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    anomaly_type: AnomalyType
    target: str
    target_type: str
    anomaly_score: float
    severity: SeverityLevel
    description: str
    features: Dict[str, Any]
    timestamp: datetime
    confidence: float = 0.0


class AnomalyDetector:
    """Main anomaly detection system."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.is_fitted = False
        self.normal_data = None
        self.feature_names = []
    
    async def detect_anomalies(self, target: str, target_type: str, 
                              features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect anomalies in the given target."""
        try:
            anomalies = []
            
            # Statistical anomaly detection
            statistical_anomalies = await self._detect_statistical_anomalies(
                target, target_type, features
            )
            anomalies.extend(statistical_anomalies)
            
            # Behavioral anomaly detection
            behavioral_anomalies = await self._detect_behavioral_anomalies(
                target, target_type, features
            )
            anomalies.extend(behavioral_anomalies)
            
            # Temporal anomaly detection
            temporal_anomalies = await self._detect_temporal_anomalies(
                target, target_type, features
            )
            anomalies.extend(temporal_anomalies)
            
            # Network anomaly detection
            network_anomalies = await self._detect_network_anomalies(
                target, target_type, features
            )
            anomalies.extend(network_anomalies)
            
            # Content anomaly detection
            content_anomalies = await self._detect_content_anomalies(
                target, target_type, features
            )
            anomalies.extend(content_anomalies)
            
            logger.info("Anomaly detection completed", 
                       target=target, 
                       anomalies_found=len(anomalies))
            
            return anomalies
            
        except Exception as e:
            logger.error("Error detecting anomalies", target=target, error=str(e))
            return []
    
    async def _detect_statistical_anomalies(self, target: str, target_type: str,
                                           features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect statistical anomalies using Isolation Forest."""
        anomalies = []
        
        try:
            # Prepare features for statistical analysis
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is None or len(feature_vector) == 0:
                return anomalies
            
            # Fit models if not already fitted
            if not self.is_fitted:
                await self._fit_statistical_models()
            
            # Detect anomalies
            anomaly_score = self.isolation_forest.decision_function([feature_vector])[0]
            is_anomaly = self.isolation_forest.predict([feature_vector])[0] == -1
            
            if is_anomaly:
                severity = self._calculate_severity(abs(anomaly_score))
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.STATISTICAL,
                    target=target,
                    target_type=target_type,
                    anomaly_score=abs(anomaly_score),
                    severity=severity,
                    description=f"Statistical anomaly detected with score {anomaly_score:.3f}",
                    features=features,
                    timestamp=datetime.utcnow(),
                    confidence=min(abs(anomaly_score), 1.0)
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            logger.error("Error in statistical anomaly detection", error=str(e))
        
        return anomalies
    
    async def _detect_behavioral_anomalies(self, target: str, target_type: str,
                                          features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect behavioral anomalies."""
        anomalies = []
        
        try:
            # Analyze behavioral patterns
            behavioral_features = features.get('behavioral', {})
            temporal_features = features.get('temporal', {})
            
            # Check for unusual request patterns
            if 'request_frequency' in behavioral_features:
                freq = behavioral_features['request_frequency']
                if freq and freq > 1000:  # More than 1000 requests per minute
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.BEHAVIORAL,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.8,
                        severity=SeverityLevel.HIGH,
                        description=f"Unusually high request frequency: {freq} requests/min",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.8
                    )
                    anomalies.append(anomaly)
            
            # Check for burst patterns
            if 'burst_ratio' in temporal_features:
                burst_ratio = temporal_features['burst_ratio']
                if burst_ratio and burst_ratio > 0.8:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.BEHAVIORAL,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.7,
                        severity=SeverityLevel.MEDIUM,
                        description=f"High burst activity detected: {burst_ratio:.2%} burst ratio",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.7
                    )
                    anomalies.append(anomaly)
            
            # Check for irregular intervals
            if 'irregular_intervals' in temporal_features:
                if temporal_features['irregular_intervals']:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.BEHAVIORAL,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.6,
                        severity=SeverityLevel.MEDIUM,
                        description="Irregular request intervals detected",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.6
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error("Error in behavioral anomaly detection", error=str(e))
        
        return anomalies
    
    async def _detect_temporal_anomalies(self, target: str, target_type: str,
                                        features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect temporal anomalies."""
        anomalies = []
        
        try:
            temporal_features = features.get('temporal', {})
            
            # Check for temporal anomaly score
            if 'temporal_anomaly_score' in temporal_features:
                score = temporal_features['temporal_anomaly_score']
                if score and score > 3.0:  # Z-score > 3
                    severity = SeverityLevel.HIGH if score > 5.0 else SeverityLevel.MEDIUM
                    
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.TEMPORAL,
                        target=target,
                        target_type=target_type,
                        anomaly_score=score / 10.0,  # Normalize to 0-1
                        severity=severity,
                        description=f"Temporal anomaly detected with Z-score {score:.2f}",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=min(score / 10.0, 1.0)
                    )
                    anomalies.append(anomaly)
            
            # Check for unusual time patterns
            if 'peak_hour' in temporal_features:
                peak_hour = temporal_features['peak_hour']
                current_hour = datetime.utcnow().hour
                
                # Check if peak hour is very different from current hour
                if abs(peak_hour - current_hour) > 6:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.TEMPORAL,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.5,
                        severity=SeverityLevel.LOW,
                        description=f"Unusual timing: peak hour {peak_hour}, current hour {current_hour}",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.5
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error("Error in temporal anomaly detection", error=str(e))
        
        return anomalies
    
    async def _detect_network_anomalies(self, target: str, target_type: str,
                                       features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect network-related anomalies."""
        anomalies = []
        
        try:
            if target_type == "ip":
                ip_features = features.get('ip', {})
                
                # Check for high abuse score
                if 'abuse_score' in ip_features:
                    abuse_score = ip_features['abuse_score']
                    if abuse_score and abuse_score > 0.8:
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.NETWORK,
                            target=target,
                            target_type=target_type,
                            anomaly_score=abuse_score,
                            severity=SeverityLevel.HIGH,
                            description=f"High abuse score detected: {abuse_score:.2%}",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=abuse_score
                        )
                        anomalies.append(anomaly)
                
                # Check for suspicious country
                if 'country' in ip_features:
                    country = ip_features['country']
                    high_risk_countries = ['CN', 'RU', 'KP', 'IR', 'SY']
                    if country in high_risk_countries:
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.NETWORK,
                            target=target,
                            target_type=target_type,
                            anomaly_score=0.6,
                            severity=SeverityLevel.MEDIUM,
                            description=f"IP from high-risk country: {country}",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=0.6
                        )
                        anomalies.append(anomaly)
            
            elif target_type == "url":
                url_features = features.get('url', {})
                
                # Check for IP address in URL
                if 'has_ip_address' in url_features and url_features['has_ip_address']:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.NETWORK,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.7,
                        severity=SeverityLevel.MEDIUM,
                        description="URL contains IP address instead of domain",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.7
                    )
                    anomalies.append(anomaly)
                
                # Check for URL shortener
                if 'has_shortener' in url_features and url_features['has_shortener']:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.NETWORK,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.5,
                        severity=SeverityLevel.LOW,
                        description="URL uses shortening service",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.5
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error("Error in network anomaly detection", error=str(e))
        
        return anomalies
    
    async def _detect_content_anomalies(self, target: str, target_type: str,
                                       features: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect content-related anomalies."""
        anomalies = []
        
        try:
            if target_type == "url":
                url_features = features.get('url', {})
                
                # Check for high entropy
                if 'entropy' in url_features:
                    entropy = url_features['entropy']
                    if entropy and entropy > 6.0:
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.CONTENT,
                            target=target,
                            target_type=target_type,
                            anomaly_score=min(entropy / 10.0, 1.0),
                            severity=SeverityLevel.MEDIUM,
                            description=f"High URL entropy detected: {entropy:.2f}",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=min(entropy / 10.0, 1.0)
                        )
                        anomalies.append(anomaly)
                
                # Check for suspicious keywords
                if 'suspicious_keywords' in url_features:
                    keywords = url_features['suspicious_keywords']
                    if keywords and len(keywords) > 3:
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.CONTENT,
                            target=target,
                            target_type=target_type,
                            anomaly_score=min(len(keywords) / 10.0, 1.0),
                            severity=SeverityLevel.HIGH,
                            description=f"Multiple suspicious keywords detected: {len(keywords)}",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=min(len(keywords) / 10.0, 1.0)
                        )
                        anomalies.append(anomaly)
                
                # Check for polymorphism
                polymorphism_features = features.get('polymorphism', {})
                if 'has_encoding' in polymorphism_features and polymorphism_features['has_encoding']:
                    anomaly = AnomalyResult(
                        anomaly_type=AnomalyType.CONTENT,
                        target=target,
                        target_type=target_type,
                        anomaly_score=0.8,
                        severity=SeverityLevel.HIGH,
                        description="URL encoding/polymorphism detected",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.8
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error("Error in content anomaly detection", error=str(e))
        
        return anomalies
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for statistical analysis."""
        try:
            # Extract numeric features
            numeric_features = []
            
            # URL features
            url_features = features.get('url', {})
            if url_features:
                numeric_features.extend([
                    url_features.get('url_length', 0),
                    url_features.get('domain_length', 0),
                    url_features.get('entropy', 0),
                    url_features.get('digit_count', 0),
                    url_features.get('letter_count', 0),
                    url_features.get('special_char_count', 0)
                ])
            
            # IP features
            ip_features = features.get('ip', {})
            if ip_features:
                numeric_features.extend([
                    ip_features.get('reputation_score', 0) or 0,
                    ip_features.get('abuse_score', 0) or 0,
                    ip_features.get('request_frequency', 0) or 0
                ])
            
            # Temporal features
            temporal_features = features.get('temporal', {})
            if temporal_features:
                numeric_features.extend([
                    temporal_features.get('total_events', 0),
                    temporal_features.get('mean_interval_seconds', 0),
                    temporal_features.get('burst_ratio', 0),
                    temporal_features.get('temporal_anomaly_score', 0)
                ])
            
            # Composite features
            composite_features = features.get('composite', {})
            if composite_features:
                numeric_features.extend([
                    composite_features.get('risk_score', 0),
                    composite_features.get('suspicion_score', 0),
                    composite_features.get('anomaly_score', 0)
                ])
            
            if numeric_features:
                return np.array(numeric_features)
            
            return None
            
        except Exception as e:
            logger.error("Error preparing feature vector", error=str(e))
            return None
    
    async def _fit_statistical_models(self):
        """Fit statistical models with historical data."""
        try:
            # Get historical data for training
            historical_data = await self._get_historical_data()
            
            if historical_data and len(historical_data) > 10:
                # Prepare feature vectors
                feature_vectors = []
                for data in historical_data:
                    vector = self._prepare_feature_vector(data)
                    if vector is not None:
                        feature_vectors.append(vector)
                
                if feature_vectors:
                    feature_matrix = np.array(feature_vectors)
                    
                    # Scale features
                    feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
                    
                    # Fit Isolation Forest
                    self.isolation_forest.fit(feature_matrix_scaled)
                    
                    # Fit DBSCAN
                    self.dbscan.fit(feature_matrix_scaled)
                    
                    self.is_fitted = True
                    self.normal_data = feature_matrix_scaled
                    
                    logger.info("Statistical models fitted successfully", 
                               samples=len(feature_vectors))
            
        except Exception as e:
            logger.error("Error fitting statistical models", error=str(e))
    
    async def _get_historical_data(self) -> List[Dict[str, Any]]:
        """Get historical data for model training."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            
            # Get recent analyses
            cursor = collection.find({
                "status": "completed",
                "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)}
            }).limit(1000)
            
            historical_data = []
            async for doc in cursor:
                # Reconstruct features from analysis result
                features = {}
                if doc.get('url_features'):
                    features['url'] = doc['url_features']
                if doc.get('ip_features'):
                    features['ip'] = doc['ip_features']
                if doc.get('temporal_features'):
                    features['temporal'] = doc['temporal_features']
                if doc.get('composite_features'):
                    features['composite'] = doc['composite_features']
                
                historical_data.append(features)
            
            return historical_data
            
        except Exception as e:
            logger.error("Error getting historical data", error=str(e))
            return []
    
    def _calculate_severity(self, anomaly_score: float) -> SeverityLevel:
        """Calculate severity based on anomaly score."""
        if anomaly_score > 0.8:
            return SeverityLevel.CRITICAL
        elif anomaly_score > 0.6:
            return SeverityLevel.HIGH
        elif anomaly_score > 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        try:
            db = await get_database()
            collection = db.get_collection("anomaly_results")
            
            # Get anomaly statistics
            pipeline = [
                {"$group": {
                    "_id": "$anomaly_type",
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$anomaly_score"},
                    "max_score": {"$max": "$anomaly_score"}
                }}
            ]
            
            stats = {}
            async for doc in collection.aggregate(pipeline):
                stats[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_score": doc["avg_score"],
                    "max_score": doc["max_score"]
                }
            
            return {
                "total_anomalies": await collection.count_documents({}),
                "anomaly_types": stats,
                "model_fitted": self.is_fitted
            }
            
        except Exception as e:
            logger.error("Error getting anomaly statistics", error=str(e))
            return {}


# Global anomaly detector instance
anomaly_detector = AnomalyDetector()

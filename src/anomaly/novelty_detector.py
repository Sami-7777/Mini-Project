"""
Novelty detection system for identifying previously unseen attack patterns.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import structlog
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..features.feature_engine import feature_engine

logger = structlog.get_logger(__name__)


class NoveltyType(str, Enum):
    """Types of novelty detection."""
    FEATURE_NOVELTY = "feature_novelty"
    PATTERN_NOVELTY = "pattern_novelty"
    BEHAVIOR_NOVELTY = "behavior_novelty"
    CONTENT_NOVELTY = "content_novelty"
    TEMPORAL_NOVELTY = "temporal_novelty"


@dataclass
class NoveltyResult:
    """Result from novelty detection."""
    novelty_type: NoveltyType
    target: str
    target_type: str
    novelty_score: float
    severity: SeverityLevel
    description: str
    features: Dict[str, Any]
    timestamp: datetime
    confidence: float = 0.0
    pattern_hash: str = ""


class NoveltyDetector:
    """Novelty detection system for identifying unknown attack patterns."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.05,  # Lower contamination for novelty detection
            random_state=42,
            n_estimators=200
        )
        self.local_outlier_factor = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05
        )
        self.dbscan = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.is_fitted = False
        self.known_patterns = set()
        self.pattern_embeddings = {}
        self.feature_names = []
    
    async def detect_novelty(self, target: str, target_type: str, 
                            features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novelty in the given target."""
        try:
            novelties = []
            
            # Feature novelty detection
            feature_novelties = await self._detect_feature_novelty(
                target, target_type, features
            )
            novelties.extend(feature_novelties)
            
            # Pattern novelty detection
            pattern_novelties = await self._detect_pattern_novelty(
                target, target_type, features
            )
            novelties.extend(pattern_novelties)
            
            # Behavior novelty detection
            behavior_novelties = await self._detect_behavior_novelty(
                target, target_type, features
            )
            novelties.extend(behavior_novelties)
            
            # Content novelty detection
            content_novelties = await self._detect_content_novelty(
                target, target_type, features
            )
            novelties.extend(content_novelties)
            
            # Temporal novelty detection
            temporal_novelties = await self._detect_temporal_novelty(
                target, target_type, features
            )
            novelties.extend(temporal_novelties)
            
            logger.info("Novelty detection completed", 
                       target=target, 
                       novelties_found=len(novelties))
            
            return novelties
            
        except Exception as e:
            logger.error("Error detecting novelty", target=target, error=str(e))
            return []
    
    async def _detect_feature_novelty(self, target: str, target_type: str,
                                     features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novelty in feature combinations."""
        novelties = []
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is None or len(feature_vector) == 0:
                return novelties
            
            # Fit models if not already fitted
            if not self.is_fitted:
                await self._fit_novelty_models()
            
            # Detect novelty using Isolation Forest
            novelty_score = self.isolation_forest.decision_function([feature_vector])[0]
            is_novel = self.isolation_forest.predict([feature_vector])[0] == -1
            
            # Also check with Local Outlier Factor
            lof_score = self.local_outlier_factor.decision_function([feature_vector])[0]
            is_lof_novel = self.local_outlier_factor.predict([feature_vector])[0] == -1
            
            # Combine scores
            combined_score = (abs(novelty_score) + abs(lof_score)) / 2
            is_combined_novel = is_novel or is_lof_novel
            
            if is_combined_novel:
                severity = self._calculate_severity(combined_score)
                
                novelty = NoveltyResult(
                    novelty_type=NoveltyType.FEATURE_NOVELTY,
                    target=target,
                    target_type=target_type,
                    novelty_score=combined_score,
                    severity=severity,
                    description=f"Novel feature combination detected with score {combined_score:.3f}",
                    features=features,
                    timestamp=datetime.utcnow(),
                    confidence=min(combined_score, 1.0),
                    pattern_hash=self._generate_pattern_hash(features)
                )
                novelties.append(novelty)
        
        except Exception as e:
            logger.error("Error in feature novelty detection", error=str(e))
        
        return novelties
    
    async def _detect_pattern_novelty(self, target: str, target_type: str,
                                     features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novel patterns in the data."""
        novelties = []
        
        try:
            # Generate pattern hash
            pattern_hash = self._generate_pattern_hash(features)
            
            # Check if this pattern has been seen before
            if pattern_hash not in self.known_patterns:
                # This is a novel pattern
                novelty_score = 0.8  # High score for completely new patterns
                
                novelty = NoveltyResult(
                    novelty_type=NoveltyType.PATTERN_NOVELTY,
                    target=target,
                    target_type=target_type,
                    novelty_score=novelty_score,
                    severity=SeverityLevel.HIGH,
                    description="Completely novel pattern detected",
                    features=features,
                    timestamp=datetime.utcnow(),
                    confidence=novelty_score,
                    pattern_hash=pattern_hash
                )
                novelties.append(novelty)
                
                # Add to known patterns
                self.known_patterns.add(pattern_hash)
                
                # Store pattern for future reference
                await self._store_novel_pattern(pattern_hash, features)
        
        except Exception as e:
            logger.error("Error in pattern novelty detection", error=str(e))
        
        return novelties
    
    async def _detect_behavior_novelty(self, target: str, target_type: str,
                                      features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novel behavioral patterns."""
        novelties = []
        
        try:
            behavioral_features = features.get('behavioral', {})
            temporal_features = features.get('temporal', {})
            
            # Check for novel request patterns
            if 'request_frequency' in behavioral_features:
                freq = behavioral_features['request_frequency']
                if freq and freq > 0:
                    # Check if this frequency is novel
                    is_novel_freq = await self._is_novel_frequency(freq)
                    
                    if is_novel_freq:
                        novelty = NoveltyResult(
                            novelty_type=NoveltyType.BEHAVIOR_NOVELTY,
                            target=target,
                            target_type=target_type,
                            novelty_score=0.7,
                            severity=SeverityLevel.MEDIUM,
                            description=f"Novel request frequency pattern: {freq} requests/min",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=0.7,
                            pattern_hash=self._generate_pattern_hash(features)
                        )
                        novelties.append(novelty)
            
            # Check for novel temporal patterns
            if 'peak_hour' in temporal_features:
                peak_hour = temporal_features['peak_hour']
                is_novel_timing = await self._is_novel_timing(peak_hour)
                
                if is_novel_timing:
                    novelty = NoveltyResult(
                        novelty_type=NoveltyType.BEHAVIOR_NOVELTY,
                        target=target,
                        target_type=target_type,
                        novelty_score=0.6,
                        severity=SeverityLevel.MEDIUM,
                        description=f"Novel timing pattern: peak hour {peak_hour}",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.6,
                        pattern_hash=self._generate_pattern_hash(features)
                    )
                    novelties.append(novelty)
        
        except Exception as e:
            logger.error("Error in behavior novelty detection", error=str(e))
        
        return novelties
    
    async def _detect_content_novelty(self, target: str, target_type: str,
                                     features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novel content patterns."""
        novelties = []
        
        try:
            if target_type == "url":
                url_features = features.get('url', {})
                
                # Check for novel URL structures
                url_structure = self._extract_url_structure(url_features)
                is_novel_structure = await self._is_novel_url_structure(url_structure)
                
                if is_novel_structure:
                    novelty = NoveltyResult(
                        novelty_type=NoveltyType.CONTENT_NOVELTY,
                        target=target,
                        target_type=target_type,
                        novelty_score=0.7,
                        severity=SeverityLevel.MEDIUM,
                        description="Novel URL structure detected",
                        features=features,
                        timestamp=datetime.utcnow(),
                        confidence=0.7,
                        pattern_hash=self._generate_pattern_hash(features)
                    )
                    novelties.append(novelty)
                
                # Check for novel keyword combinations
                keywords = url_features.get('suspicious_keywords', [])
                if keywords:
                    keyword_hash = hashlib.md5('|'.join(sorted(keywords)).encode()).hexdigest()
                    is_novel_keywords = await self._is_novel_keyword_combination(keyword_hash)
                    
                    if is_novel_keywords:
                        novelty = NoveltyResult(
                            novelty_type=NoveltyType.CONTENT_NOVELTY,
                            target=target,
                            target_type=target_type,
                            novelty_score=0.8,
                            severity=SeverityLevel.HIGH,
                            description=f"Novel keyword combination: {len(keywords)} keywords",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=0.8,
                            pattern_hash=self._generate_pattern_hash(features)
                        )
                        novelties.append(novelty)
        
        except Exception as e:
            logger.error("Error in content novelty detection", error=str(e))
        
        return novelties
    
    async def _detect_temporal_novelty(self, target: str, target_type: str,
                                      features: Dict[str, Any]) -> List[NoveltyResult]:
        """Detect novel temporal patterns."""
        novelties = []
        
        try:
            temporal_features = features.get('temporal', {})
            
            # Check for novel interval patterns
            if 'mean_interval_seconds' in temporal_features:
                mean_interval = temporal_features['mean_interval_seconds']
                if mean_interval and mean_interval > 0:
                    is_novel_interval = await self._is_novel_interval(mean_interval)
                    
                    if is_novel_interval:
                        novelty = NoveltyResult(
                            novelty_type=NoveltyType.TEMPORAL_NOVELTY,
                            target=target,
                            target_type=target_type,
                            novelty_score=0.6,
                            severity=SeverityLevel.MEDIUM,
                            description=f"Novel interval pattern: {mean_interval:.2f}s mean interval",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=0.6,
                            pattern_hash=self._generate_pattern_hash(features)
                        )
                        novelties.append(novelty)
            
            # Check for novel burst patterns
            if 'burst_ratio' in temporal_features:
                burst_ratio = temporal_features['burst_ratio']
                if burst_ratio and burst_ratio > 0:
                    is_novel_burst = await self._is_novel_burst_pattern(burst_ratio)
                    
                    if is_novel_burst:
                        novelty = NoveltyResult(
                            novelty_type=NoveltyType.TEMPORAL_NOVELTY,
                            target=target,
                            target_type=target_type,
                            novelty_score=0.7,
                            severity=SeverityLevel.MEDIUM,
                            description=f"Novel burst pattern: {burst_ratio:.2%} burst ratio",
                            features=features,
                            timestamp=datetime.utcnow(),
                            confidence=0.7,
                            pattern_hash=self._generate_pattern_hash(features)
                        )
                        novelties.append(novelty)
        
        except Exception as e:
            logger.error("Error in temporal novelty detection", error=str(e))
        
        return novelties
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for novelty detection."""
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
                    url_features.get('special_char_count', 0),
                    url_features.get('subdomain_count', 0)
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
                    temporal_features.get('temporal_anomaly_score', 0),
                    temporal_features.get('peak_hour', 0)
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
    
    def _generate_pattern_hash(self, features: Dict[str, Any]) -> str:
        """Generate a hash for the pattern."""
        try:
            # Create a simplified representation of the features
            pattern_data = {
                'url_length': features.get('url', {}).get('url_length', 0),
                'entropy': features.get('url', {}).get('entropy', 0),
                'abuse_score': features.get('ip', {}).get('abuse_score', 0) or 0,
                'risk_score': features.get('composite', {}).get('risk_score', 0),
                'peak_hour': features.get('temporal', {}).get('peak_hour', 0)
            }
            
            # Generate hash
            pattern_str = json.dumps(pattern_data, sort_keys=True)
            return hashlib.md5(pattern_str.encode()).hexdigest()
            
        except Exception as e:
            logger.error("Error generating pattern hash", error=str(e))
            return ""
    
    async def _fit_novelty_models(self):
        """Fit novelty detection models with historical data."""
        try:
            # Get historical data for training
            historical_data = await self._get_historical_data()
            
            if historical_data and len(historical_data) > 20:
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
                    
                    # Fit Local Outlier Factor
                    self.local_outlier_factor.fit(feature_matrix_scaled)
                    
                    # Fit DBSCAN
                    self.dbscan.fit(feature_matrix_scaled)
                    
                    self.is_fitted = True
                    
                    logger.info("Novelty detection models fitted successfully", 
                               samples=len(feature_vectors))
            
        except Exception as e:
            logger.error("Error fitting novelty models", error=str(e))
    
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
    
    async def _store_novel_pattern(self, pattern_hash: str, features: Dict[str, Any]):
        """Store a novel pattern for future reference."""
        try:
            db = await get_database()
            collection = db.get_collection("novel_patterns")
            
            pattern_doc = {
                "pattern_hash": pattern_hash,
                "features": features,
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "occurrence_count": 1
            }
            
            await collection.insert_one(pattern_doc)
            
        except Exception as e:
            logger.error("Error storing novel pattern", error=str(e))
    
    def _extract_url_structure(self, url_features: Dict[str, Any]) -> str:
        """Extract URL structure pattern."""
        try:
            structure = {
                'has_ip': url_features.get('has_ip_address', False),
                'has_shortener': url_features.get('has_shortener', False),
                'has_redirect': url_features.get('has_redirect', False),
                'subdomain_count': url_features.get('subdomain_count', 0),
                'tld': url_features.get('tld', '')
            }
            
            return json.dumps(structure, sort_keys=True)
            
        except Exception as e:
            logger.error("Error extracting URL structure", error=str(e))
            return ""
    
    async def _is_novel_frequency(self, frequency: float) -> bool:
        """Check if frequency is novel."""
        # Placeholder implementation
        # In practice, you would check against historical frequency patterns
        return frequency > 1000 or frequency < 0.1
    
    async def _is_novel_timing(self, peak_hour: int) -> bool:
        """Check if timing is novel."""
        # Placeholder implementation
        # In practice, you would check against historical timing patterns
        return peak_hour < 6 or peak_hour > 22
    
    async def _is_novel_url_structure(self, structure: str) -> bool:
        """Check if URL structure is novel."""
        # Placeholder implementation
        # In practice, you would check against known URL structures
        return len(structure) > 100
    
    async def _is_novel_keyword_combination(self, keyword_hash: str) -> bool:
        """Check if keyword combination is novel."""
        # Placeholder implementation
        # In practice, you would check against known keyword combinations
        return keyword_hash not in self.known_patterns
    
    async def _is_novel_interval(self, interval: float) -> bool:
        """Check if interval is novel."""
        # Placeholder implementation
        # In practice, you would check against historical interval patterns
        return interval > 3600 or interval < 0.1
    
    async def _is_novel_burst_pattern(self, burst_ratio: float) -> bool:
        """Check if burst pattern is novel."""
        # Placeholder implementation
        # In practice, you would check against historical burst patterns
        return burst_ratio > 0.9 or burst_ratio < 0.1
    
    def _calculate_severity(self, novelty_score: float) -> SeverityLevel:
        """Calculate severity based on novelty score."""
        if novelty_score > 0.8:
            return SeverityLevel.CRITICAL
        elif novelty_score > 0.6:
            return SeverityLevel.HIGH
        elif novelty_score > 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def get_novelty_statistics(self) -> Dict[str, Any]:
        """Get novelty detection statistics."""
        try:
            db = await get_database()
            collection = db.get_collection("novelty_results")
            
            # Get novelty statistics
            pipeline = [
                {"$group": {
                    "_id": "$novelty_type",
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$novelty_score"},
                    "max_score": {"$max": "$novelty_score"}
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
                "total_novelties": await collection.count_documents({}),
                "novelty_types": stats,
                "known_patterns": len(self.known_patterns),
                "model_fitted": self.is_fitted
            }
            
        except Exception as e:
            logger.error("Error getting novelty statistics", error=str(e))
            return {}


# Global novelty detector instance
novelty_detector = NoveltyDetector()

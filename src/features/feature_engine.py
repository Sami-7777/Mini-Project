"""
Main feature engineering engine that orchestrates all feature extraction.
"""
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np
import pandas as pd
import structlog

from ..database.models import URLFeatures, IPFeatures, AnalysisResult
from .url_features import url_feature_extractor, polymorphism_detector
from .ip_features import ip_feature_extractor, ip_rotation_detector
from .temporal_features import temporal_feature_extractor, access_pattern_analyzer

logger = structlog.get_logger(__name__)


class FeatureEngine:
    """Main feature engineering engine for cyberattack detection."""
    
    def __init__(self):
        self.url_extractor = url_feature_extractor
        self.ip_extractor = ip_feature_extractor
        self.temporal_extractor = temporal_feature_extractor
        self.polymorphism_detector = polymorphism_detector
        self.rotation_detector = ip_rotation_detector
        self.pattern_analyzer = access_pattern_analyzer
    
    async def extract_all_features(self, 
                                 target: str, 
                                 target_type: str,
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract all features for a given target (URL or IP)."""
        try:
            features = {
                'target': target,
                'target_type': target_type,
                'extraction_timestamp': datetime.utcnow(),
                'features': {}
            }
            
            if target_type == 'url':
                features['features'].update(await self._extract_url_features(target, context))
            elif target_type == 'ip':
                features['features'].update(await self._extract_ip_features(target, context))
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
            
            # Add temporal features if context provides timestamps
            if context and 'timestamps' in context:
                temporal_features = self.temporal_extractor.extract_temporal_features(
                    context['timestamps']
                )
                features['features']['temporal'] = temporal_features
            
            # Add contextual features
            if context:
                features['features']['contextual'] = self._extract_contextual_features(context)
            
            # Calculate composite features
            features['features']['composite'] = self._calculate_composite_features(
                features['features']
            )
            
            return features
            
        except Exception as e:
            logger.error("Error extracting features", target=target, error=str(e))
            return self._get_empty_features(target, target_type)
    
    async def _extract_url_features(self, url: str, context: Optional[Dict] = None) -> Dict:
        """Extract URL-specific features."""
        features = {}
        
        # Basic URL features
        url_features = self.url_extractor.extract_features(url)
        features['url'] = url_features.dict()
        
        # Polymorphism detection
        polymorphism_features = self.polymorphism_detector.detect_polymorphism(url)
        features['polymorphism'] = polymorphism_features
        
        # Additional URL analysis
        features['url_advanced'] = await self._extract_advanced_url_features(url, context)
        
        return features
    
    async def _extract_ip_features(self, ip_address: str, context: Optional[Dict] = None) -> Dict:
        """Extract IP-specific features."""
        features = {}
        
        # Basic IP features
        ip_features = await self.ip_extractor.extract_features(ip_address)
        features['ip'] = ip_features.dict()
        
        # IP rotation detection (if history provided)
        if context and 'ip_history' in context:
            rotation_features = self.rotation_detector.detect_rotation(context['ip_history'])
            features['rotation'] = rotation_features
        
        # Additional IP analysis
        features['ip_advanced'] = await self._extract_advanced_ip_features(ip_address, context)
        
        return features
    
    async def _extract_advanced_url_features(self, url: str, context: Optional[Dict] = None) -> Dict:
        """Extract advanced URL features."""
        features = {}
        
        try:
            # URL complexity analysis
            features['complexity'] = self._analyze_url_complexity(url)
            
            # Domain reputation analysis
            features['domain_reputation'] = await self._analyze_domain_reputation(url)
            
            # Content analysis (if available)
            if context and 'content' in context:
                features['content_analysis'] = self._analyze_url_content(context['content'])
            
            # Referrer analysis
            if context and 'referrer' in context:
                features['referrer_analysis'] = self._analyze_referrer(context['referrer'])
            
        except Exception as e:
            logger.warning("Error extracting advanced URL features", url=url, error=str(e))
        
        return features
    
    async def _extract_advanced_ip_features(self, ip_address: str, context: Optional[Dict] = None) -> Dict:
        """Extract advanced IP features."""
        features = {}
        
        try:
            # IP reputation analysis
            features['reputation_analysis'] = await self._analyze_ip_reputation(ip_address)
            
            # Network topology analysis
            features['network_topology'] = self._analyze_network_topology(ip_address)
            
            # Behavioral analysis
            if context and 'behavioral_data' in context:
                features['behavioral_analysis'] = self._analyze_ip_behavior(
                    ip_address, context['behavioral_data']
                )
            
        except Exception as e:
            logger.warning("Error extracting advanced IP features", ip=ip_address, error=str(e))
        
        return features
    
    def _extract_contextual_features(self, context: Dict) -> Dict:
        """Extract contextual features from request context."""
        features = {}
        
        # User agent analysis
        if 'user_agent' in context:
            features['user_agent'] = self._analyze_user_agent(context['user_agent'])
        
        # Session analysis
        if 'session_id' in context:
            features['session'] = self._analyze_session(context.get('session_data', {}))
        
        # Geographic context
        if 'client_ip' in context:
            features['geographic_context'] = self._analyze_geographic_context(
                context['client_ip']
            )
        
        return features
    
    def _calculate_composite_features(self, features: Dict) -> Dict:
        """Calculate composite features from extracted features."""
        composite = {}
        
        try:
            # Risk score calculation
            composite['risk_score'] = self._calculate_risk_score(features)
            
            # Suspicion score
            composite['suspicion_score'] = self._calculate_suspicion_score(features)
            
            # Anomaly score
            composite['anomaly_score'] = self._calculate_anomaly_score(features)
            
            # Threat level
            composite['threat_level'] = self._calculate_threat_level(composite)
            
        except Exception as e:
            logger.warning("Error calculating composite features", error=str(e))
            composite = {
                'risk_score': 0.0,
                'suspicion_score': 0.0,
                'anomaly_score': 0.0,
                'threat_level': 'low'
            }
        
        return composite
    
    def _analyze_url_complexity(self, url: str) -> Dict:
        """Analyze URL complexity."""
        return {
            'total_length': len(url),
            'path_depth': url.count('/'),
            'parameter_count': url.count('&') + (1 if '?' in url else 0),
            'encoding_ratio': url.count('%') / len(url) if url else 0,
            'special_char_ratio': sum(1 for c in url if not c.isalnum()) / len(url) if url else 0
        }
    
    async def _analyze_domain_reputation(self, url: str) -> Dict:
        """Analyze domain reputation."""
        # Placeholder for domain reputation analysis
        return {
            'reputation_score': 0.5,
            'trust_level': 'unknown',
            'blacklist_status': 'unknown'
        }
    
    def _analyze_url_content(self, content: str) -> Dict:
        """Analyze URL content."""
        return {
            'content_length': len(content),
            'suspicious_keywords': [],  # Would implement keyword detection
            'language': 'unknown',  # Would implement language detection
            'content_type': 'unknown'
        }
    
    def _analyze_referrer(self, referrer: str) -> Dict:
        """Analyze referrer information."""
        return {
            'has_referrer': bool(referrer),
            'referrer_domain': referrer.split('/')[2] if referrer and '/' in referrer else None,
            'is_direct': not bool(referrer)
        }
    
    async def _analyze_ip_reputation(self, ip_address: str) -> Dict:
        """Analyze IP reputation."""
        # Placeholder for IP reputation analysis
        return {
            'reputation_score': 0.5,
            'threat_level': 'unknown',
            'last_seen': None
        }
    
    def _analyze_network_topology(self, ip_address: str) -> Dict:
        """Analyze network topology."""
        return {
            'network_class': 'unknown',
            'routing_info': {},
            'neighbor_analysis': {}
        }
    
    def _analyze_ip_behavior(self, ip_address: str, behavioral_data: Dict) -> Dict:
        """Analyze IP behavioral patterns."""
        return {
            'request_patterns': {},
            'geographic_movement': {},
            'temporal_patterns': {}
        }
    
    def _analyze_user_agent(self, user_agent: str) -> Dict:
        """Analyze user agent string."""
        return {
            'browser': 'unknown',
            'os': 'unknown',
            'device_type': 'unknown',
            'is_bot': False,
            'suspicious_patterns': []
        }
    
    def _analyze_session(self, session_data: Dict) -> Dict:
        """Analyze session information."""
        return {
            'session_duration': 0,
            'request_count': 0,
            'unique_endpoints': 0,
            'suspicious_activity': False
        }
    
    def _analyze_geographic_context(self, client_ip: str) -> Dict:
        """Analyze geographic context."""
        return {
            'country': 'unknown',
            'region': 'unknown',
            'is_vpn': False,
            'is_proxy': False
        }
    
    def _calculate_risk_score(self, features: Dict) -> float:
        """Calculate overall risk score."""
        risk_factors = []
        
        # URL-based risk factors
        if 'url' in features:
            url_features = features['url']
            if url_features.get('has_ip_address', False):
                risk_factors.append(0.3)
            if url_features.get('has_shortener', False):
                risk_factors.append(0.2)
            if url_features.get('entropy', 0) > 4.0:
                risk_factors.append(0.2)
        
        # IP-based risk factors
        if 'ip' in features:
            ip_features = features['ip']
            if ip_features.get('is_private', False):
                risk_factors.append(0.1)
            if ip_features.get('abuse_score', 0) > 0.5:
                risk_factors.append(0.4)
        
        # Polymorphism risk
        if 'polymorphism' in features:
            poly_features = features['polymorphism']
            if poly_features.get('has_encoding', False):
                risk_factors.append(0.3)
        
        # Temporal risk
        if 'temporal' in features:
            temporal_features = features['temporal']
            if temporal_features.get('burst_ratio', 0) > 0.5:
                risk_factors.append(0.2)
        
        # Calculate weighted risk score
        if risk_factors:
            return min(1.0, sum(risk_factors))
        return 0.0
    
    def _calculate_suspicion_score(self, features: Dict) -> float:
        """Calculate suspicion score."""
        suspicion_factors = []
        
        # High entropy URLs
        if 'url' in features and features['url'].get('entropy', 0) > 5.0:
            suspicion_factors.append(0.3)
        
        # Suspicious keywords
        if 'url' in features and features['url'].get('suspicious_keywords'):
            suspicion_factors.append(0.4)
        
        # High abuse score IPs
        if 'ip' in features and features['ip'].get('abuse_score', 0) > 0.7:
            suspicion_factors.append(0.5)
        
        return min(1.0, sum(suspicion_factors))
    
    def _calculate_anomaly_score(self, features: Dict) -> float:
        """Calculate anomaly score."""
        anomaly_factors = []
        
        # Temporal anomalies
        if 'temporal' in features:
            temporal_features = features['temporal']
            if temporal_features.get('temporal_anomaly_score', 0) > 2.0:
                anomaly_factors.append(0.3)
        
        # Unusual patterns
        if 'temporal' in features:
            temporal_features = features['temporal']
            if temporal_features.get('irregular_intervals', False):
                anomaly_factors.append(0.2)
        
        return min(1.0, sum(anomaly_factors))
    
    def _calculate_threat_level(self, composite_features: Dict) -> str:
        """Calculate threat level based on composite features."""
        risk_score = composite_features.get('risk_score', 0)
        suspicion_score = composite_features.get('suspicion_score', 0)
        anomaly_score = composite_features.get('anomaly_score', 0)
        
        total_score = (risk_score + suspicion_score + anomaly_score) / 3
        
        if total_score >= 0.8:
            return 'critical'
        elif total_score >= 0.6:
            return 'high'
        elif total_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_empty_features(self, target: str, target_type: str) -> Dict:
        """Return empty features structure."""
        return {
            'target': target,
            'target_type': target_type,
            'extraction_timestamp': datetime.utcnow(),
            'features': {
                'composite': {
                    'risk_score': 0.0,
                    'suspicion_score': 0.0,
                    'anomaly_score': 0.0,
                    'threat_level': 'low'
                }
            }
        }


# Global feature engine instance
feature_engine = FeatureEngine()


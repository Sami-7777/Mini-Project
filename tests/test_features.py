"""
Tests for feature engineering.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.features.url_features import URLFeatureExtractor, URLPolymorphismDetector
from src.features.ip_features import IPFeatureExtractor, IPRotationDetector
from src.features.temporal_features import TemporalFeatureExtractor, AccessPatternAnalyzer
from src.features.feature_engine import FeatureEngine


class TestURLFeatures:
    """Test URL feature extraction."""
    
    @pytest.fixture
    def url_extractor(self):
        """Create URL feature extractor."""
        return URLFeatureExtractor()
    
    def test_extract_basic_url_features(self, url_extractor):
        """Test basic URL feature extraction."""
        url = "https://example.com/path?param=value#fragment"
        features = url_extractor.extract_features(url)
        
        assert features.url_length == len(url)
        assert features.domain_length == len("example.com")
        assert features.path_length == len("/path")
        assert features.query_length == len("param=value")
        assert features.fragment_length == len("fragment")
    
    def test_extract_character_features(self, url_extractor):
        """Test character feature extraction."""
        url = "https://example123.com/path"
        features = url_extractor.extract_features(url)
        
        assert features.digit_count == 3
        assert features.letter_count > 0
        assert features.special_char_count > 0
        assert features.entropy > 0
    
    def test_extract_domain_features(self, url_extractor):
        """Test domain feature extraction."""
        url = "https://sub.example.com/path"
        features = url_extractor.extract_features(url)
        
        assert features.subdomain_count == 1
        assert features.tld == "com"
    
    def test_extract_keyword_features(self, url_extractor):
        """Test keyword feature extraction."""
        url = "https://secure-login-bank.com/verify"
        features = url_extractor.extract_features(url)
        
        assert len(features.suspicious_keywords) > 0
        assert any("secure" in keyword for keyword in features.suspicious_keywords)
    
    def test_extract_structure_features(self, url_extractor):
        """Test URL structure feature extraction."""
        # Test IP address in URL
        ip_url = "https://192.168.1.1/path"
        ip_features = url_extractor.extract_features(ip_url)
        assert ip_features.has_ip_address
        
        # Test URL shortener
        shortener_url = "https://bit.ly/abc123"
        shortener_features = url_extractor.extract_features(shortener_url)
        assert shortener_features.has_shortener
    
    def test_polymorphism_detection(self):
        """Test URL polymorphism detection."""
        detector = URLPolymorphismDetector()
        
        # Test encoded URL
        encoded_url = "https://example.com/path%20with%20spaces"
        result = detector.detect_polymorphism(encoded_url)
        
        assert result['has_encoding']
        assert len(result['encoding_types']) > 0
    
    def test_minimal_features_on_error(self, url_extractor):
        """Test minimal features returned on error."""
        # This would test error handling
        # Implementation depends on specific error scenarios
        pass


class TestIPFeatures:
    """Test IP feature extraction."""
    
    @pytest.fixture
    def ip_extractor(self):
        """Create IP feature extractor."""
        return IPFeatureExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_basic_ip_features(self, ip_extractor):
        """Test basic IP feature extraction."""
        ip = "192.168.1.1"
        features = await ip_extractor.extract_features(ip)
        
        assert features.ip_address == ip
        assert features.is_private
        assert not features.is_reserved
    
    @pytest.mark.asyncio
    async def test_extract_public_ip_features(self, ip_extractor):
        """Test public IP feature extraction."""
        ip = "8.8.8.8"
        features = await ip_extractor.extract_features(ip)
        
        assert features.ip_address == ip
        assert not features.is_private
        assert not features.is_reserved
    
    @pytest.mark.asyncio
    async def test_extract_geolocation_features(self, ip_extractor):
        """Test geolocation feature extraction."""
        ip = "8.8.8.8"
        features = await ip_extractor.extract_features(ip)
        
        # These would be populated if GeoIP database is available
        assert hasattr(features, 'country')
        assert hasattr(features, 'latitude')
        assert hasattr(features, 'longitude')
    
    def test_ip_rotation_detection(self):
        """Test IP rotation detection."""
        detector = IPRotationDetector()
        
        # Test with rotating IPs
        ip_history = ["192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.1"]
        result = detector.detect_rotation(ip_history)
        
        assert result['rotation_detected']
        assert result['unique_ips'] == 3
        assert result['rotation_frequency'] > 0.5
    
    def test_no_rotation_detection(self):
        """Test no rotation detection."""
        detector = IPRotationDetector()
        
        # Test with same IP
        ip_history = ["192.168.1.1", "192.168.1.1", "192.168.1.1"]
        result = detector.detect_rotation(ip_history)
        
        assert not result['rotation_detected']
        assert result['unique_ips'] == 1
        assert result['rotation_frequency'] == 0.0


class TestTemporalFeatures:
    """Test temporal feature extraction."""
    
    @pytest.fixture
    def temporal_extractor(self):
        """Create temporal feature extractor."""
        return TemporalFeatureExtractor()
    
    def test_extract_basic_temporal_features(self, temporal_extractor):
        """Test basic temporal feature extraction."""
        from datetime import datetime, timedelta
        
        # Create sample timestamps
        base_time = datetime.utcnow()
        timestamps = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=5)
        ]
        
        features = temporal_extractor.extract_temporal_features(timestamps)
        
        assert features['total_events'] == 4
        assert features['time_span_hours'] > 0
        assert features['mean_interval_seconds'] > 0
    
    def test_extract_time_patterns(self, temporal_extractor):
        """Test time pattern extraction."""
        from datetime import datetime, timedelta
        
        # Create timestamps at specific hours
        base_date = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        timestamps = [
            base_date,
            base_date + timedelta(hours=1),
            base_date + timedelta(hours=2)
        ]
        
        features = temporal_extractor.extract_temporal_features(timestamps)
        
        assert features['peak_hour'] == 9
        assert features['hour_entropy'] >= 0
    
    def test_extract_frequency_features(self, temporal_extractor):
        """Test frequency feature extraction."""
        from datetime import datetime, timedelta
        
        # Create burst pattern
        base_time = datetime.utcnow()
        timestamps = []
        
        # Create burst of requests
        for i in range(10):
            timestamps.append(base_time + timedelta(seconds=i))
        
        # Add some spaced out requests
        for i in range(5):
            timestamps.append(base_time + timedelta(minutes=i+1))
        
        features = temporal_extractor.extract_temporal_features(timestamps)
        
        assert features['burst_count'] > 0
        assert features['max_burst_size'] > 0
        assert features['burst_ratio'] > 0
    
    def test_extract_temporal_anomalies(self, temporal_extractor):
        """Test temporal anomaly extraction."""
        from datetime import datetime, timedelta
        
        # Create irregular intervals
        base_time = datetime.utcnow()
        timestamps = [
            base_time,
            base_time + timedelta(seconds=1),
            base_time + timedelta(hours=1),  # Large gap
            base_time + timedelta(hours=1, seconds=1)
        ]
        
        features = temporal_extractor.extract_temporal_features(timestamps)
        
        assert features['temporal_anomaly_score'] > 0
        assert features['irregular_intervals']
    
    def test_access_pattern_analysis(self):
        """Test access pattern analysis."""
        analyzer = AccessPatternAnalyzer()
        
        from datetime import datetime, timedelta
        
        # Create human-like pattern
        base_time = datetime.utcnow()
        timestamps = [
            base_time,
            base_time + timedelta(seconds=30),
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2)
        ]
        
        result = analyzer.analyze_access_pattern(timestamps)
        
        assert result['pattern_type'] in ['human_like', 'bot_like', 'sparse']
        assert 'confidence' in result
        assert 'mean_interval' in result


class TestFeatureEngine:
    """Test feature engine."""
    
    @pytest.fixture
    def feature_engine(self):
        """Create feature engine."""
        return FeatureEngine()
    
    @pytest.mark.asyncio
    async def test_extract_url_features(self, feature_engine):
        """Test URL feature extraction."""
        url = "https://example.com/path"
        context = {"user_agent": "Mozilla/5.0"}
        
        features = await feature_engine.extract_all_features(url, "url", context)
        
        assert features['target'] == url
        assert features['target_type'] == "url"
        assert 'features' in features
        assert 'url' in features['features']
        assert 'composite' in features['features']
    
    @pytest.mark.asyncio
    async def test_extract_ip_features(self, feature_engine):
        """Test IP feature extraction."""
        ip = "192.168.1.1"
        context = {"behavioral_data": {"request_count": 100}}
        
        features = await feature_engine.extract_all_features(ip, "ip", context)
        
        assert features['target'] == ip
        assert features['target_type'] == "ip"
        assert 'features' in features
        assert 'ip' in features['features']
        assert 'composite' in features['features']
    
    @pytest.mark.asyncio
    async def test_extract_features_with_temporal_context(self, feature_engine):
        """Test feature extraction with temporal context."""
        from datetime import datetime, timedelta
        
        url = "https://example.com"
        base_time = datetime.utcnow()
        timestamps = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2)
        ]
        
        context = {"timestamps": timestamps}
        
        features = await feature_engine.extract_all_features(url, "url", context)
        
        assert 'temporal' in features['features']
        assert features['features']['temporal']['total_events'] == 3
    
    def test_calculate_composite_features(self, feature_engine):
        """Test composite feature calculation."""
        features = {
            'url': {
                'has_ip_address': True,
                'entropy': 5.0,
                'suspicious_keywords': ['phishing:secure', 'phishing:login']
            },
            'ip': {
                'abuse_score': 0.8
            },
            'temporal': {
                'burst_ratio': 0.9
            }
        }
        
        composite = feature_engine._calculate_composite_features(features)
        
        assert 'risk_score' in composite
        assert 'suspicion_score' in composite
        assert 'anomaly_score' in composite
        assert 'threat_level' in composite
        assert 0 <= composite['risk_score'] <= 1
        assert 0 <= composite['suspicion_score'] <= 1
        assert 0 <= composite['anomaly_score'] <= 1
        assert composite['threat_level'] in ['low', 'medium', 'high', 'critical']
    
    def test_calculate_risk_score(self, feature_engine):
        """Test risk score calculation."""
        # High risk features
        high_risk_features = {
            'url': {
                'has_ip_address': True,
                'entropy': 6.0,
                'suspicious_keywords': ['phishing:secure', 'phishing:login', 'phishing:verify']
            },
            'ip': {
                'abuse_score': 0.9
            },
            'polymorphism': {
                'has_encoding': True
            }
        }
        
        risk_score = feature_engine._calculate_risk_score(high_risk_features)
        assert risk_score > 0.7
        
        # Low risk features
        low_risk_features = {
            'url': {
                'has_ip_address': False,
                'entropy': 2.0,
                'suspicious_keywords': []
            },
            'ip': {
                'abuse_score': 0.1
            }
        }
        
        risk_score = feature_engine._calculate_risk_score(low_risk_features)
        assert risk_score < 0.3


if __name__ == "__main__":
    pytest.main([__file__])

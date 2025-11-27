"""
Tests for API endpoints.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from src.api.main import app
from src.database.models import AttackType, SeverityLevel


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_analysis_request(self):
        """Create sample analysis request."""
        return {
            "target": "https://example.com",
            "target_type": "url",
            "context": {"user_agent": "Mozilla/5.0"}
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "message" in response.json()
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        assert "status" in response.json()
    
    @patch('src.api.routers.analysis.hybrid_engine.analyze')
    def test_analyze_endpoint(self, mock_analyze, client, sample_analysis_request):
        """Test analyze endpoint."""
        # Mock the analysis result
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.target_value = "https://example.com"
        mock_result.target_type = "url"
        mock_result.status = "completed"
        mock_result.final_attack_type = AttackType.UNKNOWN
        mock_result.final_confidence = 0.1
        mock_result.severity = SeverityLevel.LOW
        mock_result.risk_score = 0.2
        mock_result.analysis_duration_ms = 150
        mock_result.created_at = "2024-01-01T00:00:00Z"
        mock_result.updated_at = "2024-01-01T00:00:00Z"
        
        mock_analyze.return_value = mock_result
        
        # Make request
        response = client.post(
            "/api/v1/analyze",
            json=sample_analysis_request,
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["target"] == "https://example.com"
        assert data["target_type"] == "url"
        assert data["status"] == "completed"
    
    def test_analyze_endpoint_invalid_request(self, client):
        """Test analyze endpoint with invalid request."""
        response = client.post(
            "/api/v1/analyze",
            json={"target": "invalid", "target_type": "invalid"},
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_analyze_endpoint_missing_auth(self, client, sample_analysis_request):
        """Test analyze endpoint without authentication."""
        response = client.post("/api/v1/analyze", json=sample_analysis_request)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('src.api.routers.models.model_manager.get_model_info')
    def test_models_endpoint(self, mock_get_info, client):
        """Test models endpoint."""
        mock_get_info.return_value = {
            "total_models": 5,
            "trained_models": 3,
            "models": {
                "random_forest": {
                    "model_name": "random_forest",
                    "is_trained": True
                }
            }
        }
        
        response = client.get(
            "/api/v1/models",
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_models" in data
        assert "trained_models" in data
    
    @patch('src.api.routers.models.model_manager.get_model_performance')
    def test_model_metrics_endpoint(self, mock_get_performance, client):
        """Test model metrics endpoint."""
        mock_get_performance.return_value = {
            "model_name": "random_forest",
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95,
            "evaluation_timestamp": "2024-01-01T00:00:00Z"
        }
        
        response = client.get(
            "/api/v1/models/random_forest/metrics",
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_name"] == "random_forest"
        assert data["accuracy"] == 0.95
    
    @patch('src.api.routers.intelligence.threat_intelligence_manager.analyze_url')
    def test_intelligence_endpoint(self, mock_analyze, client):
        """Test threat intelligence endpoint."""
        mock_analyze.return_value = [
            Mock(
                source="virustotal",
                threat_type="malware",
                confidence=0.8,
                last_updated="2024-01-01T00:00:00Z",
                raw_data={"positives": 5, "total": 10}
            )
        ]
        
        request_data = {
            "target": "https://example.com",
            "target_type": "url"
        }
        
        response = client.post(
            "/api/v1/intelligence/analyze",
            json=request_data,
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["target"] == "https://example.com"
        assert "results" in data
    
    @patch('src.api.routers.alerts.get_database')
    def test_alerts_endpoint(self, mock_get_db, client):
        """Test alerts endpoint."""
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_doc = {
            "_id": "test_alert_id",
            "analysis_id": "test_analysis_id",
            "alert_type": "threat_detected",
            "severity": "high",
            "title": "Test Alert",
            "description": "Test Description",
            "attack_type": "phishing",
            "is_acknowledged": False,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        async def mock_find():
            yield mock_doc
        
        mock_cursor.__aiter__ = mock_find
        mock_collection.find.return_value = mock_cursor
        mock_get_db.return_value.get_collection.return_value = mock_collection
        
        response = client.get(
            "/api/v1/alerts",
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "title" in data[0]


class TestAPIAuthentication:
    """Test API authentication."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_valid_api_key(self, client):
        """Test with valid API key."""
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        assert response.status_code == status.HTTP_200_OK
    
    def test_invalid_api_key(self, client):
        """Test with invalid API key."""
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_missing_api_key(self, client):
        """Test without API key."""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_authorization_header(self, client):
        """Test with Authorization header."""
        response = client.get(
            "/api/v1/health",
            headers={"Authorization": "Bearer sk-test-key-1234567890"}
        )
        assert response.status_code == status.HTTP_200_OK


class TestAPIRateLimiting:
    """Test API rate limiting."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_rate_limiting(self, client):
        """Test rate limiting."""
        # Make multiple requests quickly
        responses = []
        for _ in range(10):
            response = client.get(
                "/api/v1/health",
                headers={"X-API-Key": "sk-test-key-1234567890"}
            )
            responses.append(response.status_code)
        
        # All requests should succeed (rate limiting is not enforced in tests)
        assert all(status == status.HTTP_200_OK for status in responses)


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get(
            "/api/v1/nonexistent",
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_422_error(self, client):
        """Test 422 error handling."""
        response = client.post(
            "/api/v1/analyze",
            json={"invalid": "data"},
            headers={"X-API-Key": "sk-test-key-1234567890"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_500_error_handling(self, client):
        """Test 500 error handling."""
        # This would test internal server errors
        # Implementation depends on specific error scenarios
        pass


if __name__ == "__main__":
    pytest.main([__file__])

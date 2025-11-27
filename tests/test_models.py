"""
Tests for ML models.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import asyncio

from src.models.classical_models import RandomForestModel, XGBoostModel
from src.models.deep_learning_models import CNNLSTMModel, TransformerModel
from src.models.model_manager import ModelManager
from src.database.models import AttackType


class TestClassicalModels:
    """Test classical ML models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        X, y = sample_data
        
        model = RandomForestModel()
        model.build_model(X.shape, len(np.unique(y)))
        
        # Test training
        metrics = model.train(X, y)
        assert model.is_trained
        assert 'accuracy' in metrics or len(metrics) == 0
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        
        # Test probability prediction
        probabilities = model.predict_proba(X[:5])
        assert probabilities.shape == (5, 2)
    
    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X, y = sample_data
        
        model = XGBoostModel()
        model.build_model(X.shape, len(np.unique(y)))
        
        # Test training
        metrics = model.train(X, y)
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        
        # Test probability prediction
        probabilities = model.predict_proba(X[:5])
        assert probabilities.shape == (5, 2)


class TestDeepLearningModels:
    """Test deep learning models."""
    
    @pytest.fixture
    def sample_sequence_data(self):
        """Create sample sequence data."""
        np.random.seed(42)
        X = np.random.randn(100, 50, 10)  # (samples, timesteps, features)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_cnn_lstm_model(self, sample_sequence_data):
        """Test CNN-LSTM model."""
        X, y = sample_sequence_data
        
        model = CNNLSTMModel()
        model.build_model(X.shape[1:], len(np.unique(y)))
        
        # Test training (with small dataset)
        metrics = model.train(X[:20], y[:20])
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_transformer_model(self, sample_sequence_data):
        """Test Transformer model."""
        X, y = sample_sequence_data
        
        model = TransformerModel()
        model.build_model(X.shape[1:], len(np.unique(y)))
        
        # Test training (with small dataset)
        metrics = model.train(X[:20], y[:20])
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5


class TestModelManager:
    """Test model manager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager instance."""
        return ModelManager()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.mark.asyncio
    async def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization."""
        await model_manager.initialize()
        assert model_manager.is_initialized
    
    @pytest.mark.asyncio
    async def test_train_model(self, model_manager, sample_data):
        """Test training a specific model."""
        X, y = sample_data
        
        await model_manager.initialize()
        
        # Test training Random Forest
        metrics = await model_manager.train_model('random_forest', X, y)
        assert 'random_forest' in model_manager.models
        assert model_manager.models['random_forest'].is_trained
    
    @pytest.mark.asyncio
    async def test_predict_with_model(self, model_manager, sample_data):
        """Test prediction with a model."""
        X, y = sample_data
        
        await model_manager.initialize()
        
        # Train a model first
        await model_manager.train_model('random_forest', X, y)
        
        # Test prediction
        features = {
            'feature_0': 1.0,
            'feature_1': 2.0,
            'feature_2': 3.0,
            'feature_3': 4.0,
            'feature_4': 5.0,
            'feature_5': 6.0,
            'feature_6': 7.0,
            'feature_7': 8.0,
            'feature_8': 9.0,
            'feature_9': 10.0
        }
        
        prediction = await model_manager.predict('random_forest', features)
        assert 'attack_type' in prediction
        assert 'confidence' in prediction
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, model_manager):
        """Test getting model information."""
        await model_manager.initialize()
        
        info = await model_manager.get_model_info()
        assert 'models' in info
        assert 'ensembles' in info
        assert 'total_models' in info


class TestModelEvaluation:
    """Test model evaluation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        X_train = np.random.randn(80, 10)
        y_train = np.random.randint(0, 2, 80)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test
    
    def test_model_evaluation(self, sample_data):
        """Test model evaluation."""
        X_train, y_train, X_test, y_test = sample_data
        
        model = RandomForestModel()
        model.build_model(X_train.shape, len(np.unique(y_train)))
        model.train(X_train, y_train)
        
        # Test evaluation
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1


class TestModelPersistence:
    """Test model persistence."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        return X, y
    
    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test saving and loading models."""
        X, y = sample_data
        
        # Create model and train
        model = RandomForestModel()
        model.model_path = tmp_path / "test_model.pkl"
        model.build_model(X.shape, len(np.unique(y)))
        model.train(X, y)
        
        # Save model
        model.save_model()
        assert model.model_path.exists()
        
        # Create new model and load
        new_model = RandomForestModel()
        new_model.model_path = tmp_path / "test_model.pkl"
        
        # Load model
        success = new_model.load_model()
        assert success
        assert new_model.is_trained
        
        # Test that loaded model works
        predictions = new_model.predict(X[:5])
        assert len(predictions) == 5


if __name__ == "__main__":
    pytest.main([__file__])

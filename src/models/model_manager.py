"""
Model management system for the cyberattack detection system.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import structlog
from pathlib import Path
import joblib
import pickle

from ..core.config import settings
from ..database.models import AttackType, ModelMetrics, AnalysisResult
from ..database.connection import get_database
from .base_model import BaseModel, EnsembleModel
from .classical_models import (
    RandomForestModel, XGBoostModel, SVMModel, 
    LogisticRegressionModel, NaiveBayesModel, KNNModel
)
from .deep_learning_models import CNNLSTMModel, TransformerModel, AutoEncoderModel

logger = structlog.get_logger(__name__)


class ModelManager:
    """Manages all ML models for cyberattack detection."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.ensemble_models: Dict[str, EnsembleModel] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.is_initialized = False
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestModel,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            },
            'xgboost': {
                'class': XGBoostModel,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'svm': {
                'class': SVMModel,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            },
            'logistic_regression': {
                'class': LogisticRegressionModel,
                'params': {
                    'C': 1.0,
                    'penalty': 'l2'
                }
            },
            'naive_bayes': {
                'class': NaiveBayesModel,
                'params': {}
            },
            'knn': {
                'class': KNNModel,
                'params': {
                    'n_neighbors': 5,
                    'weights': 'distance'
                }
            },
            'cnn_lstm': {
                'class': CNNLSTMModel,
                'params': {
                    'sequence_length': 100,
                    'embedding_dim': 128,
                    'cnn_filters': 64,
                    'lstm_units': 64
                }
            },
            'transformer': {
                'class': TransformerModel,
                'params': {
                    'sequence_length': 100,
                    'd_model': 128,
                    'num_heads': 8,
                    'num_layers': 4
                }
            },
            'autoencoder': {
                'class': AutoEncoderModel,
                'params': {
                    'encoding_dim': 32,
                    'hidden_dims': [128, 64]
                }
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the model manager."""
        try:
            # Load existing models
            await self._load_existing_models()
            
            # Create ensemble models
            await self._create_ensemble_models()
            
            self.is_initialized = True
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error("Error initializing model manager", error=str(e))
            raise
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models."""
        model_dir = Path(settings.model_dir)
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for model_name, config in self.model_configs.items():
            try:
                model_class = config['class']
                model_params = config['params']
                
                # Create model instance
                model = model_class(model_name=model_name, **model_params)
                
                # Try to load existing model
                if model.load_model():
                    self.models[model_name] = model
                    logger.info(f"Loaded existing model: {model_name}")
                else:
                    logger.info(f"No existing model found for: {model_name}")
                    
            except Exception as e:
                logger.warning(f"Error loading model {model_name}", error=str(e))
    
    async def _create_ensemble_models(self) -> None:
        """Create ensemble models."""
        try:
            # Create main ensemble with all available models
            available_models = [model for model in self.models.values() if model.is_trained]
            
            if len(available_models) >= 2:
                ensemble = EnsembleModel(
                    model_name="main_ensemble",
                    base_models=available_models,
                    ensemble_method="weighted_voting"
                )
                
                # Try to load existing ensemble
                if ensemble.load_model():
                    self.ensemble_models["main_ensemble"] = ensemble
                    logger.info("Loaded existing ensemble model")
                else:
                    logger.info("No existing ensemble model found")
            
            # Create specialized ensembles for different attack types
            await self._create_specialized_ensembles()
            
        except Exception as e:
            logger.error("Error creating ensemble models", error=str(e))
    
    async def _create_specialized_ensembles(self) -> None:
        """Create specialized ensemble models for different attack types."""
        # This would create specialized ensembles for different attack types
        # For now, we'll keep it simple
        pass
    
    async def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, 
                         y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train a specific model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        try:
            config = self.model_configs[model_name]
            model_class = config['class']
            model_params = config['params']
            
            # Create model instance
            model = model_class(model_name=model_name, **model_params)
            
            # Train the model
            metrics = model.train(X_train, y_train, X_val, y_val)
            
            # Save the model
            model.save_model()
            
            # Store in manager
            self.models[model_name] = model
            
            # Store metrics
            await self._store_model_metrics(model_name, metrics, X_train.shape[0], 
                                          X_val.shape[0] if X_val is not None else 0)
            
            logger.info(f"Model {model_name} trained successfully", metrics=metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model {model_name}", error=str(e))
            raise
    
    async def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None, 
                              y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Train all available models."""
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                metrics = await self.train_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error training model {model_name}", error=str(e))
                results[model_name] = {"error": str(e)}
        
        # Retrain ensemble models
        await self._retrain_ensembles()
        
        return results
    
    async def _retrain_ensembles(self) -> None:
        """Retrain ensemble models with updated base models."""
        try:
            # Update main ensemble
            if "main_ensemble" in self.ensemble_models:
                available_models = [model for model in self.models.values() if model.is_trained]
                if len(available_models) >= 2:
                    ensemble = self.ensemble_models["main_ensemble"]
                    ensemble.base_models = available_models
                    ensemble.save_model()
        
        except Exception as e:
            logger.error("Error retraining ensembles", error=str(e))
    
    async def predict(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_name} is not trained")
        
        try:
            # Prepare features
            X = model.prepare_features(features)
            
            # Make prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Get attack type
            attack_type = AttackType.UNKNOWN
            if hasattr(model, 'classes_'):
                if prediction < len(model.classes_):
                    attack_type = AttackType(model.classes_[prediction])
            
            # Calculate confidence
            confidence = float(np.max(probabilities))
            
            result = {
                'model_name': model_name,
                'attack_type': attack_type,
                'confidence': confidence,
                'probabilities': {f"class_{i}": float(prob) for i, prob in enumerate(probabilities)},
                'prediction_timestamp': datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}", error=str(e))
            raise
    
    async def predict_ensemble(self, ensemble_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using an ensemble model."""
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        
        if not ensemble.is_trained:
            raise ValueError(f"Ensemble {ensemble_name} is not trained")
        
        try:
            # Prepare features using the first base model
            base_model = ensemble.base_models[0]
            X = base_model.prepare_features(features)
            
            # Make prediction
            prediction = ensemble.predict(X)[0]
            probabilities = ensemble.predict_proba(X)[0]
            
            # Get attack type
            attack_type = AttackType.UNKNOWN
            if len(probabilities) > 1:
                attack_type = AttackType(list(AttackType)[prediction])
            
            # Calculate confidence
            confidence = float(np.max(probabilities))
            
            result = {
                'ensemble_name': ensemble_name,
                'attack_type': attack_type,
                'confidence': confidence,
                'probabilities': {f"class_{i}": float(prob) for i, prob in enumerate(probabilities)},
                'base_model_predictions': {},
                'prediction_timestamp': datetime.utcnow()
            }
            
            # Get predictions from individual base models
            for model in ensemble.base_models:
                if model.is_trained:
                    try:
                        model_pred = model.predict(X)[0]
                        model_proba = model.predict_proba(X)[0]
                        result['base_model_predictions'][model.model_name] = {
                            'prediction': int(model_pred),
                            'confidence': float(np.max(model_proba))
                        }
                    except Exception as e:
                        logger.warning(f"Error getting prediction from base model {model.model_name}", error=str(e))
            
            return result
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction with {ensemble_name}", error=str(e))
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models."""
        info = {
            'models': {},
            'ensembles': {},
            'total_models': len(self.models),
            'trained_models': len([m for m in self.models.values() if m.is_trained]),
            'total_ensembles': len(self.ensemble_models),
            'trained_ensembles': len([e for e in self.ensemble_models.values() if e.is_trained])
        }
        
        # Model information
        for name, model in self.models.items():
            info['models'][name] = model.get_model_info()
        
        # Ensemble information
        for name, ensemble in self.ensemble_models.items():
            info['ensembles'][name] = {
                'model_name': ensemble.model_name,
                'model_version': ensemble.model_version,
                'is_trained': ensemble.is_trained,
                'base_models': [m.model_name for m in ensemble.base_models],
                'ensemble_method': ensemble.ensemble_method
            }
        
        return info
    
    async def _store_model_metrics(self, model_name: str, metrics: Dict[str, float],
                                  training_samples: int, validation_samples: int) -> None:
        """Store model metrics in database."""
        try:
            db = await get_database()
            
            model_metrics = ModelMetrics(
                model_name=model_name,
                model_version="1.0.0",
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                auc_roc=metrics.get('auc_roc', 0.0),
                true_positives=metrics.get('true_positives', 0),
                false_positives=metrics.get('false_positives', 0),
                true_negatives=metrics.get('true_negatives', 0),
                false_negatives=metrics.get('false_negatives', 0),
                training_samples=training_samples,
                validation_samples=validation_samples,
                training_duration_seconds=0,  # Would calculate actual duration
                feature_count=len(self.models[model_name].feature_names),
                model_size_mb=0.0,  # Would calculate actual size
                training_start=datetime.utcnow(),
                training_end=datetime.utcnow()
            )
            
            collection = db.get_collection("model_metrics")
            await collection.insert_one(model_metrics.dict(by_alias=True))
            
        except Exception as e:
            logger.error("Error storing model metrics", error=str(e))
    
    async def update_model(self, model_name: str, new_data: np.ndarray, 
                          new_labels: np.ndarray) -> Dict[str, float]:
        """Update a model with new data (online learning)."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_name} is not trained")
        
        try:
            # For now, we'll retrain the model with combined data
            # In a real implementation, you'd use incremental learning
            
            # This is a placeholder for online learning
            logger.info(f"Updating model {model_name} with new data")
            
            # For demonstration, we'll just retrain
            # In practice, you'd implement proper online learning
            metrics = await self.train_model(model_name, new_data, new_labels)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating model {model_name}", error=str(e))
            raise
    
    async def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        try:
            db = await get_database()
            collection = db.get_collection("model_metrics")
            
            # Get latest metrics for the model
            metrics_doc = await collection.find_one(
                {"model_name": model_name},
                sort=[("evaluation_timestamp", -1)]
            )
            
            if metrics_doc:
                return {
                    'model_name': metrics_doc['model_name'],
                    'accuracy': metrics_doc['accuracy'],
                    'precision': metrics_doc['precision'],
                    'recall': metrics_doc['recall'],
                    'f1_score': metrics_doc['f1_score'],
                    'auc_roc': metrics_doc.get('auc_roc', 0.0),
                    'evaluation_timestamp': metrics_doc['evaluation_timestamp']
                }
            else:
                return {'error': 'No metrics found for model'}
                
        except Exception as e:
            logger.error(f"Error getting performance for model {model_name}", error=str(e))
            return {'error': str(e)}


# Global model manager instance
model_manager = ModelManager()


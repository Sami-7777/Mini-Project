"""
Base model class for all ML models in the cyberattack detection system.
"""
import pickle
import joblib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import structlog
from pathlib import Path

from ..core.config import settings
from ..database.models import AttackType, ModelMetrics

logger = structlog.get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_timestamp = None
        self.model_path = Path(settings.model_dir) / f"{model_name}_{model_version}.pkl"
        
        # Ensure model directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    def save_model(self) -> None:
        """Save the trained model."""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_name': self.model_name,
                'model_version': self.model_version,
                'training_timestamp': self.training_timestamp,
                'is_trained': self.is_trained
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully", 
                       model_name=self.model_name, 
                       path=str(self.model_path))
                       
        except Exception as e:
            logger.error("Error saving model", 
                        model_name=self.model_name, 
                        error=str(e))
            raise
    
    def load_model(self) -> bool:
        """Load a trained model."""
        try:
            if not self.model_path.exists():
                logger.warning("Model file not found", path=str(self.model_path))
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_name = model_data['model_name']
            self.model_version = model_data['model_version']
            self.training_timestamp = model_data['training_timestamp']
            self.is_trained = model_data['is_trained']
            
            logger.info("Model loaded successfully", 
                       model_name=self.model_name,
                       version=self.model_version)
            return True
            
        except Exception as e:
            logger.error("Error loading model", 
                        model_name=self.model_name, 
                        error=str(e))
            return False
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, confusion_matrix
            )
            
            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Calculate AUC-ROC if binary classification
            if len(np.unique(y_test)) == 2:
                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            metrics.update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            })
            
            logger.info("Model evaluation completed", 
                       model_name=self.model_name,
                       metrics=metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Error evaluating model", 
                        model_name=self.model_name, 
                        error=str(e))
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_trained:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
                return importance_dict
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
                return importance_dict
            else:
                logger.warning("Feature importance not available for this model type")
                return None
                
        except Exception as e:
            logger.error("Error getting feature importance", error=str(e))
            return None
    
    def prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction."""
        try:
            # Convert features to array in the correct order
            feature_array = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    value = features[feature_name]
                    # Handle different data types
                    if isinstance(value, (list, dict)):
                        # Convert complex features to numeric
                        value = len(value) if isinstance(value, list) else len(str(value))
                    elif not isinstance(value, (int, float)):
                        value = 0.0
                    feature_array.append(float(value))
                else:
                    # Missing feature, use default value
                    feature_array.append(0.0)
            
            return np.array(feature_array).reshape(1, -1)
            
        except Exception as e:
            logger.error("Error preparing features", error=str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp,
            'feature_count': len(self.feature_names),
            'model_path': str(self.model_path),
            'model_size_mb': self.model_path.stat().st_size / (1024 * 1024) if self.model_path.exists() else 0
        }


class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple base models."""
    
    def __init__(self, model_name: str, base_models: List[BaseModel], 
                 ensemble_method: str = 'voting', model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = None
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build ensemble model."""
        # Ensemble models don't have a single architecture
        return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train all base models."""
        training_metrics = {}
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.model_name}")
            
            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
                training_metrics[f"{model.model_name}_metrics"] = metrics
                
            except Exception as e:
                logger.error(f"Error training base model {model.model_name}", error=str(e))
                raise
        
        # Train ensemble weights if using weighted voting
        if self.ensemble_method == 'weighted_voting':
            self._train_ensemble_weights(X_val, y_val)
        
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        return training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before prediction")
        
        predictions = []
        for model in self.base_models:
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No trained base models available")
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'voting':
            return self._voting_predict(predictions)
        elif self.ensemble_method == 'weighted_voting':
            return self._weighted_voting_predict(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before prediction")
        
        probabilities = []
        for model in self.base_models:
            if model.is_trained:
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        if not probabilities:
            raise ValueError("No trained base models available")
        
        # Average probabilities
        return np.mean(probabilities, axis=0)
    
    def _voting_predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Majority voting prediction."""
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def _weighted_voting_predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Weighted voting prediction."""
        if self.weights is None:
            return self._voting_predict(predictions)
        
        predictions = np.array(predictions)
        weighted_predictions = predictions * self.weights.reshape(-1, 1)
        return np.argmax(np.sum(weighted_predictions, axis=0), axis=0)
    
    def _train_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train ensemble weights using validation data."""
        # Simple weight training based on individual model performance
        weights = []
        for model in self.base_models:
            if model.is_trained:
                metrics = model.evaluate(X_val, y_val)
                # Use F1 score as weight
                weight = metrics.get('f1_score', 0.5)
                weights.append(weight)
            else:
                weights.append(0.0)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            self.weights = np.array(weights) / total_weight
        else:
            self.weights = np.ones(len(weights)) / len(weights)
    
    def save_model(self) -> None:
        """Save ensemble model."""
        # Save individual base models
        for model in self.base_models:
            model.save_model()
        
        # Save ensemble metadata
        ensemble_data = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'ensemble_method': self.ensemble_method,
            'base_model_names': [model.model_name for model in self.base_models],
            'weights': self.weights,
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp
        }
        
        ensemble_path = self.model_path.parent / f"{self.model_name}_ensemble.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info("Ensemble model saved", model_name=self.model_name)
    
    def load_model(self) -> bool:
        """Load ensemble model."""
        try:
            # Load ensemble metadata
            ensemble_path = self.model_path.parent / f"{self.model_name}_ensemble.pkl"
            if not ensemble_path.exists():
                return False
            
            with open(ensemble_path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            # Load base models
            for model in self.base_models:
                if not model.load_model():
                    logger.warning(f"Failed to load base model: {model.model_name}")
            
            self.ensemble_method = ensemble_data['ensemble_method']
            self.weights = ensemble_data['weights']
            self.is_trained = ensemble_data['is_trained']
            self.training_timestamp = ensemble_data['training_timestamp']
            
            return True
            
        except Exception as e:
            logger.error("Error loading ensemble model", error=str(e))
            return False


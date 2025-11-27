"""
Model explainability and interpretability utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import structlog
from dataclasses import dataclass
from enum import Enum
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")

from ..database.models import AttackType, SeverityLevel
from ..models.model_manager import model_manager

logger = structlog.get_logger(__name__)


class ExplanationMethod(str, Enum):
    """Explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    GRADIENT = "gradient"


@dataclass
class ExplanationResult:
    """Result from model explanation."""
    method: ExplanationMethod
    target: str
    target_type: str
    attack_type: AttackType
    confidence: float
    feature_importance: Dict[str, float]
    explanation_text: str
    visualization_data: Optional[Dict[str, Any]] = None
    timestamp: str = ""


class ExplainabilityEngine:
    """Engine for generating model explanations."""
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = []
        self.training_data = None
    
    async def explain_prediction(self, target: str, target_type: str, 
                                features: Dict[str, Any], 
                                model_name: str = "main_ensemble") -> ExplanationResult:
        """Explain a model prediction."""
        try:
            # Get model prediction
            if model_name == "main_ensemble":
                prediction = await model_manager.predict_ensemble("main_ensemble", features)
            else:
                prediction = await model_manager.predict(model_name, features)
            
            # Generate explanation based on available methods
            explanation = None
            
            if SHAP_AVAILABLE:
                explanation = await self._generate_shap_explanation(
                    target, target_type, features, prediction, model_name
                )
            elif LIME_AVAILABLE:
                explanation = await self._generate_lime_explanation(
                    target, target_type, features, prediction, model_name
                )
            else:
                explanation = await self._generate_feature_importance_explanation(
                    target, target_type, features, prediction, model_name
                )
            
            return explanation
            
        except Exception as e:
            logger.error("Error explaining prediction", error=str(e))
            # Return fallback explanation
            return self._generate_fallback_explanation(target, target_type, features, prediction)
    
    async def _generate_shap_explanation(self, target: str, target_type: str,
                                        features: Dict[str, Any], 
                                        prediction: Dict[str, Any],
                                        model_name: str) -> ExplanationResult:
        """Generate SHAP explanation."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                raise ValueError("Could not prepare feature vector")
            
            # Get model
            if model_name == "main_ensemble":
                model = model_manager.ensemble_models["main_ensemble"]
            else:
                model = model_manager.models[model_name]
            
            # Create SHAP explainer if not exists
            if self.shap_explainer is None:
                await self._initialize_shap_explainer(model, model_name)
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_vector)
            
            # Process SHAP values
            if isinstance(shap_values, list):
                # Multi-class case
                class_idx = prediction.get('attack_type', 0)
                if isinstance(class_idx, str):
                    class_idx = list(AttackType).index(AttackType(class_idx))
                shap_values = shap_values[class_idx]
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(shap_values[0]):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_importance[feature_name] = float(importance)
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature_importance, prediction, "SHAP"
            )
            
            # Create visualization data
            visualization_data = {
                "shap_values": shap_values.tolist(),
                "feature_names": self.feature_names,
                "base_value": float(self.shap_explainer.expected_value),
                "method": "shap"
            }
            
            return ExplanationResult(
                method=ExplanationMethod.SHAP,
                target=target,
                target_type=target_type,
                attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                confidence=prediction.get('confidence', 0.0),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                visualization_data=visualization_data,
                timestamp=prediction.get('prediction_timestamp', '').isoformat()
            )
            
        except Exception as e:
            logger.error("Error generating SHAP explanation", error=str(e))
            raise
    
    async def _generate_lime_explanation(self, target: str, target_type: str,
                                        features: Dict[str, Any], 
                                        prediction: Dict[str, Any],
                                        model_name: str) -> ExplanationResult:
        """Generate LIME explanation."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                raise ValueError("Could not prepare feature vector")
            
            # Get model
            if model_name == "main_ensemble":
                model = model_manager.ensemble_models["main_ensemble"]
            else:
                model = model_manager.models[model_name]
            
            # Create LIME explainer if not exists
            if self.lime_explainer is None:
                await self._initialize_lime_explainer(model, model_name)
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                feature_vector[0], 
                model.predict_proba,
                num_features=min(10, len(self.feature_names))
            )
            
            # Process explanation
            feature_importance = {}
            for feature, importance in explanation.as_list():
                feature_importance[feature] = float(importance)
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature_importance, prediction, "LIME"
            )
            
            # Create visualization data
            visualization_data = {
                "explanation": explanation.as_list(),
                "feature_names": self.feature_names,
                "method": "lime"
            }
            
            return ExplanationResult(
                method=ExplanationMethod.LIME,
                target=target,
                target_type=target_type,
                attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                confidence=prediction.get('confidence', 0.0),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                visualization_data=visualization_data,
                timestamp=prediction.get('prediction_timestamp', '').isoformat()
            )
            
        except Exception as e:
            logger.error("Error generating LIME explanation", error=str(e))
            raise
    
    async def _generate_feature_importance_explanation(self, target: str, target_type: str,
                                                      features: Dict[str, Any], 
                                                      prediction: Dict[str, Any],
                                                      model_name: str) -> ExplanationResult:
        """Generate feature importance explanation."""
        try:
            # Get model
            if model_name == "main_ensemble":
                model = model_manager.ensemble_models["main_ensemble"]
            else:
                model = model_manager.models[model_name]
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            if feature_importance is None:
                # Fallback to manual feature analysis
                feature_importance = self._analyze_features_manually(features)
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature_importance, prediction, "Feature Importance"
            )
            
            return ExplanationResult(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                target=target,
                target_type=target_type,
                attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                confidence=prediction.get('confidence', 0.0),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                timestamp=prediction.get('prediction_timestamp', '').isoformat()
            )
            
        except Exception as e:
            logger.error("Error generating feature importance explanation", error=str(e))
            raise
    
    def _generate_fallback_explanation(self, target: str, target_type: str,
                                      features: Dict[str, Any], 
                                      prediction: Dict[str, Any]) -> ExplanationResult:
        """Generate fallback explanation when other methods fail."""
        try:
            # Manual feature analysis
            feature_importance = self._analyze_features_manually(features)
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature_importance, prediction, "Manual Analysis"
            )
            
            return ExplanationResult(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                target=target,
                target_type=target_type,
                attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                confidence=prediction.get('confidence', 0.0),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                timestamp=prediction.get('prediction_timestamp', '').isoformat()
            )
            
        except Exception as e:
            logger.error("Error generating fallback explanation", error=str(e))
            # Return minimal explanation
            return ExplanationResult(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                target=target,
                target_type=target_type,
                attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                confidence=prediction.get('confidence', 0.0),
                feature_importance={},
                explanation_text="Explanation not available",
                timestamp=prediction.get('prediction_timestamp', '').isoformat()
            )
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for explanation."""
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
                return np.array(numeric_features).reshape(1, -1)
            
            return None
            
        except Exception as e:
            logger.error("Error preparing feature vector", error=str(e))
            return None
    
    async def _initialize_shap_explainer(self, model, model_name: str):
        """Initialize SHAP explainer."""
        try:
            # Get training data for background
            if self.training_data is None:
                await self._load_training_data()
            
            if self.training_data is not None:
                # Create TreeExplainer for tree-based models
                if hasattr(model, 'base_models'):
                    # Ensemble model
                    base_model = model.base_models[0]
                    self.shap_explainer = shap.TreeExplainer(base_model.model)
                else:
                    # Single model
                    self.shap_explainer = shap.TreeExplainer(model.model)
                
                # Set feature names
                self.feature_names = model.feature_names if hasattr(model, 'feature_names') else []
                
        except Exception as e:
            logger.error("Error initializing SHAP explainer", error=str(e))
            raise
    
    async def _initialize_lime_explainer(self, model, model_name: str):
        """Initialize LIME explainer."""
        try:
            # Get training data for background
            if self.training_data is None:
                await self._load_training_data()
            
            if self.training_data is not None:
                # Create LIME explainer
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    self.training_data,
                    feature_names=self.feature_names,
                    class_names=[at.value for at in AttackType],
                    mode='classification'
                )
                
        except Exception as e:
            logger.error("Error initializing LIME explainer", error=str(e))
            raise
    
    async def _load_training_data(self):
        """Load training data for explainers."""
        try:
            # This would load actual training data
            # For now, we'll create synthetic data
            n_samples = 1000
            n_features = 20
            
            self.training_data = np.random.randn(n_samples, n_features)
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
            
        except Exception as e:
            logger.error("Error loading training data", error=str(e))
    
    def _analyze_features_manually(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Manually analyze features for importance."""
        try:
            importance = {}
            
            # URL features
            url_features = features.get('url', {})
            if url_features:
                importance['url_length'] = min(url_features.get('url_length', 0) / 100, 1.0)
                importance['entropy'] = min(url_features.get('entropy', 0) / 10, 1.0)
                importance['suspicious_keywords'] = len(url_features.get('suspicious_keywords', [])) / 10
            
            # IP features
            ip_features = features.get('ip', {})
            if ip_features:
                importance['abuse_score'] = ip_features.get('abuse_score', 0) or 0
                importance['reputation_score'] = ip_features.get('reputation_score', 0) or 0
            
            # Composite features
            composite_features = features.get('composite', {})
            if composite_features:
                importance['risk_score'] = composite_features.get('risk_score', 0)
                importance['suspicion_score'] = composite_features.get('suspicion_score', 0)
            
            return importance
            
        except Exception as e:
            logger.error("Error analyzing features manually", error=str(e))
            return {}
    
    def _generate_explanation_text(self, feature_importance: Dict[str, float], 
                                  prediction: Dict[str, Any], 
                                  method: str) -> str:
        """Generate human-readable explanation text."""
        try:
            attack_type = prediction.get('attack_type', AttackType.UNKNOWN)
            confidence = prediction.get('confidence', 0.0)
            
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Generate explanation
            explanation = f"""
            **Prediction Explanation ({method})**
            
            **Attack Type:** {attack_type.value.title()}
            **Confidence:** {confidence:.2%}
            
            **Key Factors:**
            """
            
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                direction = "increases" if importance > 0 else "decreases"
                explanation += f"\n{i+1}. **{feature}** {direction} the likelihood of {attack_type.value} (impact: {importance:.3f})"
            
            # Add risk assessment
            risk_score = prediction.get('risk_score', 0.0)
            if risk_score > 0.8:
                explanation += f"\n\n**Risk Assessment:** HIGH RISK - This target shows strong indicators of malicious activity."
            elif risk_score > 0.5:
                explanation += f"\n\n**Risk Assessment:** MEDIUM RISK - This target shows some suspicious characteristics."
            else:
                explanation += f"\n\n**Risk Assessment:** LOW RISK - This target appears to be legitimate."
            
            return explanation.strip()
            
        except Exception as e:
            logger.error("Error generating explanation text", error=str(e))
            return "Explanation not available"
    
    async def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get explanation statistics."""
        try:
            return {
                "shap_available": SHAP_AVAILABLE,
                "lime_available": LIME_AVAILABLE,
                "feature_count": len(self.feature_names),
                "training_data_loaded": self.training_data is not None
            }
            
        except Exception as e:
            logger.error("Error getting explanation statistics", error=str(e))
            return {}


# Global explainability engine instance
explainability_engine = ExplainabilityEngine()

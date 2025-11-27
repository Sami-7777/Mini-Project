"""
Advanced Explainable AI for cyberattack detection with multiple explanation methods.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from ..database.models import AttackType, SeverityLevel
from ..models.model_manager import model_manager

logger = structlog.get_logger(__name__)


class ExplanationMethod(str, Enum):
    """Explanation methods for AI models."""
    SHAP = "shap"
    LIME = "lime"
    GRADIENT = "gradient"
    ATTENTION = "attention"
    COUNTERFACTUAL = "counterfactual"
    CAUSAL = "causal"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class ExplanationResult:
    """Comprehensive explanation result."""
    method: ExplanationMethod
    target: str
    target_type: str
    attack_type: AttackType
    confidence: float
    feature_importance: Dict[str, float]
    explanation_text: str
    visualization_data: Optional[Dict[str, Any]] = None
    counterfactual_examples: Optional[List[Dict[str, Any]]] = None
    causal_factors: Optional[Dict[str, Any]] = None
    attention_weights: Optional[Dict[str, float]] = None
    uncertainty_quantification: Optional[Dict[str, float]] = None
    timestamp: datetime = None


class SHAPExplainer:
    """SHAP-based explainer for model interpretability."""
    
    def __init__(self):
        self.explainers = {}
        self.background_data = None
    
    async def explain_prediction(self, model, features: Dict[str, Any], 
                               target: str, target_type: str) -> ExplanationResult:
        """Explain prediction using SHAP."""
        try:
            if not SHAP_AVAILABLE:
                raise ValueError("SHAP not available")
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                raise ValueError("Could not prepare feature vector")
            
            # Get or create explainer
            model_name = getattr(model, 'model_name', 'unknown')
            if model_name not in self.explainers:
                await self._create_shap_explainer(model, model_name)
            
            explainer = self.explainers[model_name]
            
            # Generate SHAP values
            shap_values = explainer.shap_values(feature_vector)
            
            # Process SHAP values
            if isinstance(shap_values, list):
                # Multi-class case
                class_idx = 0  # Use first class for now
                shap_values = shap_values[class_idx]
            
            # Create feature importance dictionary
            feature_names = getattr(model, 'feature_names', [f"feature_{i}" for i in range(len(feature_vector[0]))])
            feature_importance = {}
            
            for i, importance in enumerate(shap_values[0]):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                feature_importance[feature_name] = float(importance)
            
            # Generate explanation text
            explanation_text = self._generate_shap_explanation_text(feature_importance, features)
            
            # Create visualization data
            visualization_data = self._create_shap_visualization(shap_values, feature_names)
            
            return ExplanationResult(
                method=ExplanationMethod.SHAP,
                target=target,
                target_type=target_type,
                attack_type=AttackType.UNKNOWN,  # Would be determined by model
                confidence=0.0,  # Would be determined by model
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                visualization_data=visualization_data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error explaining with SHAP", error=str(e))
            raise
    
    async def _create_shap_explainer(self, model, model_name: str):
        """Create SHAP explainer for a model."""
        try:
            # Get background data
            if self.background_data is None:
                await self._load_background_data()
            
            # Create explainer based on model type
            if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                # Tree-based model
                self.explainers[model_name] = shap.TreeExplainer(model.model)
            else:
                # Generic explainer
                self.explainers[model_name] = shap.Explainer(
                    model.predict_proba,
                    self.background_data
                )
            
        except Exception as e:
            logger.error("Error creating SHAP explainer", error=str(e))
            raise
    
    async def _load_background_data(self):
        """Load background data for SHAP."""
        try:
            # Generate synthetic background data
            # In practice, this would load from historical data
            n_samples = 100
            n_features = 20
            
            self.background_data = np.random.randn(n_samples, n_features)
            
        except Exception as e:
            logger.error("Error loading background data", error=str(e))
            raise
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for SHAP."""
        try:
            # Extract numeric features
            numeric_features = []
            
            # URL features
            url_features = features.get('url', {})
            if url_features:
                numeric_features.extend([
                    url_features.get('url_length', 0),
                    url_features.get('entropy', 0),
                    url_features.get('digit_count', 0)
                ])
            
            # IP features
            ip_features = features.get('ip', {})
            if ip_features:
                numeric_features.extend([
                    ip_features.get('reputation_score', 0) or 0,
                    ip_features.get('abuse_score', 0) or 0
                ])
            
            # Composite features
            composite_features = features.get('composite', {})
            if composite_features:
                numeric_features.extend([
                    composite_features.get('risk_score', 0),
                    composite_features.get('suspicion_score', 0)
                ])
            
            # Pad to fixed size
            target_size = 20
            if len(numeric_features) < target_size:
                numeric_features.extend([0.0] * (target_size - len(numeric_features)))
            else:
                numeric_features = numeric_features[:target_size]
            
            return np.array(numeric_features).reshape(1, -1)
            
        except Exception as e:
            logger.error("Error preparing feature vector", error=str(e))
            return None
    
    def _generate_shap_explanation_text(self, feature_importance: Dict[str, float], 
                                      features: Dict[str, Any]) -> str:
        """Generate human-readable SHAP explanation."""
        try:
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanation = "**SHAP Explanation:**\n\n"
            explanation += "**Top Contributing Factors:**\n"
            
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                direction = "increases" if importance > 0 else "decreases"
                explanation += f"{i+1}. **{feature}** {direction} the threat likelihood (impact: {importance:.3f})\n"
            
            # Add feature values
            explanation += "\n**Current Feature Values:**\n"
            for feature, importance in sorted_features[:5]:
                current_value = self._get_feature_value(features, feature)
                explanation += f"- {feature}: {current_value}\n"
            
            return explanation
            
        except Exception as e:
            logger.error("Error generating SHAP explanation text", error=str(e))
            return "SHAP explanation not available"
    
    def _get_feature_value(self, features: Dict[str, Any], feature_name: str) -> Any:
        """Get current value of a feature."""
        try:
            # Map feature names to actual values
            if 'url_length' in feature_name:
                return features.get('url', {}).get('url_length', 0)
            elif 'entropy' in feature_name:
                return features.get('url', {}).get('entropy', 0)
            elif 'risk_score' in feature_name:
                return features.get('composite', {}).get('risk_score', 0)
            else:
                return "N/A"
                
        except Exception as e:
            logger.error("Error getting feature value", error=str(e))
            return "N/A"
    
    def _create_shap_visualization(self, shap_values: np.ndarray, 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Create SHAP visualization data."""
        try:
            # Create waterfall plot data
            waterfall_data = []
            for i, (feature, value) in enumerate(zip(feature_names, shap_values[0])):
                waterfall_data.append({
                    "feature": feature,
                    "value": float(value),
                    "cumulative": float(np.sum(shap_values[0][:i+1]))
                })
            
            # Create summary plot data
            summary_data = {
                "features": feature_names,
                "values": shap_values[0].tolist(),
                "base_value": 0.0  # Would be actual base value
            }
            
            return {
                "waterfall": waterfall_data,
                "summary": summary_data,
                "plot_type": "shap"
            }
            
        except Exception as e:
            logger.error("Error creating SHAP visualization", error=str(e))
            return {}


class CounterfactualExplainer:
    """Counterfactual explanation generator."""
    
    def __init__(self):
        self.counterfactual_examples = {}
    
    async def generate_counterfactuals(self, features: Dict[str, Any], 
                                     target_prediction: AttackType,
                                     desired_prediction: AttackType) -> List[Dict[str, Any]]:
        """Generate counterfactual examples."""
        try:
            counterfactuals = []
            
            # Generate multiple counterfactual examples
            for i in range(5):
                counterfactual = self._generate_single_counterfactual(
                    features, target_prediction, desired_prediction
                )
                counterfactuals.append(counterfactual)
            
            return counterfactuals
            
        except Exception as e:
            logger.error("Error generating counterfactuals", error=str(e))
            return []
    
    def _generate_single_counterfactual(self, original_features: Dict[str, Any],
                                      target_prediction: AttackType,
                                      desired_prediction: AttackType) -> Dict[str, Any]:
        """Generate a single counterfactual example."""
        try:
            # Create modified features
            modified_features = original_features.copy()
            
            # Modify features to change prediction
            if target_prediction == AttackType.PHISHING and desired_prediction == AttackType.UNKNOWN:
                # Reduce phishing indicators
                if 'url' in modified_features:
                    modified_features['url']['suspicious_keywords'] = []
                    modified_features['url']['entropy'] = max(0, modified_features['url'].get('entropy', 0) - 2)
                
                if 'composite' in modified_features:
                    modified_features['composite']['risk_score'] = max(0, modified_features['composite'].get('risk_score', 0) - 0.3)
            
            elif target_prediction == AttackType.MALWARE and desired_prediction == AttackType.UNKNOWN:
                # Reduce malware indicators
                if 'ip' in modified_features:
                    modified_features['ip']['abuse_score'] = max(0, modified_features['ip'].get('abuse_score', 0) - 0.5)
                
                if 'composite' in modified_features:
                    modified_features['composite']['suspicion_score'] = max(0, modified_features['composite'].get('suspicion_score', 0) - 0.4)
            
            # Calculate changes
            changes = self._calculate_feature_changes(original_features, modified_features)
            
            return {
                "original_features": original_features,
                "modified_features": modified_features,
                "changes": changes,
                "target_prediction": target_prediction.value,
                "desired_prediction": desired_prediction.value,
                "confidence_change": 0.3,  # Simulated confidence change
                "explanation": f"To change from {target_prediction.value} to {desired_prediction.value}, modify: {', '.join(changes.keys())}"
            }
            
        except Exception as e:
            logger.error("Error generating single counterfactual", error=str(e))
            return {}
    
    def _calculate_feature_changes(self, original: Dict[str, Any], 
                                 modified: Dict[str, Any]) -> Dict[str, float]:
        """Calculate changes between original and modified features."""
        try:
            changes = {}
            
            # Compare URL features
            if 'url' in original and 'url' in modified:
                orig_url = original['url']
                mod_url = modified['url']
                
                if orig_url.get('entropy', 0) != mod_url.get('entropy', 0):
                    changes['url_entropy'] = mod_url.get('entropy', 0) - orig_url.get('entropy', 0)
                
                if len(orig_url.get('suspicious_keywords', [])) != len(mod_url.get('suspicious_keywords', [])):
                    changes['suspicious_keywords'] = len(mod_url.get('suspicious_keywords', [])) - len(orig_url.get('suspicious_keywords', []))
            
            # Compare IP features
            if 'ip' in original and 'ip' in modified:
                orig_ip = original['ip']
                mod_ip = modified['ip']
                
                if orig_ip.get('abuse_score', 0) != mod_ip.get('abuse_score', 0):
                    changes['ip_abuse_score'] = mod_ip.get('abuse_score', 0) - orig_ip.get('abuse_score', 0)
            
            # Compare composite features
            if 'composite' in original and 'composite' in modified:
                orig_comp = original['composite']
                mod_comp = modified['composite']
                
                if orig_comp.get('risk_score', 0) != mod_comp.get('risk_score', 0):
                    changes['risk_score'] = mod_comp.get('risk_score', 0) - orig_comp.get('risk_score', 0)
                
                if orig_comp.get('suspicion_score', 0) != mod_comp.get('suspicion_score', 0):
                    changes['suspicion_score'] = mod_comp.get('suspicion_score', 0) - orig_comp.get('suspicion_score', 0)
            
            return changes
            
        except Exception as e:
            logger.error("Error calculating feature changes", error=str(e))
            return {}


class CausalExplainer:
    """Causal explanation generator."""
    
    def __init__(self):
        self.causal_graph = {}
        self.causal_effects = {}
    
    async def explain_causality(self, features: Dict[str, Any], 
                              prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Explain causal relationships."""
        try:
            # Build causal graph
            causal_graph = self._build_causal_graph(features)
            
            # Calculate causal effects
            causal_effects = self._calculate_causal_effects(features, prediction)
            
            # Identify causal paths
            causal_paths = self._identify_causal_paths(features, prediction)
            
            return {
                "causal_graph": causal_graph,
                "causal_effects": causal_effects,
                "causal_paths": causal_paths,
                "direct_causes": self._identify_direct_causes(features, prediction),
                "indirect_causes": self._identify_indirect_causes(features, prediction)
            }
            
        except Exception as e:
            logger.error("Error explaining causality", error=str(e))
            return {}
    
    def _build_causal_graph(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build causal graph from features."""
        try:
            graph = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            if 'url' in features:
                graph["nodes"].append({"id": "url_features", "type": "feature_group"})
                graph["nodes"].append({"id": "url_entropy", "type": "feature", "parent": "url_features"})
                graph["nodes"].append({"id": "url_keywords", "type": "feature", "parent": "url_features"})
            
            if 'ip' in features:
                graph["nodes"].append({"id": "ip_features", "type": "feature_group"})
                graph["nodes"].append({"id": "ip_reputation", "type": "feature", "parent": "ip_features"})
                graph["nodes"].append({"id": "ip_abuse", "type": "feature", "parent": "ip_features"})
            
            if 'composite' in features:
                graph["nodes"].append({"id": "composite_features", "type": "feature_group"})
                graph["nodes"].append({"id": "risk_score", "type": "feature", "parent": "composite_features"})
                graph["nodes"].append({"id": "suspicion_score", "type": "feature", "parent": "composite_features"})
            
            # Add edges (causal relationships)
            graph["edges"].extend([
                {"from": "url_entropy", "to": "risk_score", "strength": 0.3},
                {"from": "url_keywords", "to": "suspicion_score", "strength": 0.5},
                {"from": "ip_reputation", "to": "risk_score", "strength": 0.4},
                {"from": "ip_abuse", "to": "suspicion_score", "strength": 0.6},
                {"from": "risk_score", "to": "final_prediction", "strength": 0.7},
                {"from": "suspicion_score", "to": "final_prediction", "strength": 0.8}
            ])
            
            return graph
            
        except Exception as e:
            logger.error("Error building causal graph", error=str(e))
            return {}
    
    def _calculate_causal_effects(self, features: Dict[str, Any], 
                                prediction: Dict[str, Any]) -> Dict[str, float]:
        """Calculate causal effects of features."""
        try:
            effects = {}
            
            # Calculate direct effects
            if 'url' in features:
                url_entropy = features['url'].get('entropy', 0)
                effects['url_entropy'] = url_entropy * 0.3
            
            if 'ip' in features:
                ip_abuse = features['ip'].get('abuse_score', 0) or 0
                effects['ip_abuse'] = ip_abuse * 0.6
            
            if 'composite' in features:
                risk_score = features['composite'].get('risk_score', 0)
                effects['risk_score'] = risk_score * 0.7
            
            return effects
            
        except Exception as e:
            logger.error("Error calculating causal effects", error=str(e))
            return {}
    
    def _identify_causal_paths(self, features: Dict[str, Any], 
                             prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify causal paths to prediction."""
        try:
            paths = []
            
            # Path 1: URL -> Risk -> Prediction
            if 'url' in features and 'composite' in features:
                paths.append({
                    "path": ["url_features", "risk_score", "final_prediction"],
                    "strength": 0.3,
                    "description": "URL features influence risk score, which affects final prediction"
                })
            
            # Path 2: IP -> Suspicion -> Prediction
            if 'ip' in features and 'composite' in features:
                paths.append({
                    "path": ["ip_features", "suspicion_score", "final_prediction"],
                    "strength": 0.6,
                    "description": "IP features influence suspicion score, which affects final prediction"
                })
            
            return paths
            
        except Exception as e:
            logger.error("Error identifying causal paths", error=str(e))
            return []
    
    def _identify_direct_causes(self, features: Dict[str, Any], 
                              prediction: Dict[str, Any]) -> List[str]:
        """Identify direct causes of prediction."""
        try:
            direct_causes = []
            
            if 'composite' in features:
                if features['composite'].get('risk_score', 0) > 0.5:
                    direct_causes.append("high_risk_score")
                
                if features['composite'].get('suspicion_score', 0) > 0.5:
                    direct_causes.append("high_suspicion_score")
            
            return direct_causes
            
        except Exception as e:
            logger.error("Error identifying direct causes", error=str(e))
            return []
    
    def _identify_indirect_causes(self, features: Dict[str, Any], 
                                prediction: Dict[str, Any]) -> List[str]:
        """Identify indirect causes of prediction."""
        try:
            indirect_causes = []
            
            if 'url' in features:
                if features['url'].get('entropy', 0) > 5:
                    indirect_causes.append("high_url_entropy")
                
                if len(features['url'].get('suspicious_keywords', [])) > 2:
                    indirect_causes.append("multiple_suspicious_keywords")
            
            if 'ip' in features:
                if features['ip'].get('abuse_score', 0) and features['ip']['abuse_score'] > 0.7:
                    indirect_causes.append("high_ip_abuse_score")
            
            return indirect_causes
            
        except Exception as e:
            logger.error("Error identifying indirect causes", error=str(e))
            return []


class ExplainableAIEngine:
    """Main engine for explainable AI."""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.counterfactual_explainer = CounterfactualExplainer()
        self.causal_explainer = CausalExplainer()
        self.explanation_cache = {}
    
    async def explain_prediction(self, target: str, target_type: str, 
                               features: Dict[str, Any], 
                               prediction: Dict[str, Any],
                               methods: List[ExplanationMethod] = None) -> List[ExplanationResult]:
        """Explain prediction using multiple methods."""
        try:
            if methods is None:
                methods = [ExplanationMethod.SHAP, ExplanationMethod.COUNTERFACTUAL, ExplanationMethod.CAUSAL]
            
            explanations = []
            
            for method in methods:
                try:
                    if method == ExplanationMethod.SHAP and SHAP_AVAILABLE:
                        explanation = await self.shap_explainer.explain_prediction(
                            model=None,  # Would pass actual model
                            features=features,
                            target=target,
                            target_type=target_type
                        )
                        explanations.append(explanation)
                    
                    elif method == ExplanationMethod.COUNTERFACTUAL:
                        counterfactuals = await self.counterfactual_explainer.generate_counterfactuals(
                            features=features,
                            target_prediction=prediction.get('attack_type', AttackType.UNKNOWN),
                            desired_prediction=AttackType.UNKNOWN
                        )
                        
                        explanation = ExplanationResult(
                            method=ExplanationMethod.COUNTERFACTUAL,
                            target=target,
                            target_type=target_type,
                            attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                            confidence=prediction.get('confidence', 0.0),
                            feature_importance={},
                            explanation_text="Counterfactual explanation generated",
                            counterfactual_examples=counterfactuals,
                            timestamp=datetime.utcnow()
                        )
                        explanations.append(explanation)
                    
                    elif method == ExplanationMethod.CAUSAL:
                        causal_explanation = await self.causal_explainer.explain_causality(
                            features=features,
                            prediction=prediction
                        )
                        
                        explanation = ExplanationResult(
                            method=ExplanationMethod.CAUSAL,
                            target=target,
                            target_type=target_type,
                            attack_type=prediction.get('attack_type', AttackType.UNKNOWN),
                            confidence=prediction.get('confidence', 0.0),
                            feature_importance={},
                            explanation_text="Causal explanation generated",
                            causal_factors=causal_explanation,
                            timestamp=datetime.utcnow()
                        )
                        explanations.append(explanation)
                
                except Exception as e:
                    logger.error(f"Error explaining with {method}", error=str(e))
                    continue
            
            logger.info("Prediction explanation completed", 
                       target=target,
                       methods=len(explanations))
            
            return explanations
            
        except Exception as e:
            logger.error("Error explaining prediction", error=str(e))
            return []
    
    async def generate_explanation_report(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate comprehensive explanation report."""
        try:
            report = {
                "summary": {
                    "total_explanations": len(explanations),
                    "methods_used": [exp.method.value for exp in explanations],
                    "target": explanations[0].target if explanations else "",
                    "target_type": explanations[0].target_type if explanations else "",
                    "attack_type": explanations[0].attack_type.value if explanations else "",
                    "confidence": explanations[0].confidence if explanations else 0.0
                },
                "explanations": [],
                "consensus": {},
                "recommendations": []
            }
            
            # Process each explanation
            for explanation in explanations:
                exp_data = {
                    "method": explanation.method.value,
                    "explanation_text": explanation.explanation_text,
                    "feature_importance": explanation.feature_importance,
                    "visualization_data": explanation.visualization_data,
                    "counterfactual_examples": explanation.counterfactual_examples,
                    "causal_factors": explanation.causal_factors
                }
                report["explanations"].append(exp_data)
            
            # Generate consensus
            report["consensus"] = self._generate_explanation_consensus(explanations)
            
            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(explanations)
            
            return report
            
        except Exception as e:
            logger.error("Error generating explanation report", error=str(e))
            return {}
    
    def _generate_explanation_consensus(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate consensus across explanation methods."""
        try:
            consensus = {
                "agreement_score": 0.0,
                "common_factors": [],
                "conflicting_factors": [],
                "confidence_range": {"min": 0.0, "max": 0.0}
            }
            
            if not explanations:
                return consensus
            
            # Extract common important features
            all_features = []
            for explanation in explanations:
                if explanation.feature_importance:
                    all_features.extend(explanation.feature_importance.keys())
            
            # Find common features
            from collections import Counter
            feature_counts = Counter(all_features)
            common_features = [feature for feature, count in feature_counts.items() if count >= 2]
            
            consensus["common_factors"] = common_features
            
            # Calculate agreement score
            if len(explanations) > 1:
                # Simple agreement based on common features
                consensus["agreement_score"] = len(common_features) / len(set(all_features))
            
            # Calculate confidence range
            confidences = [exp.confidence for exp in explanations if exp.confidence > 0]
            if confidences:
                consensus["confidence_range"] = {
                    "min": min(confidences),
                    "max": max(confidences)
                }
            
            return consensus
            
        except Exception as e:
            logger.error("Error generating explanation consensus", error=str(e))
            return {}
    
    def _generate_recommendations(self, explanations: List[ExplanationResult]) -> List[str]:
        """Generate recommendations based on explanations."""
        try:
            recommendations = []
            
            # Analyze explanations for recommendations
            for explanation in explanations:
                if explanation.method == ExplanationMethod.SHAP:
                    # SHAP-based recommendations
                    if explanation.feature_importance:
                        top_feature = max(explanation.feature_importance.items(), key=lambda x: abs(x[1]))
                        recommendations.append(f"Focus on {top_feature[0]} as it has the highest impact ({top_feature[1]:.3f})")
                
                elif explanation.method == ExplanationMethod.COUNTERFACTUAL:
                    # Counterfactual-based recommendations
                    if explanation.counterfactual_examples:
                        recommendations.append("Consider the counterfactual examples to understand what changes would alter the prediction")
                
                elif explanation.method == ExplanationMethod.CAUSAL:
                    # Causal-based recommendations
                    if explanation.causal_factors:
                        recommendations.append("Review the causal relationships to understand the underlying mechanisms")
            
            # General recommendations
            if len(explanations) > 1:
                recommendations.append("Multiple explanation methods provide different perspectives on the prediction")
            
            return recommendations
            
        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            return []
    
    async def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get explanation statistics."""
        try:
            return {
                "shap_available": SHAP_AVAILABLE,
                "lime_available": LIME_AVAILABLE,
                "cached_explanations": len(self.explanation_cache),
                "available_methods": [method.value for method in ExplanationMethod],
                "background_data_loaded": self.shap_explainer.background_data is not None
            }
            
        except Exception as e:
            logger.error("Error getting explanation statistics", error=str(e))
            return {}


# Global explainable AI engine instance
explainable_ai_engine = ExplainableAIEngine()

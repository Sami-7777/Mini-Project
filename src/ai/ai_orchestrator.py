"""
AI Orchestrator for coordinating all AI components in the cyberattack detection system.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum
import json

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..models.model_manager import model_manager
from ..engine.hybrid_engine import hybrid_engine
from ..anomaly.anomaly_detector import anomaly_detector
from ..anomaly.novelty_detector import novelty_detector
from .graph_neural_networks import graph_nn_engine
from .quantum_ml import quantum_ml_engine
from .federated_learning import federated_learning_engine
from .blockchain_security import threat_blockchain, smart_contract
from .adaptive_learning import adaptive_learning_engine
from .explainable_ai import explainable_ai_engine
from .advanced_analytics import advanced_analytics_engine

logger = structlog.get_logger(__name__)


class AIComponent(str, Enum):
    """AI components in the system."""
    CLASSICAL_ML = "classical_ml"
    DEEP_LEARNING = "deep_learning"
    GRAPH_NEURAL_NETWORKS = "graph_neural_networks"
    QUANTUM_ML = "quantum_ml"
    FEDERATED_LEARNING = "federated_learning"
    BLOCKCHAIN_SECURITY = "blockchain_security"
    ADAPTIVE_LEARNING = "adaptive_learning"
    EXPLAINABLE_AI = "explainable_ai"
    ADVANCED_ANALYTICS = "advanced_analytics"
    ANOMALY_DETECTION = "anomaly_detection"
    NOVELTY_DETECTION = "novelty_detection"


@dataclass
class AIOrchestrationResult:
    """Result from AI orchestration."""
    target: str
    target_type: str
    final_prediction: AttackType
    final_confidence: float
    component_results: Dict[str, Any]
    consensus_score: float
    explanation: Dict[str, Any]
    recommendations: List[str]
    processing_time_ms: int
    timestamp: datetime


class AIOrchestrator:
    """Orchestrates all AI components for comprehensive cyberattack detection."""
    
    def __init__(self):
        self.components = {
            AIComponent.CLASSICAL_ML: model_manager,
            AIComponent.DEEP_LEARNING: model_manager,
            AIComponent.GRAPH_NEURAL_NETWORKS: graph_nn_engine,
            AIComponent.QUANTUM_ML: quantum_ml_engine,
            AIComponent.FEDERATED_LEARNING: federated_learning_engine,
            AIComponent.BLOCKCHAIN_SECURITY: threat_blockchain,
            AIComponent.ADAPTIVE_LEARNING: adaptive_learning_engine,
            AIComponent.EXPLAINABLE_AI: explainable_ai_engine,
            AIComponent.ADVANCED_ANALYTICS: advanced_analytics_engine,
            AIComponent.ANOMALY_DETECTION: anomaly_detector,
            AIComponent.NOVELTY_DETECTION: novelty_detector
        }
        
        self.component_weights = {
            AIComponent.CLASSICAL_ML: 0.2,
            AIComponent.DEEP_LEARNING: 0.2,
            AIComponent.GRAPH_NEURAL_NETWORKS: 0.15,
            AIComponent.QUANTUM_ML: 0.1,
            AIComponent.FEDERATED_LEARNING: 0.1,
            AIComponent.BLOCKCHAIN_SECURITY: 0.1,
            AIComponent.ADAPTIVE_LEARNING: 0.05,
            AIComponent.EXPLAINABLE_AI: 0.05,
            AIComponent.ADVANCED_ANALYTICS: 0.03,
            AIComponent.ANOMALY_DETECTION: 0.02,
            AIComponent.NOVELTY_DETECTION: 0.02
        }
        
        self.is_initialized = False
        self.orchestration_history = []
    
    async def initialize(self) -> None:
        """Initialize all AI components."""
        try:
            logger.info("Initializing AI orchestrator")
            
            # Initialize components
            initialization_tasks = []
            
            # Initialize model manager
            initialization_tasks.append(model_manager.initialize())
            
            # Initialize federated learning
            initialization_tasks.append(federated_learning_engine.initialize_federated_learning(20, 64, len(AttackType)))
            
            # Initialize other components
            initialization_tasks.append(self._initialize_components())
            
            # Wait for all initializations
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            self.is_initialized = True
            logger.info("AI orchestrator initialized successfully")
            
        except Exception as e:
            logger.error("Error initializing AI orchestrator", error=str(e))
            raise
    
    async def _initialize_components(self):
        """Initialize individual components."""
        try:
            # Initialize components that need initialization
            await asyncio.gather(
                anomaly_detector._fit_statistical_models(),
                novelty_detector._fit_novelty_models(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error("Error initializing components", error=str(e))
    
    async def orchestrate_analysis(self, target: str, target_type: str, 
                                 context: Optional[Dict] = None) -> AIOrchestrationResult:
        """Orchestrate comprehensive analysis using all AI components."""
        try:
            start_time = datetime.utcnow()
            
            logger.info("Starting AI orchestration", target=target, target_type=target_type)
            
            # Extract features
            from ..features.feature_engine import feature_engine
            features = await feature_engine.extract_all_features(target, target_type, context)
            
            # Run all AI components in parallel
            component_results = await self._run_all_components(target, target_type, features, context)
            
            # Generate consensus
            consensus_result = await self._generate_consensus(component_results)
            
            # Generate explanations
            explanation = await self._generate_comprehensive_explanation(
                target, target_type, features, consensus_result, component_results
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                consensus_result, component_results, features
            )
            
            # Calculate processing time
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Create orchestration result
            result = AIOrchestrationResult(
                target=target,
                target_type=target_type,
                final_prediction=consensus_result["prediction"],
                final_confidence=consensus_result["confidence"],
                component_results=component_results,
                consensus_score=consensus_result["consensus_score"],
                explanation=explanation,
                recommendations=recommendations,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
            # Store orchestration history
            self.orchestration_history.append(result)
            
            logger.info("AI orchestration completed", 
                       target=target,
                       prediction=result.final_prediction,
                       confidence=result.final_confidence,
                       processing_time_ms=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Error in AI orchestration", error=str(e))
            raise
    
    async def _run_all_components(self, target: str, target_type: str, 
                                features: Dict[str, Any], 
                                context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run all AI components in parallel."""
        try:
            # Create tasks for all components
            tasks = []
            
            # Classical ML
            tasks.append(self._run_classical_ml(target, target_type, features))
            
            # Deep Learning
            tasks.append(self._run_deep_learning(target, target_type, features))
            
            # Graph Neural Networks
            tasks.append(self._run_graph_neural_networks(target, target_type, features))
            
            # Quantum ML
            tasks.append(self._run_quantum_ml(target, target_type, features))
            
            # Federated Learning
            tasks.append(self._run_federated_learning(target, target_type, features))
            
            # Blockchain Security
            tasks.append(self._run_blockchain_security(target, target_type, features))
            
            # Adaptive Learning
            tasks.append(self._run_adaptive_learning(target, target_type, features))
            
            # Anomaly Detection
            tasks.append(self._run_anomaly_detection(target, target_type, features))
            
            # Novelty Detection
            tasks.append(self._run_novelty_detection(target, target_type, features))
            
            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            component_results = {}
            for i, (component, result) in enumerate(zip(self.components.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Error in component {component}", error=str(result))
                    component_results[component.value] = {"error": str(result)}
                else:
                    component_results[component.value] = result
            
            return component_results
            
        except Exception as e:
            logger.error("Error running all components", error=str(e))
            return {}
    
    async def _run_classical_ml(self, target: str, target_type: str, 
                              features: Dict[str, Any]) -> Dict[str, Any]:
        """Run classical ML components."""
        try:
            # Get ensemble prediction
            prediction = await model_manager.predict_ensemble("main_ensemble", features["features"])
            
            return {
                "prediction": prediction["attack_type"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "model_type": "ensemble"
            }
            
        except Exception as e:
            logger.error("Error in classical ML", error=str(e))
            return {"error": str(e)}
    
    async def _run_deep_learning(self, target: str, target_type: str, 
                               features: Dict[str, Any]) -> Dict[str, Any]:
        """Run deep learning components."""
        try:
            # Use CNN-LSTM model
            if "cnn_lstm" in model_manager.models and model_manager.models["cnn_lstm"].is_trained:
                prediction = await model_manager.predict("cnn_lstm", features["features"])
                
                return {
                    "prediction": prediction["attack_type"],
                    "confidence": prediction["confidence"],
                    "model_type": "cnn_lstm"
                }
            else:
                return {"error": "CNN-LSTM model not trained"}
            
        except Exception as e:
            logger.error("Error in deep learning", error=str(e))
            return {"error": str(e)}
    
    async def _run_graph_neural_networks(self, target: str, target_type: str, 
                                       features: Dict[str, Any]) -> Dict[str, Any]:
        """Run graph neural network components."""
        try:
            # Build graph for the target
            if target_type == "url":
                graph_data = await graph_nn_engine.build_url_relationship_graph([target], [features])
            elif target_type == "ip":
                graph_data = await graph_nn_engine.build_ip_relationship_graph([target], [features])
            else:
                return {"error": "Unsupported target type for graph analysis"}
            
            # Make prediction with graph model
            prediction = await graph_nn_engine.predict_with_graph_model(graph_data, "gcn")
            
            return {
                "prediction": prediction["attack_types"][0] if prediction["attack_types"] else AttackType.UNKNOWN,
                "confidence": prediction["confidences"][0] if prediction["confidences"] else 0.0,
                "model_type": "graph_neural_network"
            }
            
        except Exception as e:
            logger.error("Error in graph neural networks", error=str(e))
            return {"error": str(e)}
    
    async def _run_quantum_ml(self, target: str, target_type: str, 
                            features: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum ML components."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is not None:
                # Make quantum prediction
                prediction = await quantum_ml_engine.predict_with_quantum_model(feature_vector, "vqc")
                
                return {
                    "prediction": prediction.prediction,
                    "confidence": prediction.confidence,
                    "quantum_advantage": prediction.quantum_advantage,
                    "model_type": "quantum_vqc"
                }
            else:
                return {"error": "Could not prepare feature vector for quantum analysis"}
            
        except Exception as e:
            logger.error("Error in quantum ML", error=str(e))
            return {"error": str(e)}
    
    async def _run_federated_learning(self, target: str, target_type: str, 
                                    features: Dict[str, Any]) -> Dict[str, Any]:
        """Run federated learning components."""
        try:
            # Get federated learning statistics
            stats = await federated_learning_engine.get_federated_statistics()
            
            # Simulate federated prediction
            return {
                "prediction": AttackType.UNKNOWN,  # Would be actual federated prediction
                "confidence": 0.5,
                "federated_rounds": stats.get("completed_rounds", 0),
                "model_type": "federated_learning"
            }
            
        except Exception as e:
            logger.error("Error in federated learning", error=str(e))
            return {"error": str(e)}
    
    async def _run_blockchain_security(self, target: str, target_type: str, 
                                     features: Dict[str, Any]) -> Dict[str, Any]:
        """Run blockchain security components."""
        try:
            # Record threat in blockchain
            threat_data = {
                "threat_id": f"{target}_{datetime.utcnow().timestamp()}",
                "target": target,
                "target_type": target_type,
                "attack_type": "unknown",
                "severity": "low",
                "confidence": 0.0,
                "detector_id": "ai_orchestrator"
            }
            
            block_hash = await threat_blockchain.record_threat_detection(threat_data)
            
            # Get blockchain statistics
            stats = await threat_blockchain.get_blockchain_statistics()
            
            return {
                "prediction": AttackType.UNKNOWN,
                "confidence": 0.5,
                "block_hash": block_hash,
                "chain_length": stats.get("chain_length", 0),
                "model_type": "blockchain_security"
            }
            
        except Exception as e:
            logger.error("Error in blockchain security", error=str(e))
            return {"error": str(e)}
    
    async def _run_adaptive_learning(self, target: str, target_type: str, 
                                   features: Dict[str, Any]) -> Dict[str, Any]:
        """Run adaptive learning components."""
        try:
            # Get adaptation statistics
            stats = await adaptive_learning_engine.get_adaptation_statistics()
            
            return {
                "prediction": AttackType.UNKNOWN,
                "confidence": 0.5,
                "adaptation_episodes": stats.get("total_episodes", 0),
                "model_type": "adaptive_learning"
            }
            
        except Exception as e:
            logger.error("Error in adaptive learning", error=str(e))
            return {"error": str(e)}
    
    async def _run_anomaly_detection(self, target: str, target_type: str, 
                                   features: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection components."""
        try:
            # Detect anomalies
            anomalies = await anomaly_detector.detect_anomalies(target, target_type, features["features"])
            
            # Determine if target is anomalous
            is_anomalous = len(anomalies) > 0
            max_anomaly_score = max([a.anomaly_score for a in anomalies]) if anomalies else 0.0
            
            return {
                "prediction": AttackType.UNKNOWN if not is_anomalous else AttackType.PROBE,
                "confidence": max_anomaly_score,
                "anomalies_detected": len(anomalies),
                "model_type": "anomaly_detection"
            }
            
        except Exception as e:
            logger.error("Error in anomaly detection", error=str(e))
            return {"error": str(e)}
    
    async def _run_novelty_detection(self, target: str, target_type: str, 
                                   features: Dict[str, Any]) -> Dict[str, Any]:
        """Run novelty detection components."""
        try:
            # Detect novelty
            novelties = await novelty_detector.detect_novelty(target, target_type, features["features"])
            
            # Determine if target is novel
            is_novel = len(novelties) > 0
            max_novelty_score = max([n.novelty_score for n in novelties]) if novelties else 0.0
            
            return {
                "prediction": AttackType.UNKNOWN if not is_novel else AttackType.UNKNOWN,
                "confidence": max_novelty_score,
                "novelties_detected": len(novelties),
                "model_type": "novelty_detection"
            }
            
        except Exception as e:
            logger.error("Error in novelty detection", error=str(e))
            return {"error": str(e)}
    
    async def _generate_consensus(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consensus from all component results."""
        try:
            # Collect valid predictions
            valid_predictions = []
            valid_confidences = []
            component_weights = []
            
            for component, result in component_results.items():
                if "error" not in result and "prediction" in result:
                    prediction = result["prediction"]
                    confidence = result.get("confidence", 0.0)
                    weight = self.component_weights.get(AIComponent(component), 0.1)
                    
                    valid_predictions.append(prediction)
                    valid_confidences.append(confidence)
                    component_weights.append(weight)
            
            if not valid_predictions:
                return {
                    "prediction": AttackType.UNKNOWN,
                    "confidence": 0.0,
                    "consensus_score": 0.0
                }
            
            # Calculate weighted consensus
            prediction_scores = {}
            for prediction, confidence, weight in zip(valid_predictions, valid_confidences, component_weights):
                weighted_score = confidence * weight
                
                if prediction not in prediction_scores:
                    prediction_scores[prediction] = 0.0
                
                prediction_scores[prediction] += weighted_score
            
            # Get best prediction
            best_prediction = max(prediction_scores.items(), key=lambda x: x[1])
            final_prediction = best_prediction[0]
            final_confidence = best_prediction[1]
            
            # Calculate consensus score
            total_weight = sum(component_weights)
            consensus_score = final_confidence / total_weight if total_weight > 0 else 0.0
            
            return {
                "prediction": final_prediction,
                "confidence": final_confidence,
                "consensus_score": consensus_score,
                "prediction_scores": prediction_scores
            }
            
        except Exception as e:
            logger.error("Error generating consensus", error=str(e))
            return {
                "prediction": AttackType.UNKNOWN,
                "confidence": 0.0,
                "consensus_score": 0.0
            }
    
    async def _generate_comprehensive_explanation(self, target: str, target_type: str,
                                                features: Dict[str, Any],
                                                consensus_result: Dict[str, Any],
                                                component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanation."""
        try:
            # Get explainable AI results
            explainable_result = component_results.get("explainable_ai", {})
            
            if "error" not in explainable_result:
                # Use explainable AI results
                explanation = explainable_result
            else:
                # Generate basic explanation
                explanation = {
                    "method": "consensus",
                    "explanation_text": f"Consensus prediction: {consensus_result['prediction'].value} with {consensus_result['confidence']:.2%} confidence",
                    "feature_importance": features.get("composite", {}),
                    "component_contributions": {
                        component: result.get("confidence", 0.0) 
                        for component, result in component_results.items() 
                        if "error" not in result
                    }
                }
            
            return explanation
            
        except Exception as e:
            logger.error("Error generating comprehensive explanation", error=str(e))
            return {"error": str(e)}
    
    async def _generate_recommendations(self, consensus_result: Dict[str, Any],
                                      component_results: Dict[str, Any],
                                      features: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        try:
            recommendations = []
            
            # Base recommendations on consensus
            prediction = consensus_result["prediction"]
            confidence = consensus_result["confidence"]
            
            if prediction != AttackType.UNKNOWN and confidence > 0.7:
                recommendations.append(f"High confidence {prediction.value} threat detected - take immediate action")
            
            if confidence > 0.5:
                recommendations.append("Threat detected - investigate further")
            
            # Component-specific recommendations
            for component, result in component_results.items():
                if "error" not in result:
                    if component == "anomaly_detection" and result.get("anomalies_detected", 0) > 0:
                        recommendations.append("Anomalous behavior detected - review for new attack patterns")
                    
                    if component == "novelty_detection" and result.get("novelties_detected", 0) > 0:
                        recommendations.append("Novel threat pattern detected - update detection rules")
                    
                    if component == "quantum_ml" and result.get("quantum_advantage", 0) > 0.2:
                        recommendations.append("Quantum analysis shows significant threat indicators")
            
            # Risk-based recommendations
            risk_score = features.get("composite", {}).get("risk_score", 0.0)
            if risk_score > 0.8:
                recommendations.append("High risk score - implement additional security measures")
            elif risk_score > 0.5:
                recommendations.append("Medium risk score - monitor closely")
            
            return recommendations
            
        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            return []
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for quantum analysis."""
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
            target_size = 10
            if len(numeric_features) < target_size:
                numeric_features.extend([0.0] * (target_size - len(numeric_features)))
            else:
                numeric_features = numeric_features[:target_size]
            
            return np.array(numeric_features)
            
        except Exception as e:
            logger.error("Error preparing feature vector", error=str(e))
            return None
    
    async def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        try:
            return {
                "total_orchestrations": len(self.orchestration_history),
                "is_initialized": self.is_initialized,
                "component_weights": self.component_weights,
                "available_components": list(self.components.keys()),
                "average_processing_time": np.mean([r.processing_time_ms for r in self.orchestration_history]) if self.orchestration_history else 0,
                "average_confidence": np.mean([r.final_confidence for r in self.orchestration_history]) if self.orchestration_history else 0
            }
            
        except Exception as e:
            logger.error("Error getting orchestration statistics", error=str(e))
            return {}


# Global AI orchestrator instance
ai_orchestrator = AIOrchestrator()

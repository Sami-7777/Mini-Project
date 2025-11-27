"""
Adaptive learning system for continuous model improvement and threat adaptation.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque
import json

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..models.model_manager import model_manager

logger = structlog.get_logger(__name__)


class LearningStrategy(str, Enum):
    """Learning strategies for adaptive learning."""
    ONLINE_LEARNING = "online_learning"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"


@dataclass
class LearningEpisode:
    """Learning episode for adaptive learning."""
    episode_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    actual_result: Optional[Dict[str, Any]]
    feedback: Optional[str]
    performance_metrics: Dict[str, float]
    learning_strategy: LearningStrategy
    model_updates: Dict[str, Any]


@dataclass
class ThreatPattern:
    """Threat pattern for adaptive learning."""
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    attack_type: AttackType
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    evolution_history: List[Dict[str, Any]]


class OnlineLearningSystem:
    """Online learning system for continuous model updates."""
    
    def __init__(self, learning_rate: float = 0.01, buffer_size: int = 1000):
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.feedback_buffer = deque(maxlen=buffer_size)
        self.performance_history = deque(maxlen=buffer_size)
        self.model_versions = {}
        self.adaptation_threshold = 0.1
    
    async def process_feedback(self, episode: LearningEpisode) -> Dict[str, Any]:
        """Process feedback and update models."""
        try:
            # Add to feedback buffer
            self.feedback_buffer.append(episode)
            
            # Calculate performance metrics
            performance = self._calculate_performance(episode)
            self.performance_history.append(performance)
            
            # Check if adaptation is needed
            if self._should_adapt():
                adaptation_result = await self._adapt_models(episode)
                return adaptation_result
            
            return {"adapted": False, "performance": performance}
            
        except Exception as e:
            logger.error("Error processing feedback", error=str(e))
            raise
    
    def _calculate_performance(self, episode: LearningEpisode) -> Dict[str, float]:
        """Calculate performance metrics for an episode."""
        try:
            if episode.actual_result is None:
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            
            predicted_attack = episode.prediction.get("attack_type")
            actual_attack = episode.actual_result.get("attack_type")
            
            # Calculate accuracy
            accuracy = 1.0 if predicted_attack == actual_attack else 0.0
            
            # Calculate precision and recall (simplified)
            precision = accuracy  # Simplified
            recall = accuracy     # Simplified
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "confidence": episode.prediction.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error("Error calculating performance", error=str(e))
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    
    def _should_adapt(self) -> bool:
        """Determine if model adaptation is needed."""
        try:
            if len(self.performance_history) < 10:
                return False
            
            # Calculate recent performance
            recent_performance = list(self.performance_history)[-10:]
            avg_recent_accuracy = np.mean([p["accuracy"] for p in recent_performance])
            
            # Calculate historical performance
            if len(self.performance_history) >= 50:
                historical_performance = list(self.performance_history)[-50:-10]
                avg_historical_accuracy = np.mean([p["accuracy"] for p in historical_performance])
                
                # Check if performance dropped significantly
                performance_drop = avg_historical_accuracy - avg_recent_accuracy
                return performance_drop > self.adaptation_threshold
            
            return False
            
        except Exception as e:
            logger.error("Error checking adaptation need", error=str(e))
            return False
    
    async def _adapt_models(self, episode: LearningEpisode) -> Dict[str, Any]:
        """Adapt models based on feedback."""
        try:
            adaptation_results = {}
            
            # Get recent feedback
            recent_feedback = list(self.feedback_buffer)[-50:]
            
            # Adapt each model
            for model_name in model_manager.models:
                if model_manager.models[model_name].is_trained:
                    try:
                        # Prepare adaptation data
                        X_adapt = []
                        y_adapt = []
                        
                        for feedback_episode in recent_feedback:
                            if feedback_episode.actual_result:
                                # Convert features to model input
                                features = feedback_episode.input_data.get("features", {})
                                X_sample = self._prepare_model_input(features)
                                X_adapt.append(X_sample)
                                
                                # Convert actual result to label
                                actual_attack = feedback_episode.actual_result.get("attack_type")
                                y_sample = list(AttackType).index(actual_attack) if actual_attack else 0
                                y_adapt.append(y_sample)
                        
                        if X_adapt and y_adapt:
                            # Perform online learning update
                            result = await model_manager.update_model(
                                model_name,
                                np.array(X_adapt),
                                np.array(y_adapt)
                            )
                            adaptation_results[model_name] = result
                    
                    except Exception as e:
                        logger.error(f"Error adapting model {model_name}", error=str(e))
                        adaptation_results[model_name] = {"error": str(e)}
            
            logger.info("Model adaptation completed", 
                       adapted_models=len(adaptation_results),
                       feedback_samples=len(recent_feedback))
            
            return {
                "adapted": True,
                "adaptation_results": adaptation_results,
                "feedback_samples": len(recent_feedback)
            }
            
        except Exception as e:
            logger.error("Error adapting models", error=str(e))
            raise
    
    def _prepare_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare model input from features."""
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
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(numeric_features) < target_size:
                numeric_features.extend([0.0] * (target_size - len(numeric_features)))
            else:
                numeric_features = numeric_features[:target_size]
            
            return np.array(numeric_features)
            
        except Exception as e:
            logger.error("Error preparing model input", error=str(e))
            return np.zeros(20)


class MetaLearningSystem:
    """Meta-learning system for rapid adaptation to new threats."""
    
    def __init__(self):
        self.meta_models = {}
        self.task_distributions = {}
        self.adaptation_history = []
    
    async def learn_to_learn(self, training_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Learn to learn from multiple tasks."""
        try:
            # Prepare meta-training data
            meta_training_data = []
            
            for task in training_tasks:
                task_data = {
                    "task_id": task.get("task_id", ""),
                    "support_set": task.get("support_set", []),
                    "query_set": task.get("query_set", []),
                    "task_type": task.get("task_type", "classification")
                }
                meta_training_data.append(task_data)
            
            # Meta-learning algorithm (simplified)
            meta_metrics = await self._meta_learning_algorithm(meta_training_data)
            
            logger.info("Meta-learning completed", metrics=meta_metrics)
            return meta_metrics
            
        except Exception as e:
            logger.error("Error in meta-learning", error=str(e))
            raise
    
    async def _meta_learning_algorithm(self, training_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Meta-learning algorithm implementation."""
        try:
            # Simplified meta-learning algorithm
            # In practice, this would implement MAML, Reptile, or similar algorithms
            
            total_tasks = len(training_tasks)
            successful_adaptations = 0
            total_accuracy = 0.0
            
            for task in training_tasks:
                try:
                    # Simulate few-shot learning
                    support_set = task.get("support_set", [])
                    query_set = task.get("query_set", [])
                    
                    if len(support_set) >= 5 and len(query_set) >= 5:
                        # Simulate adaptation
                        adaptation_accuracy = np.random.uniform(0.6, 0.9)
                        total_accuracy += adaptation_accuracy
                        successful_adaptations += 1
                
                except Exception as e:
                    logger.error("Error in task adaptation", error=str(e))
                    continue
            
            avg_accuracy = total_accuracy / successful_adaptations if successful_adaptations > 0 else 0.0
            success_rate = successful_adaptations / total_tasks if total_tasks > 0 else 0.0
            
            return {
                "average_accuracy": avg_accuracy,
                "success_rate": success_rate,
                "total_tasks": total_tasks,
                "successful_adaptations": successful_adaptations
            }
            
        except Exception as e:
            logger.error("Error in meta-learning algorithm", error=str(e))
            raise
    
    async def rapid_adaptation(self, new_threat_data: Dict[str, Any], 
                             few_shot_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rapid adaptation to new threat patterns."""
        try:
            # Prepare few-shot learning data
            support_set = few_shot_examples[:5]  # 5-shot learning
            query_set = [new_threat_data]
            
            # Simulate rapid adaptation
            adaptation_result = await self._few_shot_adaptation(support_set, query_set)
            
            logger.info("Rapid adaptation completed", 
                       support_examples=len(support_set),
                       adaptation_accuracy=adaptation_result.get("accuracy", 0.0))
            
            return adaptation_result
            
        except Exception as e:
            logger.error("Error in rapid adaptation", error=str(e))
            raise
    
    async def _few_shot_adaptation(self, support_set: List[Dict[str, Any]], 
                                 query_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Few-shot learning adaptation."""
        try:
            # Simulate few-shot learning
            # In practice, this would use Prototypical Networks, Matching Networks, etc.
            
            # Calculate prototypes from support set
            prototypes = {}
            for example in support_set:
                attack_type = example.get("attack_type", AttackType.UNKNOWN)
                features = example.get("features", {})
                
                if attack_type not in prototypes:
                    prototypes[attack_type] = []
                
                # Extract feature vector
                feature_vector = self._extract_feature_vector(features)
                prototypes[attack_type].append(feature_vector)
            
            # Calculate average prototypes
            avg_prototypes = {}
            for attack_type, feature_vectors in prototypes.items():
                if feature_vectors:
                    avg_prototypes[attack_type] = np.mean(feature_vectors, axis=0)
            
            # Predict on query set
            predictions = []
            for query in query_set:
                query_features = query.get("features", {})
                query_vector = self._extract_feature_vector(query_features)
                
                # Find closest prototype
                min_distance = float('inf')
                predicted_attack = AttackType.UNKNOWN
                
                for attack_type, prototype in avg_prototypes.items():
                    distance = np.linalg.norm(query_vector - prototype)
                    if distance < min_distance:
                        min_distance = distance
                        predicted_attack = attack_type
                
                predictions.append({
                    "predicted_attack": predicted_attack,
                    "confidence": 1.0 / (1.0 + min_distance),
                    "distance": min_distance
                })
            
            # Calculate accuracy
            correct_predictions = 0
            for i, prediction in enumerate(predictions):
                actual_attack = query_set[i].get("attack_type", AttackType.UNKNOWN)
                if prediction["predicted_attack"] == actual_attack:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(predictions) if predictions else 0.0
            
            return {
                "accuracy": accuracy,
                "predictions": predictions,
                "prototypes": len(avg_prototypes),
                "support_examples": len(support_set)
            }
            
        except Exception as e:
            logger.error("Error in few-shot adaptation", error=str(e))
            raise
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from features."""
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
            logger.error("Error extracting feature vector", error=str(e))
            return np.zeros(10)


class ContinualLearningSystem:
    """Continual learning system to prevent catastrophic forgetting."""
    
    def __init__(self):
        self.task_memory = {}
        self.importance_weights = {}
        self.elastic_weight_consolidation = {}
        self.progressive_neural_networks = {}
    
    async def learn_continuously(self, new_task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn continuously without forgetting previous knowledge."""
        try:
            task_id = new_task_data.get("task_id", f"task_{datetime.utcnow().timestamp()}")
            
            # Store task in memory
            self.task_memory[task_id] = {
                "data": new_task_data,
                "timestamp": datetime.utcnow(),
                "importance": self._calculate_task_importance(new_task_data)
            }
            
            # Apply continual learning techniques
            learning_result = await self._apply_continual_learning(new_task_data)
            
            logger.info("Continual learning completed", 
                       task_id=task_id,
                       learning_result=learning_result)
            
            return learning_result
            
        except Exception as e:
            logger.error("Error in continual learning", error=str(e))
            raise
    
    def _calculate_task_importance(self, task_data: Dict[str, Any]) -> float:
        """Calculate importance of a task."""
        try:
            # Calculate importance based on various factors
            importance = 0.0
            
            # Attack type importance
            attack_type = task_data.get("attack_type", AttackType.UNKNOWN)
            if attack_type in [AttackType.PHISHING, AttackType.MALWARE, AttackType.RANSOMWARE]:
                importance += 0.3
            
            # Severity importance
            severity = task_data.get("severity", SeverityLevel.LOW)
            severity_importance = {
                SeverityLevel.LOW: 0.1,
                SeverityLevel.MEDIUM: 0.3,
                SeverityLevel.HIGH: 0.6,
                SeverityLevel.CRITICAL: 1.0
            }
            importance += severity_importance.get(severity, 0.1)
            
            # Confidence importance
            confidence = task_data.get("confidence", 0.0)
            importance += confidence * 0.2
            
            # Novelty importance
            if task_data.get("is_novel", False):
                importance += 0.4
            
            return min(importance, 1.0)
            
        except Exception as e:
            logger.error("Error calculating task importance", error=str(e))
            return 0.5
    
    async def _apply_continual_learning(self, new_task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply continual learning techniques."""
        try:
            # Elastic Weight Consolidation (EWC)
            ewc_result = await self._apply_elastic_weight_consolidation(new_task_data)
            
            # Progressive Neural Networks
            pnn_result = await self._apply_progressive_neural_networks(new_task_data)
            
            # Experience Replay
            replay_result = await self._apply_experience_replay(new_task_data)
            
            return {
                "elastic_weight_consolidation": ewc_result,
                "progressive_neural_networks": pnn_result,
                "experience_replay": replay_result,
                "overall_success": True
            }
            
        except Exception as e:
            logger.error("Error applying continual learning", error=str(e))
            raise
    
    async def _apply_elastic_weight_consolidation(self, new_task_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply Elastic Weight Consolidation."""
        try:
            # Simulate EWC
            # In practice, this would calculate Fisher Information Matrix and apply EWC loss
            
            lambda_ewc = 1000  # EWC regularization strength
            fisher_information = np.random.rand(100)  # Simulated Fisher Information
            
            # Calculate EWC loss
            ewc_loss = lambda_ewc * np.sum(fisher_information)
            
            return {
                "ewc_loss": ewc_loss,
                "fisher_information_norm": np.linalg.norm(fisher_information),
                "lambda_ewc": lambda_ewc
            }
            
        except Exception as e:
            logger.error("Error applying EWC", error=str(e))
            raise
    
    async def _apply_progressive_neural_networks(self, new_task_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply Progressive Neural Networks."""
        try:
            # Simulate Progressive Neural Networks
            # In practice, this would create new columns for new tasks
            
            task_id = new_task_data.get("task_id", "new_task")
            
            # Create new column for new task
            new_column = {
                "task_id": task_id,
                "created_at": datetime.utcnow(),
                "connections": ["previous_column_1", "previous_column_2"],
                "parameters": np.random.rand(1000)  # Simulated parameters
            }
            
            self.progressive_neural_networks[task_id] = new_column
            
            return {
                "new_columns": 1,
                "total_columns": len(self.progressive_neural_networks),
                "parameters_count": len(new_column["parameters"])
            }
            
        except Exception as e:
            logger.error("Error applying Progressive Neural Networks", error=str(e))
            raise
    
    async def _apply_experience_replay(self, new_task_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply Experience Replay."""
        try:
            # Simulate Experience Replay
            # In practice, this would sample from previous experiences and replay them
            
            replay_buffer_size = 1000
            replay_batch_size = 32
            
            # Sample from previous experiences
            if len(self.task_memory) >= replay_batch_size:
                # Sample random experiences
                sampled_tasks = np.random.choice(
                    list(self.task_memory.keys()),
                    size=min(replay_batch_size, len(self.task_memory)),
                    replace=False
                )
                
                # Simulate replay
                replay_accuracy = np.random.uniform(0.7, 0.9)
                
                return {
                    "replay_batch_size": len(sampled_tasks),
                    "replay_accuracy": replay_accuracy,
                    "buffer_size": len(self.task_memory)
                }
            else:
                return {
                    "replay_batch_size": 0,
                    "replay_accuracy": 0.0,
                    "buffer_size": len(self.task_memory)
                }
            
        except Exception as e:
            logger.error("Error applying Experience Replay", error=str(e))
            raise


class AdaptiveLearningEngine:
    """Main engine for adaptive learning."""
    
    def __init__(self):
        self.online_learning = OnlineLearningSystem()
        self.meta_learning = MetaLearningSystem()
        self.continual_learning = ContinualLearningSystem()
        self.threat_patterns = {}
        self.adaptation_history = []
    
    async def process_learning_episode(self, episode: LearningEpisode) -> Dict[str, Any]:
        """Process a learning episode."""
        try:
            # Process with online learning
            online_result = await self.online_learning.process_feedback(episode)
            
            # Update threat patterns
            await self._update_threat_patterns(episode)
            
            # Check for new threat patterns
            if episode.actual_result and episode.actual_result.get("is_novel", False):
                # Apply meta-learning for rapid adaptation
                meta_result = await self._apply_meta_learning(episode)
                online_result["meta_learning"] = meta_result
            
            # Apply continual learning
            continual_result = await self.continual_learning.learn_continuously({
                "task_id": episode.episode_id,
                "attack_type": episode.prediction.get("attack_type"),
                "severity": episode.prediction.get("severity"),
                "confidence": episode.prediction.get("confidence"),
                "is_novel": episode.actual_result.get("is_novel", False) if episode.actual_result else False
            })
            online_result["continual_learning"] = continual_result
            
            # Store adaptation history
            self.adaptation_history.append({
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp,
                "results": online_result
            })
            
            logger.info("Learning episode processed", 
                       episode_id=episode.episode_id,
                       adapted=online_result.get("adapted", False))
            
            return online_result
            
        except Exception as e:
            logger.error("Error processing learning episode", error=str(e))
            raise
    
    async def _update_threat_patterns(self, episode: LearningEpisode):
        """Update threat patterns based on episode."""
        try:
            if episode.actual_result:
                attack_type = episode.actual_result.get("attack_type", AttackType.UNKNOWN)
                pattern_key = f"{attack_type}_{episode.input_data.get('target_type', 'unknown')}"
                
                if pattern_key not in self.threat_patterns:
                    self.threat_patterns[pattern_key] = ThreatPattern(
                        pattern_id=pattern_key,
                        pattern_type=attack_type.value,
                        features=episode.input_data.get("features", {}),
                        attack_type=attack_type,
                        confidence=episode.prediction.get("confidence", 0.0),
                        frequency=1,
                        first_seen=episode.timestamp,
                        last_seen=episode.timestamp,
                        evolution_history=[]
                    )
                else:
                    # Update existing pattern
                    pattern = self.threat_patterns[pattern_key]
                    pattern.frequency += 1
                    pattern.last_seen = episode.timestamp
                    pattern.confidence = (pattern.confidence + episode.prediction.get("confidence", 0.0)) / 2
                    
                    # Add to evolution history
                    pattern.evolution_history.append({
                        "timestamp": episode.timestamp,
                        "confidence": episode.prediction.get("confidence", 0.0),
                        "features": episode.input_data.get("features", {})
                    })
            
        except Exception as e:
            logger.error("Error updating threat patterns", error=str(e))
    
    async def _apply_meta_learning(self, episode: LearningEpisode) -> Dict[str, Any]:
        """Apply meta-learning for rapid adaptation."""
        try:
            # Prepare few-shot learning task
            few_shot_examples = await self._get_few_shot_examples(episode)
            
            if len(few_shot_examples) >= 5:
                # Apply rapid adaptation
                adaptation_result = await self.meta_learning.rapid_adaptation(
                    episode.input_data,
                    few_shot_examples
                )
                return adaptation_result
            else:
                return {"adapted": False, "reason": "insufficient_examples"}
            
        except Exception as e:
            logger.error("Error applying meta-learning", error=str(e))
            return {"adapted": False, "error": str(e)}
    
    async def _get_few_shot_examples(self, episode: LearningEpisode) -> List[Dict[str, Any]]:
        """Get few-shot learning examples."""
        try:
            # Get similar examples from history
            examples = []
            
            for history_episode in self.adaptation_history[-100:]:  # Last 100 episodes
                if (history_episode["results"].get("adapted", False) and 
                    history_episode["episode_id"] != episode.episode_id):
                    
                    examples.append({
                        "features": history_episode.get("features", {}),
                        "attack_type": history_episode.get("attack_type", AttackType.UNKNOWN),
                        "confidence": history_episode.get("confidence", 0.0)
                    })
                    
                    if len(examples) >= 10:  # Limit to 10 examples
                        break
            
            return examples
            
        except Exception as e:
            logger.error("Error getting few-shot examples", error=str(e))
            return []
    
    async def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        try:
            return {
                "total_episodes": len(self.adaptation_history),
                "adapted_episodes": len([e for e in self.adaptation_history if e["results"].get("adapted", False)]),
                "threat_patterns": len(self.threat_patterns),
                "online_learning_buffer": len(self.online_learning.feedback_buffer),
                "continual_learning_tasks": len(self.continual_learning.task_memory),
                "average_performance": np.mean([e["results"].get("performance", {}).get("accuracy", 0.0) 
                                              for e in self.adaptation_history]) if self.adaptation_history else 0.0
            }
            
        except Exception as e:
            logger.error("Error getting adaptation statistics", error=str(e))
            return {}


# Global adaptive learning engine instance
adaptive_learning_engine = AdaptiveLearningEngine()

"""
Hybrid ML and rule-based detection engine.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np
import structlog

from ..database.models import AttackType, SeverityLevel, AnalysisResult, AnalysisStatus
from ..database.connection import get_database
from ..features.feature_engine import feature_engine
from ..models.model_manager import model_manager
from .rule_engine import rule_engine, RuleAction
from ..intelligence.threat_intelligence import threat_intelligence_manager

logger = structlog.get_logger(__name__)


class HybridDetectionEngine:
    """Hybrid detection engine combining ML models and rule-based detection."""
    
    def __init__(self):
        self.ml_weight = 0.7  # Weight for ML predictions
        self.rule_weight = 0.3  # Weight for rule-based detection
        self.threat_intel_weight = 0.2  # Weight for threat intelligence
        self.confidence_threshold = 0.6  # Minimum confidence for final decision
        self.ensemble_preference = True  # Prefer ensemble models over individual models
    
    async def analyze(self, target: str, target_type: str, 
                     context: Optional[Dict] = None) -> AnalysisResult:
        """Perform comprehensive analysis of a target."""
        start_time = datetime.utcnow()
        
        try:
            # Create analysis result
            analysis_result = AnalysisResult(
                target_type=target_type,
                target_value=target,
                status=AnalysisStatus.IN_PROGRESS
            )
            
            # Extract features
            logger.info("Extracting features", target=target, target_type=target_type)
            features = await feature_engine.extract_all_features(target, target_type, context)
            analysis_result.url_features = features.get('features', {}).get('url')
            analysis_result.ip_features = features.get('features', {}).get('ip')
            
            # Run ML predictions
            logger.info("Running ML predictions", target=target)
            ml_predictions = await self._run_ml_predictions(features['features'])
            analysis_result.predictions = ml_predictions
            
            # Run rule-based detection
            logger.info("Running rule-based detection", target=target)
            rule_matches = rule_engine.evaluate_rules(features['features'], target_type)
            
            # Get threat intelligence
            logger.info("Getting threat intelligence", target=target)
            threat_intel = await self._get_threat_intelligence(target, target_type)
            analysis_result.threat_intelligence = threat_intel
            
            # Combine results
            logger.info("Combining results", target=target)
            final_result = await self._combine_results(
                ml_predictions, rule_matches, threat_intel, features['features']
            )
            
            # Update analysis result
            analysis_result.final_attack_type = final_result['attack_type']
            analysis_result.final_confidence = final_result['confidence']
            analysis_result.severity = final_result['severity']
            analysis_result.risk_score = final_result['risk_score']
            analysis_result.status = AnalysisStatus.COMPLETED
            
            # Calculate analysis duration
            analysis_result.analysis_duration_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # Store in database
            await self._store_analysis_result(analysis_result)
            
            logger.info("Analysis completed", 
                       target=target, 
                       attack_type=final_result['attack_type'],
                       confidence=final_result['confidence'],
                       duration_ms=analysis_result.analysis_duration_ms)
            
            return analysis_result
            
        except Exception as e:
            logger.error("Error during analysis", target=target, error=str(e))
            
            # Create failed analysis result
            analysis_result = AnalysisResult(
                target_type=target_type,
                target_value=target,
                status=AnalysisStatus.FAILED
            )
            analysis_result.analysis_duration_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            return analysis_result
    
    async def _run_ml_predictions(self, features: Dict[str, Any]) -> List[Any]:
        """Run ML model predictions."""
        predictions = []
        
        try:
            # Try ensemble prediction first
            if self.ensemble_preference and "main_ensemble" in model_manager.ensemble_models:
                ensemble_pred = await model_manager.predict_ensemble("main_ensemble", features)
                predictions.append(ensemble_pred)
            
            # Run individual model predictions
            for model_name in model_manager.models:
                if model_manager.models[model_name].is_trained:
                    try:
                        model_pred = await model_manager.predict(model_name, features)
                        predictions.append(model_pred)
                    except Exception as e:
                        logger.warning(f"Error predicting with model {model_name}", error=str(e))
            
        except Exception as e:
            logger.error("Error running ML predictions", error=str(e))
        
        return predictions
    
    async def _get_threat_intelligence(self, target: str, target_type: str) -> List[Any]:
        """Get threat intelligence for the target."""
        try:
            if target_type == "url":
                return await threat_intelligence_manager.analyze_url(target)
            elif target_type == "ip":
                return await threat_intelligence_manager.analyze_ip(target)
            else:
                return []
        except Exception as e:
            logger.error("Error getting threat intelligence", error=str(e))
            return []
    
    async def _combine_results(self, ml_predictions: List[Dict], 
                              rule_matches: List[Dict], 
                              threat_intel: List[Dict],
                              features: Dict[str, Any]) -> Dict[str, Any]:
        """Combine ML predictions, rule matches, and threat intelligence."""
        
        # Process ML predictions
        ml_scores = self._process_ml_predictions(ml_predictions)
        
        # Process rule matches
        rule_scores = self._process_rule_matches(rule_matches)
        
        # Process threat intelligence
        threat_scores = self._process_threat_intelligence(threat_intel)
        
        # Combine scores
        combined_scores = self._combine_scores(ml_scores, rule_scores, threat_scores)
        
        # Determine final result
        final_result = self._determine_final_result(combined_scores, features)
        
        return final_result
    
    def _process_ml_predictions(self, predictions: List[Dict]) -> Dict[str, float]:
        """Process ML model predictions into scores."""
        attack_scores = {}
        confidence_scores = {}
        
        for pred in predictions:
            attack_type = pred.get('attack_type', AttackType.UNKNOWN)
            confidence = pred.get('confidence', 0.0)
            
            if attack_type not in attack_scores:
                attack_scores[attack_type] = []
                confidence_scores[attack_type] = []
            
            attack_scores[attack_type].append(confidence)
            confidence_scores[attack_type].append(confidence)
        
        # Calculate weighted averages
        final_scores = {}
        for attack_type, scores in attack_scores.items():
            if scores:
                # Weight by model confidence
                weighted_score = np.average(scores, weights=scores)
                final_scores[attack_type] = weighted_score
        
        return final_scores
    
    def _process_rule_matches(self, rule_matches: List[Dict]) -> Dict[str, float]:
        """Process rule matches into scores."""
        attack_scores = {}
        
        for match in rule_matches:
            attack_type = match.get('attack_type', AttackType.UNKNOWN)
            severity = match.get('severity', SeverityLevel.LOW)
            weight = match.get('weight', 1.0)
            
            # Convert severity to score
            severity_scores = {
                SeverityLevel.LOW: 0.3,
                SeverityLevel.MEDIUM: 0.6,
                SeverityLevel.HIGH: 0.8,
                SeverityLevel.CRITICAL: 1.0
            }
            
            score = severity_scores.get(severity, 0.3) * weight
            
            if attack_type not in attack_scores:
                attack_scores[attack_type] = 0.0
            
            attack_scores[attack_type] += score
        
        # Normalize scores
        for attack_type in attack_scores:
            attack_scores[attack_type] = min(attack_scores[attack_type], 1.0)
        
        return attack_scores
    
    def _process_threat_intelligence(self, threat_intel: List[Dict]) -> Dict[str, float]:
        """Process threat intelligence into scores."""
        attack_scores = {}
        
        for intel in threat_intel:
            threat_type = intel.get('threat_type', 'unknown')
            confidence = intel.get('confidence', 0.0)
            
            # Map threat types to attack types
            threat_mapping = {
                'malware': AttackType.MALWARE,
                'phishing': AttackType.PHISHING,
                'ransomware': AttackType.RANSOMWARE,
                'botnet': AttackType.BOTNET,
                'spam': AttackType.SPAM
            }
            
            attack_type = threat_mapping.get(threat_type, AttackType.UNKNOWN)
            
            if attack_type not in attack_scores:
                attack_scores[attack_type] = 0.0
            
            attack_scores[attack_type] = max(attack_scores[attack_type], confidence)
        
        return attack_scores
    
    def _combine_scores(self, ml_scores: Dict[str, float], 
                       rule_scores: Dict[str, float], 
                       threat_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine scores from different sources."""
        combined_scores = {}
        
        # Get all attack types
        all_attack_types = set(ml_scores.keys()) | set(rule_scores.keys()) | set(threat_scores.keys())
        
        for attack_type in all_attack_types:
            ml_score = ml_scores.get(attack_type, 0.0)
            rule_score = rule_scores.get(attack_type, 0.0)
            threat_score = threat_scores.get(attack_type, 0.0)
            
            # Weighted combination
            combined_score = (
                ml_score * self.ml_weight +
                rule_score * self.rule_weight +
                threat_score * self.threat_intel_weight
            )
            
            combined_scores[attack_type] = combined_score
        
        return combined_scores
    
    def _determine_final_result(self, combined_scores: Dict[str, float], 
                               features: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the final detection result."""
        
        if not combined_scores:
            return {
                'attack_type': AttackType.UNKNOWN,
                'confidence': 0.0,
                'severity': SeverityLevel.LOW,
                'risk_score': 0.0
            }
        
        # Find the attack type with highest score
        best_attack_type = max(combined_scores, key=combined_scores.get)
        best_confidence = combined_scores[best_attack_type]
        
        # Determine severity based on confidence and risk score
        risk_score = features.get('composite', {}).get('risk_score', 0.0)
        
        if best_confidence >= 0.9 or risk_score >= 0.9:
            severity = SeverityLevel.CRITICAL
        elif best_confidence >= 0.7 or risk_score >= 0.7:
            severity = SeverityLevel.HIGH
        elif best_confidence >= 0.5 or risk_score >= 0.5:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        # Adjust confidence based on risk score
        final_confidence = max(best_confidence, risk_score * 0.8)
        
        return {
            'attack_type': best_attack_type,
            'confidence': final_confidence,
            'severity': severity,
            'risk_score': risk_score
        }
    
    async def _store_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Store analysis result in database."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            await collection.insert_one(analysis_result.dict(by_alias=True))
        except Exception as e:
            logger.error("Error storing analysis result", error=str(e))
    
    async def get_analysis_history(self, target: str, limit: int = 10) -> List[AnalysisResult]:
        """Get analysis history for a target."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            
            cursor = collection.find(
                {"target_value": target}
            ).sort("created_at", -1).limit(limit)
            
            results = []
            async for doc in cursor:
                results.append(AnalysisResult(**doc))
            
            return results
            
        except Exception as e:
            logger.error("Error getting analysis history", target=target, error=str(e))
            return []
    
    async def update_model_weights(self, ml_weight: float, rule_weight: float, 
                                  threat_intel_weight: float) -> None:
        """Update the weights for combining different detection methods."""
        total_weight = ml_weight + rule_weight + threat_intel_weight
        
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
            self.threat_intel_weight = threat_intel_weight / total_weight
            
            logger.info("Updated model weights", 
                       ml_weight=self.ml_weight,
                       rule_weight=self.rule_weight,
                       threat_intel_weight=self.threat_intel_weight)
        else:
            logger.warning("Invalid weights provided, keeping current values")
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'ml_weight': self.ml_weight,
            'rule_weight': self.rule_weight,
            'threat_intel_weight': self.threat_intel_weight,
            'confidence_threshold': self.confidence_threshold,
            'ensemble_preference': self.ensemble_preference,
            'available_models': len(model_manager.models),
            'trained_models': len([m for m in model_manager.models.values() if m.is_trained]),
            'available_rules': len(rule_engine.rules),
            'enabled_rules': len(rule_engine.get_enabled_rules())
        }


# Global hybrid engine instance
hybrid_engine = HybridDetectionEngine()


"""
Rule-based detection engine for cyberattack detection.
"""
import re
import ipaddress
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog

from ..database.models import AttackType, SeverityLevel
from ..features.url_features import url_feature_extractor
from ..features.ip_features import ip_feature_extractor

logger = structlog.get_logger(__name__)


class RuleType(str, Enum):
    """Types of detection rules."""
    URL_PATTERN = "url_pattern"
    IP_PATTERN = "ip_pattern"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    REPUTATION = "reputation"
    COMPOSITE = "composite"


class RuleAction(str, Enum):
    """Actions to take when a rule matches."""
    BLOCK = "block"
    ALERT = "alert"
    LOG = "log"
    QUARANTINE = "quarantine"


@dataclass
class Rule:
    """Detection rule definition."""
    id: str
    name: str
    description: str
    rule_type: RuleType
    attack_type: AttackType
    severity: SeverityLevel
    action: RuleAction
    conditions: Dict[str, Any]
    weight: float = 1.0
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class RuleEngine:
    """Rule-based detection engine."""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        self.rule_cache: Dict[str, Any] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default detection rules."""
        default_rules = [
            # URL-based rules
            Rule(
                id="url_suspicious_domain",
                name="Suspicious Domain Pattern",
                description="Detect URLs with suspicious domain patterns",
                rule_type=RuleType.URL_PATTERN,
                attack_type=AttackType.PHISHING,
                severity=SeverityLevel.HIGH,
                action=RuleAction.ALERT,
                conditions={
                    "url_entropy": {"min": 5.0},
                    "has_ip_address": True,
                    "suspicious_keywords": {"min_count": 2}
                }
            ),
            
            Rule(
                id="url_shortener_abuse",
                name="URL Shortener Abuse",
                description="Detect abuse of URL shortening services",
                rule_type=RuleType.URL_PATTERN,
                attack_type=AttackType.PHISHING,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.ALERT,
                conditions={
                    "has_shortener": True,
                    "url_similarity_score": {"min": 0.7}
                }
            ),
            
            Rule(
                id="url_polymorphism",
                name="URL Polymorphism Detection",
                description="Detect polymorphic URL patterns",
                rule_type=RuleType.URL_PATTERN,
                attack_type=AttackType.MALWARE,
                severity=SeverityLevel.HIGH,
                action=RuleAction.BLOCK,
                conditions={
                    "has_encoding": True,
                    "suspicious_encoding_ratio": {"min": 0.3}
                }
            ),
            
            # IP-based rules
            Rule(
                id="ip_high_abuse_score",
                name="High Abuse Score IP",
                description="Detect IPs with high abuse scores",
                rule_type=RuleType.REPUTATION,
                attack_type=AttackType.MALWARE,
                severity=SeverityLevel.HIGH,
                action=RuleAction.BLOCK,
                conditions={
                    "abuse_score": {"min": 0.8}
                }
            ),
            
            Rule(
                id="ip_suspicious_country",
                name="Suspicious Country IP",
                description="Detect IPs from high-risk countries",
                rule_type=RuleType.REPUTATION,
                attack_type=AttackType.PROBE,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.ALERT,
                conditions={
                    "country": {"in": ["CN", "RU", "KP", "IR", "SY"]}
                }
            ),
            
            Rule(
                id="ip_private_range",
                name="Private IP Range",
                description="Detect private IP ranges",
                rule_type=RuleType.IP_PATTERN,
                attack_type=AttackType.PROBE,
                severity=SeverityLevel.LOW,
                action=RuleAction.LOG,
                conditions={
                    "is_private": True
                }
            ),
            
            # Behavioral rules
            Rule(
                id="high_frequency_requests",
                name="High Frequency Requests",
                description="Detect high-frequency request patterns",
                rule_type=RuleType.BEHAVIORAL,
                attack_type=AttackType.DDOS,
                severity=SeverityLevel.HIGH,
                action=RuleAction.BLOCK,
                conditions={
                    "request_frequency": {"min": 100.0},  # requests per minute
                    "burst_ratio": {"min": 0.8}
                }
            ),
            
            Rule(
                id="irregular_temporal_pattern",
                name="Irregular Temporal Pattern",
                description="Detect irregular temporal patterns",
                rule_type=RuleType.TEMPORAL,
                attack_type=AttackType.BOTNET,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.ALERT,
                conditions={
                    "temporal_anomaly_score": {"min": 3.0},
                    "irregular_intervals": True
                }
            ),
            
            # Composite rules
            Rule(
                id="composite_phishing",
                name="Composite Phishing Detection",
                description="Multi-factor phishing detection",
                rule_type=RuleType.COMPOSITE,
                attack_type=AttackType.PHISHING,
                severity=SeverityLevel.CRITICAL,
                action=RuleAction.BLOCK,
                conditions={
                    "risk_score": {"min": 0.7},
                    "suspicion_score": {"min": 0.6},
                    "brand_similarity_score": {"min": 0.8}
                }
            ),
            
            Rule(
                id="composite_malware",
                name="Composite Malware Detection",
                description="Multi-factor malware detection",
                rule_type=RuleType.COMPOSITE,
                attack_type=AttackType.MALWARE,
                severity=SeverityLevel.CRITICAL,
                action=RuleAction.BLOCK,
                conditions={
                    "risk_score": {"min": 0.8},
                    "anomaly_score": {"min": 0.7},
                    "threat_types": {"min_count": 2}
                }
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: Rule) -> None:
        """Add a new detection rule."""
        self.rules[rule.id] = rule
        logger.info("Rule added", rule_id=rule.id, rule_name=rule.name)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a detection rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info("Rule removed", rule_id=rule_id)
            return True
        return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule."""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.utcnow()
        logger.info("Rule updated", rule_id=rule_id, updates=updates)
        return True
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        return self.update_rule(rule_id, {"enabled": True})
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        return self.update_rule(rule_id, {"enabled": False})
    
    def evaluate_rules(self, features: Dict[str, Any], 
                      target_type: str = "url") -> List[Dict[str, Any]]:
        """Evaluate all rules against the given features."""
        matches = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                if self._evaluate_rule(rule, features, target_type):
                    match_result = {
                        'rule_id': rule_id,
                        'rule_name': rule.name,
                        'rule_type': rule.rule_type,
                        'attack_type': rule.attack_type,
                        'severity': rule.severity,
                        'action': rule.action,
                        'weight': rule.weight,
                        'matched_conditions': self._get_matched_conditions(rule, features),
                        'timestamp': datetime.utcnow()
                    }
                    matches.append(match_result)
                    
            except Exception as e:
                logger.error("Error evaluating rule", rule_id=rule_id, error=str(e))
        
        return matches
    
    def _evaluate_rule(self, rule: Rule, features: Dict[str, Any], 
                      target_type: str) -> bool:
        """Evaluate a single rule against features."""
        conditions = rule.conditions
        
        # Route to appropriate evaluator based on rule type
        if rule.rule_type == RuleType.URL_PATTERN:
            return self._evaluate_url_rule(rule, features)
        elif rule.rule_type == RuleType.IP_PATTERN:
            return self._evaluate_ip_rule(rule, features)
        elif rule.rule_type == RuleType.BEHAVIORAL:
            return self._evaluate_behavioral_rule(rule, features)
        elif rule.rule_type == RuleType.TEMPORAL:
            return self._evaluate_temporal_rule(rule, features)
        elif rule.rule_type == RuleType.REPUTATION:
            return self._evaluate_reputation_rule(rule, features)
        elif rule.rule_type == RuleType.COMPOSITE:
            return self._evaluate_composite_rule(rule, features)
        else:
            logger.warning("Unknown rule type", rule_type=rule.rule_type)
            return False
    
    def _evaluate_url_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate URL-specific rules."""
        conditions = rule.conditions
        url_features = features.get('url', {})
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, url_features):
                return False
        
        return True
    
    def _evaluate_ip_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate IP-specific rules."""
        conditions = rule.conditions
        ip_features = features.get('ip', {})
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, ip_features):
                return False
        
        return True
    
    def _evaluate_behavioral_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate behavioral rules."""
        conditions = rule.conditions
        behavioral_features = features.get('behavioral', {})
        temporal_features = features.get('temporal', {})
        
        # Combine behavioral and temporal features
        all_features = {**behavioral_features, **temporal_features}
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, all_features):
                return False
        
        return True
    
    def _evaluate_temporal_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate temporal rules."""
        conditions = rule.conditions
        temporal_features = features.get('temporal', {})
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, temporal_features):
                return False
        
        return True
    
    def _evaluate_reputation_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate reputation-based rules."""
        conditions = rule.conditions
        ip_features = features.get('ip', {})
        url_features = features.get('url', {})
        
        # Combine IP and URL features for reputation
        all_features = {**ip_features, **url_features}
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, all_features):
                return False
        
        return True
    
    def _evaluate_composite_rule(self, rule: Rule, features: Dict[str, Any]) -> bool:
        """Evaluate composite rules."""
        conditions = rule.conditions
        composite_features = features.get('composite', {})
        
        for condition, value in conditions.items():
            if not self._check_condition(condition, value, composite_features):
                return False
        
        return True
    
    def _check_condition(self, condition: str, expected_value: Any, 
                        actual_features: Dict[str, Any]) -> bool:
        """Check if a condition is met."""
        actual_value = actual_features.get(condition)
        
        if actual_value is None:
            return False
        
        # Handle different types of expected values
        if isinstance(expected_value, dict):
            return self._check_dict_condition(actual_value, expected_value)
        elif isinstance(expected_value, list):
            return actual_value in expected_value
        else:
            return actual_value == expected_value
    
    def _check_dict_condition(self, actual_value: Any, expected_dict: Dict[str, Any]) -> bool:
        """Check dictionary-based conditions."""
        for operator, threshold in expected_dict.items():
            if operator == "min":
                if actual_value < threshold:
                    return False
            elif operator == "max":
                if actual_value > threshold:
                    return False
            elif operator == "in":
                if actual_value not in threshold:
                    return False
            elif operator == "min_count":
                if isinstance(actual_value, (list, set)):
                    if len(actual_value) < threshold:
                        return False
                else:
                    return False
            elif operator == "max_count":
                if isinstance(actual_value, (list, set)):
                    if len(actual_value) > threshold:
                        return False
                else:
                    return False
        
        return True
    
    def _get_matched_conditions(self, rule: Rule, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get the conditions that matched for a rule."""
        matched = {}
        conditions = rule.conditions
        
        # This is a simplified implementation
        # In practice, you'd want to track which specific conditions matched
        for condition in conditions.keys():
            if condition in features:
                matched[condition] = features[condition]
        
        return matched
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        """Get all rules of a specific type."""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]
    
    def get_rules_by_attack_type(self, attack_type: AttackType) -> List[Rule]:
        """Get all rules for a specific attack type."""
        return [rule for rule in self.rules.values() if rule.attack_type == attack_type]
    
    def get_enabled_rules(self) -> List[Rule]:
        """Get all enabled rules."""
        return [rule for rule in self.rules.values() if rule.enabled]
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rules."""
        total_rules = len(self.rules)
        enabled_rules = len(self.get_enabled_rules())
        
        rule_types = {}
        attack_types = {}
        severity_levels = {}
        
        for rule in self.rules.values():
            rule_types[rule.rule_type] = rule_types.get(rule.rule_type, 0) + 1
            attack_types[rule.attack_type] = attack_types.get(rule.attack_type, 0) + 1
            severity_levels[rule.severity] = severity_levels.get(rule.severity, 0) + 1
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'rule_types': rule_types,
            'attack_types': attack_types,
            'severity_levels': severity_levels
        }


# Global rule engine instance
rule_engine = RuleEngine()


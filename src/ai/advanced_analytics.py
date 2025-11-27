"""
Advanced analytics and insights for cyberattack detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, t-SNE
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database

logger = structlog.get_logger(__name__)


class AnalyticsType(str, Enum):
    """Types of analytics."""
    THREAT_LANDSCAPE = "threat_landscape"
    ATTACK_CORRELATION = "attack_correlation"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    GEOGRAPHIC_ANALYSIS = "geographic_analysis"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"


@dataclass
class ThreatLandscape:
    """Threat landscape analysis result."""
    total_threats: int
    threat_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    temporal_trends: Dict[str, List[float]]
    geographic_distribution: Dict[str, int]
    emerging_threats: List[Dict[str, Any]]
    threat_evolution: Dict[str, List[Dict[str, Any]]]


@dataclass
class AttackCorrelation:
    """Attack correlation analysis result."""
    correlation_matrix: np.ndarray
    strong_correlations: List[Dict[str, Any]]
    attack_sequences: List[List[str]]
    co_occurrence_patterns: Dict[str, List[str]]
    causal_relationships: Dict[str, List[str]]


@dataclass
class PredictiveInsight:
    """Predictive insight result."""
    prediction_type: str
    predicted_value: Any
    confidence: float
    time_horizon: str
    factors: List[str]
    risk_assessment: Dict[str, float]
    recommendations: List[str]


class ThreatLandscapeAnalyzer:
    """Analyzes the overall threat landscape."""
    
    def __init__(self):
        self.threat_history = []
        self.geographic_data = {}
        self.temporal_patterns = {}
    
    async def analyze_threat_landscape(self, time_window: int = 30) -> ThreatLandscape:
        """Analyze the current threat landscape."""
        try:
            # Get threat data from database
            threat_data = await self._get_threat_data(time_window)
            
            # Analyze threat distribution
            threat_distribution = self._analyze_threat_distribution(threat_data)
            
            # Analyze severity distribution
            severity_distribution = self._analyze_severity_distribution(threat_data)
            
            # Analyze temporal trends
            temporal_trends = self._analyze_temporal_trends(threat_data)
            
            # Analyze geographic distribution
            geographic_distribution = self._analyze_geographic_distribution(threat_data)
            
            # Identify emerging threats
            emerging_threats = self._identify_emerging_threats(threat_data)
            
            # Analyze threat evolution
            threat_evolution = self._analyze_threat_evolution(threat_data)
            
            return ThreatLandscape(
                total_threats=len(threat_data),
                threat_distribution=threat_distribution,
                severity_distribution=severity_distribution,
                temporal_trends=temporal_trends,
                geographic_distribution=geographic_distribution,
                emerging_threats=emerging_threats,
                threat_evolution=threat_evolution
            )
            
        except Exception as e:
            logger.error("Error analyzing threat landscape", error=str(e))
            raise
    
    async def _get_threat_data(self, time_window: int) -> List[Dict[str, Any]]:
        """Get threat data from database."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            
            # Get recent threats
            cutoff_time = datetime.utcnow() - timedelta(days=time_window)
            
            cursor = collection.find({
                "status": "completed",
                "final_attack_type": {"$ne": "unknown"},
                "created_at": {"$gte": cutoff_time}
            })
            
            threat_data = []
            async for doc in cursor:
                threat_data.append(doc)
            
            return threat_data
            
        except Exception as e:
            logger.error("Error getting threat data", error=str(e))
            return []
    
    def _analyze_threat_distribution(self, threat_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of threat types."""
        try:
            distribution = Counter()
            
            for threat in threat_data:
                attack_type = threat.get("final_attack_type", "unknown")
                distribution[attack_type] += 1
            
            return dict(distribution)
            
        except Exception as e:
            logger.error("Error analyzing threat distribution", error=str(e))
            return {}
    
    def _analyze_severity_distribution(self, threat_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of severity levels."""
        try:
            distribution = Counter()
            
            for threat in threat_data:
                severity = threat.get("severity", "low")
                distribution[severity] += 1
            
            return dict(distribution)
            
        except Exception as e:
            logger.error("Error analyzing severity distribution", error=str(e))
            return {}
    
    def _analyze_temporal_trends(self, threat_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Analyze temporal trends in threats."""
        try:
            trends = defaultdict(list)
            
            # Group by day
            daily_threats = defaultdict(int)
            for threat in threat_data:
                date = threat.get("created_at", datetime.utcnow()).date()
                daily_threats[date] += 1
            
            # Convert to time series
            dates = sorted(daily_threats.keys())
            threat_counts = [daily_threats[date] for date in dates]
            
            trends["daily_threats"] = threat_counts
            trends["dates"] = [date.isoformat() for date in dates]
            
            # Calculate moving average
            if len(threat_counts) >= 7:
                moving_avg = []
                for i in range(6, len(threat_counts)):
                    avg = np.mean(threat_counts[i-6:i+1])
                    moving_avg.append(avg)
                trends["moving_average"] = moving_avg
            
            return dict(trends)
            
        except Exception as e:
            logger.error("Error analyzing temporal trends", error=str(e))
            return {}
    
    def _analyze_geographic_distribution(self, threat_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze geographic distribution of threats."""
        try:
            distribution = Counter()
            
            for threat in threat_data:
                # Extract geographic information from IP features
                ip_features = threat.get("ip_features", {})
                country = ip_features.get("country", "unknown")
                distribution[country] += 1
            
            return dict(distribution)
            
        except Exception as e:
            logger.error("Error analyzing geographic distribution", error=str(e))
            return {}
    
    def _identify_emerging_threats(self, threat_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emerging threat patterns."""
        try:
            emerging_threats = []
            
            # Group threats by attack type and time
            threat_groups = defaultdict(list)
            for threat in threat_data:
                attack_type = threat.get("final_attack_type", "unknown")
                threat_groups[attack_type].append(threat)
            
            # Identify emerging patterns
            for attack_type, threats in threat_groups.items():
                if len(threats) >= 5:  # Minimum threshold
                    # Calculate growth rate
                    recent_threats = [t for t in threats if 
                                    (datetime.utcnow() - t.get("created_at", datetime.utcnow())).days <= 7]
                    
                    if len(recent_threats) > len(threats) * 0.3:  # 30% in last week
                        emerging_threats.append({
                            "attack_type": attack_type,
                            "total_count": len(threats),
                            "recent_count": len(recent_threats),
                            "growth_rate": len(recent_threats) / len(threats),
                            "first_seen": min(t.get("created_at", datetime.utcnow()) for t in threats),
                            "last_seen": max(t.get("created_at", datetime.utcnow()) for t in threats)
                        })
            
            return emerging_threats
            
        except Exception as e:
            logger.error("Error identifying emerging threats", error=str(e))
            return []
    
    def _analyze_threat_evolution(self, threat_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze evolution of threat patterns."""
        try:
            evolution = defaultdict(list)
            
            # Group by attack type
            threat_groups = defaultdict(list)
            for threat in threat_data:
                attack_type = threat.get("final_attack_type", "unknown")
                threat_groups[attack_type].append(threat)
            
            # Analyze evolution for each attack type
            for attack_type, threats in threat_groups.items():
                if len(threats) >= 10:  # Minimum for evolution analysis
                    # Sort by timestamp
                    threats.sort(key=lambda x: x.get("created_at", datetime.utcnow()))
                    
                    # Analyze evolution over time
                    evolution_data = []
                    for i, threat in enumerate(threats):
                        evolution_data.append({
                            "timestamp": threat.get("created_at", datetime.utcnow()).isoformat(),
                            "confidence": threat.get("final_confidence", 0.0),
                            "severity": threat.get("severity", "low"),
                            "sequence_number": i
                        })
                    
                    evolution[attack_type] = evolution_data
            
            return dict(evolution)
            
        except Exception as e:
            logger.error("Error analyzing threat evolution", error=str(e))
            return {}


class AttackCorrelationAnalyzer:
    """Analyzes correlations between different attack types."""
    
    def __init__(self):
        self.attack_sequences = []
        self.correlation_matrix = None
        self.network_graph = nx.Graph()
    
    async def analyze_attack_correlations(self, time_window: int = 30) -> AttackCorrelation:
        """Analyze correlations between attack types."""
        try:
            # Get attack data
            attack_data = await self._get_attack_data(time_window)
            
            # Build correlation matrix
            correlation_matrix = self._build_correlation_matrix(attack_data)
            
            # Find strong correlations
            strong_correlations = self._find_strong_correlations(correlation_matrix)
            
            # Analyze attack sequences
            attack_sequences = self._analyze_attack_sequences(attack_data)
            
            # Find co-occurrence patterns
            co_occurrence_patterns = self._find_co_occurrence_patterns(attack_data)
            
            # Identify causal relationships
            causal_relationships = self._identify_causal_relationships(attack_data)
            
            return AttackCorrelation(
                correlation_matrix=correlation_matrix,
                strong_correlations=strong_correlations,
                attack_sequences=attack_sequences,
                co_occurrence_patterns=co_occurrence_patterns,
                causal_relationships=causal_relationships
            )
            
        except Exception as e:
            logger.error("Error analyzing attack correlations", error=str(e))
            raise
    
    async def _get_attack_data(self, time_window: int) -> List[Dict[str, Any]]:
        """Get attack data from database."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            
            # Get recent attacks
            cutoff_time = datetime.utcnow() - timedelta(days=time_window)
            
            cursor = collection.find({
                "status": "completed",
                "final_attack_type": {"$ne": "unknown"},
                "created_at": {"$gte": cutoff_time}
            }).sort("created_at", 1)
            
            attack_data = []
            async for doc in cursor:
                attack_data.append(doc)
            
            return attack_data
            
        except Exception as e:
            logger.error("Error getting attack data", error=str(e))
            return []
    
    def _build_correlation_matrix(self, attack_data: List[Dict[str, Any]]) -> np.ndarray:
        """Build correlation matrix between attack types."""
        try:
            # Get unique attack types
            attack_types = list(set(attack.get("final_attack_type", "unknown") for attack in attack_data))
            attack_types.sort()
            
            # Create correlation matrix
            n_types = len(attack_types)
            correlation_matrix = np.zeros((n_types, n_types))
            
            # Calculate correlations
            for i, attack_type1 in enumerate(attack_types):
                for j, attack_type2 in enumerate(attack_types):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        correlation = self._calculate_attack_correlation(
                            attack_data, attack_type1, attack_type2
                        )
                        correlation_matrix[i, j] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            logger.error("Error building correlation matrix", error=str(e))
            return np.array([])
    
    def _calculate_attack_correlation(self, attack_data: List[Dict[str, Any]], 
                                    attack_type1: str, attack_type2: str) -> float:
        """Calculate correlation between two attack types."""
        try:
            # Group attacks by time windows
            time_windows = defaultdict(list)
            
            for attack in attack_data:
                attack_type = attack.get("final_attack_type", "unknown")
                timestamp = attack.get("created_at", datetime.utcnow())
                
                # Group by hour
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                time_windows[hour_key].append(attack_type)
            
            # Calculate correlation
            type1_counts = []
            type2_counts = []
            
            for hour, attacks in time_windows.items():
                type1_count = attacks.count(attack_type1)
                type2_count = attacks.count(attack_type2)
                
                type1_counts.append(type1_count)
                type2_counts.append(type2_count)
            
            # Calculate Pearson correlation
            if len(type1_counts) > 1 and len(type2_counts) > 1:
                correlation = np.corrcoef(type1_counts, type2_counts)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0
            
        except Exception as e:
            logger.error("Error calculating attack correlation", error=str(e))
            return 0.0
    
    def _find_strong_correlations(self, correlation_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Find strong correlations in the matrix."""
        try:
            strong_correlations = []
            threshold = 0.5
            
            for i in range(correlation_matrix.shape[0]):
                for j in range(i+1, correlation_matrix.shape[1]):
                    correlation = correlation_matrix[i, j]
                    
                    if abs(correlation) > threshold:
                        strong_correlations.append({
                            "attack_type1": f"attack_{i}",
                            "attack_type2": f"attack_{j}",
                            "correlation": correlation,
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate"
                        })
            
            return strong_correlations
            
        except Exception as e:
            logger.error("Error finding strong correlations", error=str(e))
            return []
    
    def _analyze_attack_sequences(self, attack_data: List[Dict[str, Any]]) -> List[List[str]]:
        """Analyze attack sequences."""
        try:
            sequences = []
            
            # Group attacks by source (IP or session)
            attack_groups = defaultdict(list)
            
            for attack in attack_data:
                # Use IP as grouping key
                ip_features = attack.get("ip_features", {})
                source_ip = ip_features.get("ip_address", "unknown")
                
                attack_groups[source_ip].append(attack)
            
            # Extract sequences
            for source_ip, attacks in attack_groups.items():
                if len(attacks) >= 2:  # Minimum sequence length
                    # Sort by timestamp
                    attacks.sort(key=lambda x: x.get("created_at", datetime.utcnow()))
                    
                    # Extract attack sequence
                    sequence = [attack.get("final_attack_type", "unknown") for attack in attacks]
                    sequences.append(sequence)
            
            return sequences
            
        except Exception as e:
            logger.error("Error analyzing attack sequences", error=str(e))
            return []
    
    def _find_co_occurrence_patterns(self, attack_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find co-occurrence patterns."""
        try:
            co_occurrence = defaultdict(list)
            
            # Group attacks by time windows
            time_windows = defaultdict(list)
            
            for attack in attack_data:
                timestamp = attack.get("created_at", datetime.utcnow())
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                time_windows[hour_key].append(attack.get("final_attack_type", "unknown"))
            
            # Find co-occurrences
            for hour, attacks in time_windows.items():
                if len(attacks) >= 2:
                    unique_attacks = list(set(attacks))
                    
                    for i, attack1 in enumerate(unique_attacks):
                        for attack2 in unique_attacks[i+1:]:
                            co_occurrence[attack1].append(attack2)
                            co_occurrence[attack2].append(attack1)
            
            # Remove duplicates and count
            for attack_type in co_occurrence:
                co_occurrence[attack_type] = list(set(co_occurrence[attack_type]))
            
            return dict(co_occurrence)
            
        except Exception as e:
            logger.error("Error finding co-occurrence patterns", error=str(e))
            return {}
    
    def _identify_causal_relationships(self, attack_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify causal relationships between attacks."""
        try:
            causal_relationships = defaultdict(list)
            
            # Analyze temporal relationships
            attack_sequences = self._analyze_attack_sequences(attack_data)
            
            for sequence in attack_sequences:
                if len(sequence) >= 2:
                    # Look for causal patterns
                    for i in range(len(sequence) - 1):
                        current_attack = sequence[i]
                        next_attack = sequence[i + 1]
                        
                        # Check if there's a causal relationship
                        if self._is_causal_relationship(current_attack, next_attack):
                            causal_relationships[current_attack].append(next_attack)
            
            # Remove duplicates
            for attack_type in causal_relationships:
                causal_relationships[attack_type] = list(set(causal_relationships[attack_type]))
            
            return dict(causal_relationships)
            
        except Exception as e:
            logger.error("Error identifying causal relationships", error=str(e))
            return {}
    
    def _is_causal_relationship(self, attack1: str, attack2: str) -> bool:
        """Check if there's a causal relationship between two attacks."""
        try:
            # Define known causal relationships
            causal_patterns = {
                "probe": ["dos", "ddos", "r2l"],
                "r2l": ["u2r"],
                "phishing": ["malware", "ransomware"],
                "malware": ["ransomware", "botnet"]
            }
            
            return attack2 in causal_patterns.get(attack1, [])
            
        except Exception as e:
            logger.error("Error checking causal relationship", error=str(e))
            return False


class PredictiveAnalyzer:
    """Predictive analytics for threat forecasting."""
    
    def __init__(self):
        self.prediction_models = {}
        self.forecast_history = []
    
    async def predict_threat_trends(self, time_horizon: str = "7d") -> List[PredictiveInsight]:
        """Predict future threat trends."""
        try:
            insights = []
            
            # Get historical data
            historical_data = await self._get_historical_data()
            
            # Predict attack volume
            volume_prediction = await self._predict_attack_volume(historical_data, time_horizon)
            insights.append(volume_prediction)
            
            # Predict attack types
            type_prediction = await self._predict_attack_types(historical_data, time_horizon)
            insights.append(type_prediction)
            
            # Predict severity trends
            severity_prediction = await self._predict_severity_trends(historical_data, time_horizon)
            insights.append(severity_prediction)
            
            # Predict geographic trends
            geographic_prediction = await self._predict_geographic_trends(historical_data, time_horizon)
            insights.append(geographic_prediction)
            
            logger.info("Threat trend prediction completed", 
                       time_horizon=time_horizon,
                       insights_count=len(insights))
            
            return insights
            
        except Exception as e:
            logger.error("Error predicting threat trends", error=str(e))
            raise
    
    async def _get_historical_data(self, days: int = 90) -> List[Dict[str, Any]]:
        """Get historical threat data."""
        try:
            db = await get_database()
            collection = db.get_collection("analysis_results")
            
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            cursor = collection.find({
                "status": "completed",
                "created_at": {"$gte": cutoff_time}
            }).sort("created_at", 1)
            
            historical_data = []
            async for doc in cursor:
                historical_data.append(doc)
            
            return historical_data
            
        except Exception as e:
            logger.error("Error getting historical data", error=str(e))
            return []
    
    async def _predict_attack_volume(self, historical_data: List[Dict[str, Any]], 
                                   time_horizon: str) -> PredictiveInsight:
        """Predict future attack volume."""
        try:
            # Analyze historical volume
            daily_volumes = defaultdict(int)
            
            for data in historical_data:
                date = data.get("created_at", datetime.utcnow()).date()
                daily_volumes[date] += 1
            
            # Calculate trend
            dates = sorted(daily_volumes.keys())
            volumes = [daily_volumes[date] for date in dates]
            
            if len(volumes) >= 7:
                # Simple linear trend
                x = np.arange(len(volumes))
                trend = np.polyfit(x, volumes, 1)
                
                # Predict future volume
                future_days = 7 if time_horizon == "7d" else 30
                predicted_volume = trend[0] * (len(volumes) + future_days) + trend[1]
                
                # Calculate confidence
                confidence = min(0.9, 0.5 + len(volumes) / 100)
                
                return PredictiveInsight(
                    prediction_type="attack_volume",
                    predicted_value=int(predicted_volume),
                    confidence=confidence,
                    time_horizon=time_horizon,
                    factors=["historical_trend", "seasonal_patterns"],
                    risk_assessment={"volume_risk": min(predicted_volume / 100, 1.0)},
                    recommendations=["Increase monitoring capacity", "Prepare for higher attack volume"]
                )
            else:
                return PredictiveInsight(
                    prediction_type="attack_volume",
                    predicted_value=0,
                    confidence=0.0,
                    time_horizon=time_horizon,
                    factors=["insufficient_data"],
                    risk_assessment={"volume_risk": 0.0},
                    recommendations=["Collect more historical data"]
                )
            
        except Exception as e:
            logger.error("Error predicting attack volume", error=str(e))
            raise
    
    async def _predict_attack_types(self, historical_data: List[Dict[str, Any]], 
                                  time_horizon: str) -> PredictiveInsight:
        """Predict future attack types."""
        try:
            # Analyze attack type trends
            attack_type_counts = Counter()
            
            for data in historical_data:
                attack_type = data.get("final_attack_type", "unknown")
                attack_type_counts[attack_type] += 1
            
            # Predict most likely attack type
            if attack_type_counts:
                most_common = attack_type_counts.most_common(1)[0]
                predicted_type = most_common[0]
                confidence = most_common[1] / len(historical_data)
                
                return PredictiveInsight(
                    prediction_type="attack_type",
                    predicted_value=predicted_type,
                    confidence=confidence,
                    time_horizon=time_horizon,
                    factors=["historical_frequency", "current_trends"],
                    risk_assessment={"type_risk": confidence},
                    recommendations=[f"Focus defenses on {predicted_type} attacks", "Update detection rules"]
                )
            else:
                return PredictiveInsight(
                    prediction_type="attack_type",
                    predicted_value="unknown",
                    confidence=0.0,
                    time_horizon=time_horizon,
                    factors=["no_data"],
                    risk_assessment={"type_risk": 0.0},
                    recommendations=["Collect more data"]
                )
            
        except Exception as e:
            logger.error("Error predicting attack types", error=str(e))
            raise
    
    async def _predict_severity_trends(self, historical_data: List[Dict[str, Any]], 
                                     time_horizon: str) -> PredictiveInsight:
        """Predict severity trends."""
        try:
            # Analyze severity trends
            severity_counts = Counter()
            
            for data in historical_data:
                severity = data.get("severity", "low")
                severity_counts[severity] += 1
            
            # Calculate severity risk
            high_severity_count = severity_counts.get("high", 0) + severity_counts.get("critical", 0)
            severity_risk = high_severity_count / len(historical_data) if historical_data else 0
            
            # Predict future severity
            predicted_severity = "high" if severity_risk > 0.3 else "medium" if severity_risk > 0.1 else "low"
            
            return PredictiveInsight(
                prediction_type="severity_trend",
                predicted_value=predicted_severity,
                confidence=severity_risk,
                time_horizon=time_horizon,
                factors=["historical_severity", "threat_evolution"],
                risk_assessment={"severity_risk": severity_risk},
                recommendations=["Enhance high-severity threat detection", "Prepare incident response"]
            )
            
        except Exception as e:
            logger.error("Error predicting severity trends", error=str(e))
            raise
    
    async def _predict_geographic_trends(self, historical_data: List[Dict[str, Any]], 
                                       time_horizon: str) -> PredictiveInsight:
        """Predict geographic trends."""
        try:
            # Analyze geographic distribution
            country_counts = Counter()
            
            for data in historical_data:
                ip_features = data.get("ip_features", {})
                country = ip_features.get("country", "unknown")
                country_counts[country] += 1
            
            # Predict most likely source country
            if country_counts:
                most_common = country_counts.most_common(1)[0]
                predicted_country = most_common[0]
                confidence = most_common[1] / len(historical_data)
                
                return PredictiveInsight(
                    prediction_type="geographic_trend",
                    predicted_value=predicted_country,
                    confidence=confidence,
                    time_horizon=time_horizon,
                    factors=["historical_geographic_distribution", "current_threats"],
                    risk_assessment={"geographic_risk": confidence},
                    recommendations=[f"Monitor threats from {predicted_country}", "Update geo-blocking rules"]
                )
            else:
                return PredictiveInsight(
                    prediction_type="geographic_trend",
                    predicted_value="unknown",
                    confidence=0.0,
                    time_horizon=time_horizon,
                    factors=["no_geographic_data"],
                    risk_assessment={"geographic_risk": 0.0},
                    recommendations=["Collect geographic data"]
                )
            
        except Exception as e:
            logger.error("Error predicting geographic trends", error=str(e))
            raise


class AdvancedAnalyticsEngine:
    """Main engine for advanced analytics."""
    
    def __init__(self):
        self.threat_landscape_analyzer = ThreatLandscapeAnalyzer()
        self.attack_correlation_analyzer = AttackCorrelationAnalyzer()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.analytics_cache = {}
    
    async def run_comprehensive_analysis(self, time_window: int = 30) -> Dict[str, Any]:
        """Run comprehensive analytics analysis."""
        try:
            # Analyze threat landscape
            threat_landscape = await self.threat_landscape_analyzer.analyze_threat_landscape(time_window)
            
            # Analyze attack correlations
            attack_correlations = await self.attack_correlation_analyzer.analyze_attack_correlations(time_window)
            
            # Generate predictive insights
            predictive_insights = await self.predictive_analyzer.predict_threat_trends("7d")
            
            # Combine results
            comprehensive_analysis = {
                "threat_landscape": threat_landscape,
                "attack_correlations": attack_correlations,
                "predictive_insights": predictive_insights,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "time_window_days": time_window
            }
            
            logger.info("Comprehensive analysis completed", 
                       time_window=time_window,
                       total_threats=threat_landscape.total_threats)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error("Error running comprehensive analysis", error=str(e))
            raise
    
    async def generate_analytics_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        try:
            report = {
                "executive_summary": self._generate_executive_summary(analysis_results),
                "threat_landscape": self._format_threat_landscape(analysis_results["threat_landscape"]),
                "attack_correlations": self._format_attack_correlations(analysis_results["attack_correlations"]),
                "predictive_insights": self._format_predictive_insights(analysis_results["predictive_insights"]),
                "recommendations": self._generate_recommendations(analysis_results),
                "visualizations": await self._generate_visualizations(analysis_results)
            }
            
            return report
            
        except Exception as e:
            logger.error("Error generating analytics report", error=str(e))
            return {}
    
    def _generate_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        try:
            threat_landscape = analysis_results["threat_landscape"]
            predictive_insights = analysis_results["predictive_insights"]
            
            # Calculate key metrics
            total_threats = threat_landscape.total_threats
            emerging_threats_count = len(threat_landscape.emerging_threats)
            
            # Get top threat type
            top_threat = max(threat_landscape.threat_distribution.items(), key=lambda x: x[1]) if threat_landscape.threat_distribution else ("unknown", 0)
            
            # Get predicted volume
            volume_prediction = next((insight for insight in predictive_insights if insight.prediction_type == "attack_volume"), None)
            predicted_volume = volume_prediction.predicted_value if volume_prediction else 0
            
            return {
                "total_threats_analyzed": total_threats,
                "emerging_threats": emerging_threats_count,
                "top_threat_type": top_threat[0],
                "top_threat_count": top_threat[1],
                "predicted_future_volume": predicted_volume,
                "key_findings": [
                    f"Analyzed {total_threats} threats in the last 30 days",
                    f"Identified {emerging_threats_count} emerging threat patterns",
                    f"Most common threat type: {top_threat[0]} ({top_threat[1]} occurrences)",
                    f"Predicted future attack volume: {predicted_volume} attacks"
                ]
            }
            
        except Exception as e:
            logger.error("Error generating executive summary", error=str(e))
            return {}
    
    def _format_threat_landscape(self, threat_landscape: ThreatLandscape) -> Dict[str, Any]:
        """Format threat landscape for report."""
        try:
            return {
                "total_threats": threat_landscape.total_threats,
                "threat_distribution": threat_landscape.threat_distribution,
                "severity_distribution": threat_landscape.severity_distribution,
                "emerging_threats": threat_landscape.emerging_threats,
                "geographic_distribution": threat_landscape.geographic_distribution
            }
            
        except Exception as e:
            logger.error("Error formatting threat landscape", error=str(e))
            return {}
    
    def _format_attack_correlations(self, attack_correlations: AttackCorrelation) -> Dict[str, Any]:
        """Format attack correlations for report."""
        try:
            return {
                "strong_correlations": attack_correlations.strong_correlations,
                "attack_sequences": attack_correlations.attack_sequences,
                "co_occurrence_patterns": attack_correlations.co_occurrence_patterns,
                "causal_relationships": attack_correlations.causal_relationships
            }
            
        except Exception as e:
            logger.error("Error formatting attack correlations", error=str(e))
            return {}
    
    def _format_predictive_insights(self, predictive_insights: List[PredictiveInsight]) -> Dict[str, Any]:
        """Format predictive insights for report."""
        try:
            formatted_insights = {}
            
            for insight in predictive_insights:
                formatted_insights[insight.prediction_type] = {
                    "predicted_value": insight.predicted_value,
                    "confidence": insight.confidence,
                    "time_horizon": insight.time_horizon,
                    "factors": insight.factors,
                    "risk_assessment": insight.risk_assessment,
                    "recommendations": insight.recommendations
                }
            
            return formatted_insights
            
        except Exception as e:
            logger.error("Error formatting predictive insights", error=str(e))
            return {}
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        try:
            recommendations = []
            
            # Threat landscape recommendations
            threat_landscape = analysis_results["threat_landscape"]
            if threat_landscape.emerging_threats:
                recommendations.append("Focus on emerging threat patterns and update detection rules")
            
            # Attack correlation recommendations
            attack_correlations = analysis_results["attack_correlations"]
            if attack_correlations.strong_correlations:
                recommendations.append("Implement correlation-based detection to catch related attacks")
            
            # Predictive insights recommendations
            predictive_insights = analysis_results["predictive_insights"]
            for insight in predictive_insights:
                recommendations.extend(insight.recommendations)
            
            # General recommendations
            recommendations.extend([
                "Continuously monitor threat landscape changes",
                "Update ML models with new threat patterns",
                "Enhance threat intelligence integration",
                "Implement proactive defense measures"
            ])
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            return []
    
    async def _generate_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization data."""
        try:
            visualizations = {}
            
            # Threat distribution pie chart
            threat_landscape = analysis_results["threat_landscape"]
            if threat_landscape.threat_distribution:
                pie_chart = self._create_pie_chart(
                    threat_landscape.threat_distribution,
                    "Threat Type Distribution"
                )
                visualizations["threat_distribution"] = pie_chart
            
            # Temporal trends line chart
            if threat_landscape.temporal_trends:
                line_chart = self._create_line_chart(
                    threat_landscape.temporal_trends,
                    "Threat Trends Over Time"
                )
                visualizations["temporal_trends"] = line_chart
            
            # Geographic distribution map
            if threat_landscape.geographic_distribution:
                map_chart = self._create_geographic_map(
                    threat_landscape.geographic_distribution,
                    "Geographic Threat Distribution"
                )
                visualizations["geographic_distribution"] = map_chart
            
            return visualizations
            
        except Exception as e:
            logger.error("Error generating visualizations", error=str(e))
            return {}
    
    def _create_pie_chart(self, data: Dict[str, int], title: str) -> str:
        """Create pie chart visualization."""
        try:
            plt.figure(figsize=(10, 6))
            plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
            plt.title(title)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            logger.error("Error creating pie chart", error=str(e))
            return ""
    
    def _create_line_chart(self, data: Dict[str, List[float]], title: str) -> str:
        """Create line chart visualization."""
        try:
            plt.figure(figsize=(12, 6))
            
            if "daily_threats" in data and "dates" in data:
                dates = data["dates"]
                threats = data["daily_threats"]
                
                plt.plot(dates, threats, marker='o')
                plt.title(title)
                plt.xlabel("Date")
                plt.ylabel("Number of Threats")
                plt.xticks(rotation=45)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            logger.error("Error creating line chart", error=str(e))
            return ""
    
    def _create_geographic_map(self, data: Dict[str, int], title: str) -> str:
        """Create geographic map visualization."""
        try:
            # This would create a world map with threat distribution
            # For now, create a bar chart
            plt.figure(figsize=(12, 6))
            
            countries = list(data.keys())[:10]  # Top 10 countries
            counts = [data[country] for country in countries]
            
            plt.bar(countries, counts)
            plt.title(title)
            plt.xlabel("Country")
            plt.ylabel("Number of Threats")
            plt.xticks(rotation=45)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            logger.error("Error creating geographic map", error=str(e))
            return ""
    
    async def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        try:
            return {
                "cached_analyses": len(self.analytics_cache),
                "available_analyzers": [
                    "threat_landscape_analyzer",
                    "attack_correlation_analyzer",
                    "predictive_analyzer"
                ],
                "supported_analytics_types": [atype.value for atype in AnalyticsType]
            }
            
        except Exception as e:
            logger.error("Error getting analytics statistics", error=str(e))
            return {}


# Global advanced analytics engine instance
advanced_analytics_engine = AdvancedAnalyticsEngine()

"""
Monitoring and metrics collection for the cyberattack detection system.
"""
import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis.asyncio as redis

from ..core.config import settings
from ..database.connection import get_database
from ..database.models import SystemMetrics

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """Metric data structure."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    metric_type: MetricType


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        self.redis_client = None
        self.metrics_cache = {}
        self._initialize_redis()
        self._initialize_prometheus_metrics()
    
    def _initialize_redis(self):
        """Initialize Redis client for metrics storage."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.warning("Could not initialize Redis for metrics", error=str(e))
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Request metrics
            self.request_count = Counter(
                'cyberattack_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status_code']
            )
            
            self.request_duration = Histogram(
                'cyberattack_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            )
            
            # Analysis metrics
            self.analysis_count = Counter(
                'cyberattack_analyses_total',
                'Total number of analyses',
                ['target_type', 'attack_type', 'status']
            )
            
            self.analysis_duration = Histogram(
                'cyberattack_analysis_duration_seconds',
                'Analysis duration in seconds',
                ['target_type']
            )
            
            # Model metrics
            self.model_predictions = Counter(
                'cyberattack_model_predictions_total',
                'Total number of model predictions',
                ['model_name', 'attack_type']
            )
            
            self.model_accuracy = Gauge(
                'cyberattack_model_accuracy',
                'Model accuracy',
                ['model_name']
            )
            
            # System metrics
            self.system_cpu_usage = Gauge(
                'cyberattack_system_cpu_usage_percent',
                'System CPU usage percentage'
            )
            
            self.system_memory_usage = Gauge(
                'cyberattack_system_memory_usage_percent',
                'System memory usage percentage'
            )
            
            self.system_disk_usage = Gauge(
                'cyberattack_system_disk_usage_percent',
                'System disk usage percentage'
            )
            
            # Threat intelligence metrics
            self.threat_intel_queries = Counter(
                'cyberattack_threat_intel_queries_total',
                'Total number of threat intelligence queries',
                ['source', 'status']
            )
            
            self.threat_intel_cache_hits = Counter(
                'cyberattack_threat_intel_cache_hits_total',
                'Total number of threat intelligence cache hits',
                ['source']
            )
            
            # Alert metrics
            self.alerts_generated = Counter(
                'cyberattack_alerts_generated_total',
                'Total number of alerts generated',
                ['severity', 'attack_type']
            )
            
            self.alerts_acknowledged = Counter(
                'cyberattack_alerts_acknowledged_total',
                'Total number of alerts acknowledged',
                ['severity']
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error("Error initializing Prometheus metrics", error=str(e))
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            metrics = {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_memory_mb': process_memory,
                'process_cpu_percent': process_cpu,
                'timestamp': datetime.utcnow()
            }
            
            # Update Prometheus metrics
            self.system_cpu_usage.set(cpu_usage)
            self.system_memory_usage.set(memory_usage)
            self.system_disk_usage.set(disk_usage)
            
            return metrics
            
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
            return {}
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        try:
            db = await get_database()
            
            # Get analysis statistics
            analysis_collection = db.get_collection("analysis_results")
            total_analyses = await analysis_collection.count_documents({})
            
            # Get recent analyses (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_analyses = await analysis_collection.count_documents({
                "created_at": {"$gte": recent_cutoff}
            })
            
            # Get alert statistics
            alerts_collection = db.get_collection("alerts")
            total_alerts = await alerts_collection.count_documents({})
            unacknowledged_alerts = await alerts_collection.count_documents({
                "is_acknowledged": False
            })
            
            # Get model statistics
            models_collection = db.get_collection("model_metrics")
            total_models = await models_collection.count_documents({})
            
            metrics = {
                'total_analyses': total_analyses,
                'recent_analyses_24h': recent_analyses,
                'total_alerts': total_alerts,
                'unacknowledged_alerts': unacknowledged_alerts,
                'total_models': total_models,
                'timestamp': datetime.utcnow()
            }
            
            return metrics
            
        except Exception as e:
            logger.error("Error collecting application metrics", error=str(e))
            return {}
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            # Get metrics from Redis cache
            if self.redis_client:
                # Request metrics
                request_metrics = await self._get_redis_metrics('requests')
                
                # Analysis metrics
                analysis_metrics = await self._get_redis_metrics('analyses')
                
                # Model metrics
                model_metrics = await self._get_redis_metrics('models')
                
                metrics = {
                    'requests': request_metrics,
                    'analyses': analysis_metrics,
                    'models': model_metrics,
                    'timestamp': datetime.utcnow()
                }
                
                return metrics
            
            return {}
            
        except Exception as e:
            logger.error("Error collecting performance metrics", error=str(e))
            return {}
    
    async def _get_redis_metrics(self, metric_type: str) -> Dict[str, Any]:
        """Get metrics from Redis."""
        try:
            if not self.redis_client:
                return {}
            
            # Get current minute key
            current_minute = int(time.time() // 60)
            key = f"metrics:{metric_type}:{current_minute}"
            
            # Get metrics
            metrics = await self.redis_client.hgetall(key)
            
            # Convert string values to appropriate types
            converted_metrics = {}
            for k, v in metrics.items():
                try:
                    converted_metrics[k] = float(v)
                except ValueError:
                    converted_metrics[k] = v
            
            return converted_metrics
            
        except Exception as e:
            logger.error("Error getting Redis metrics", error=str(e))
            return {}
    
    async def store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store metrics in database."""
        try:
            db = await get_database()
            collection = db.get_collection("system_metrics")
            
            # Create system metrics document
            system_metrics = SystemMetrics(
                requests_per_second=metrics.get('requests_per_second', 0),
                average_response_time_ms=metrics.get('average_response_time_ms', 0),
                error_rate=metrics.get('error_rate', 0),
                cpu_usage_percent=metrics.get('cpu_usage_percent', 0),
                memory_usage_percent=metrics.get('memory_usage_percent', 0),
                disk_usage_percent=metrics.get('disk_usage_percent', 0),
                model_accuracy=metrics.get('model_accuracy', {}),
                prediction_latency_ms=metrics.get('prediction_latency_ms', {}),
                threats_detected=metrics.get('threats_detected', 0),
                false_positives=metrics.get('false_positives', 0),
                attack_type_distribution=metrics.get('attack_type_distribution', {})
            )
            
            await collection.insert_one(system_metrics.dict(by_alias=True))
            
        except Exception as e:
            logger.error("Error storing metrics", error=str(e))
    
    async def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history."""
        try:
            db = await get_database()
            collection = db.get_collection("system_metrics")
            
            # Get metrics from the last N hours
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            cursor = collection.find({
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", 1)
            
            metrics_history = []
            async for doc in cursor:
                metrics_history.append(doc)
            
            return metrics_history
            
        except Exception as e:
            logger.error("Error getting metrics history", error=str(e))
            return []
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        try:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error("Error recording request metrics", error=str(e))
    
    def record_analysis(self, target_type: str, attack_type: str, status: str, duration: float):
        """Record analysis metrics."""
        try:
            self.analysis_count.labels(
                target_type=target_type,
                attack_type=attack_type,
                status=status
            ).inc()
            
            self.analysis_duration.labels(
                target_type=target_type
            ).observe(duration)
            
        except Exception as e:
            logger.error("Error recording analysis metrics", error=str(e))
    
    def record_model_prediction(self, model_name: str, attack_type: str):
        """Record model prediction metrics."""
        try:
            self.model_predictions.labels(
                model_name=model_name,
                attack_type=attack_type
            ).inc()
            
        except Exception as e:
            logger.error("Error recording model prediction metrics", error=str(e))
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric."""
        try:
            self.model_accuracy.labels(
                model_name=model_name
            ).set(accuracy)
            
        except Exception as e:
            logger.error("Error updating model accuracy", error=str(e))
    
    def record_threat_intel_query(self, source: str, status: str):
        """Record threat intelligence query metrics."""
        try:
            self.threat_intel_queries.labels(
                source=source,
                status=status
            ).inc()
            
        except Exception as e:
            logger.error("Error recording threat intel query metrics", error=str(e))
    
    def record_threat_intel_cache_hit(self, source: str):
        """Record threat intelligence cache hit metrics."""
        try:
            self.threat_intel_cache_hits.labels(
                source=source
            ).inc()
            
        except Exception as e:
            logger.error("Error recording threat intel cache hit metrics", error=str(e))
    
    def record_alert(self, severity: str, attack_type: str):
        """Record alert metrics."""
        try:
            self.alerts_generated.labels(
                severity=severity,
                attack_type=attack_type
            ).inc()
            
        except Exception as e:
            logger.error("Error recording alert metrics", error=str(e))
    
    def record_alert_acknowledgment(self, severity: str):
        """Record alert acknowledgment metrics."""
        try:
            self.alerts_acknowledged.labels(
                severity=severity
            ).inc()
            
        except Exception as e:
            logger.error("Error recording alert acknowledgment metrics", error=str(e))
    
    async def start_prometheus_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(settings.prometheus_port)
            logger.info("Prometheus metrics server started", port=settings.prometheus_port)
            
        except Exception as e:
            logger.error("Error starting Prometheus server", error=str(e))
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        try:
            # Collect all metrics
            system_metrics = await self.collect_system_metrics()
            application_metrics = await self.collect_application_metrics()
            performance_metrics = await self.collect_performance_metrics()
            
            # Combine metrics
            summary = {
                'system': system_metrics,
                'application': application_metrics,
                'performance': performance_metrics,
                'timestamp': datetime.utcnow()
            }
            
            return summary
            
        except Exception as e:
            logger.error("Error getting metrics summary", error=str(e))
            return {}


class HealthChecker:
    """Checks system health and generates alerts."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_thresholds = {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 5.0,
            'response_time_ms': 1000.0
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            health_status = {
                'overall_status': 'healthy',
                'checks': {},
                'timestamp': datetime.utcnow()
            }
            
            # Get current metrics
            metrics = await self.metrics_collector.get_metrics_summary()
            
            # Check system metrics
            system_metrics = metrics.get('system', {})
            
            # CPU check
            cpu_usage = system_metrics.get('cpu_usage_percent', 0)
            cpu_status = 'healthy' if cpu_usage < self.health_thresholds['cpu_usage_percent'] else 'unhealthy'
            health_status['checks']['cpu'] = {
                'status': cpu_status,
                'value': cpu_usage,
                'threshold': self.health_thresholds['cpu_usage_percent']
            }
            
            # Memory check
            memory_usage = system_metrics.get('memory_usage_percent', 0)
            memory_status = 'healthy' if memory_usage < self.health_thresholds['memory_usage_percent'] else 'unhealthy'
            health_status['checks']['memory'] = {
                'status': memory_status,
                'value': memory_usage,
                'threshold': self.health_thresholds['memory_usage_percent']
            }
            
            # Disk check
            disk_usage = system_metrics.get('disk_usage_percent', 0)
            disk_status = 'healthy' if disk_usage < self.health_thresholds['disk_usage_percent'] else 'unhealthy'
            health_status['checks']['disk'] = {
                'status': disk_status,
                'value': disk_usage,
                'threshold': self.health_thresholds['disk_usage_percent']
            }
            
            # Application checks
            application_metrics = metrics.get('application', {})
            
            # Database check
            db_status = 'healthy' if application_metrics.get('total_analyses', 0) >= 0 else 'unhealthy'
            health_status['checks']['database'] = {
                'status': db_status,
                'value': application_metrics.get('total_analyses', 0)
            }
            
            # Determine overall status
            unhealthy_checks = [check for check in health_status['checks'].values() if check['status'] == 'unhealthy']
            
            if len(unhealthy_checks) > 0:
                health_status['overall_status'] = 'unhealthy'
            elif len(unhealthy_checks) > 2:
                health_status['overall_status'] = 'critical'
            
            return health_status
            
        except Exception as e:
            logger.error("Error checking health", error=str(e))
            return {
                'overall_status': 'unknown',
                'checks': {},
                'timestamp': datetime.utcnow(),
                'error': str(e)
            }


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker(metrics_collector)

#!/usr/bin/env python3
"""
Database initialization script for the cyberattack detection system.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.connection import db_manager
from database.models import AnalysisResult, Alert, ModelMetrics, SystemMetrics
from core.logger import logger


async def init_database():
    """Initialize the database with sample data."""
    try:
        # Connect to database
        await db_manager.connect()
        logger.info("Connected to database")
        
        # Create indexes
        await db_manager._create_indexes()
        logger.info("Created database indexes")
        
        # Insert sample data
        await insert_sample_data()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error("Error initializing database", error=str(e))
        raise
    finally:
        await db_manager.disconnect()


async def insert_sample_data():
    """Insert sample data into the database."""
    try:
        # Sample analysis results
        sample_analyses = [
            {
                "target_type": "url",
                "target_value": "https://example.com",
                "status": "completed",
                "final_attack_type": "unknown",
                "final_confidence": 0.1,
                "severity": "low",
                "risk_score": 0.2,
                "analysis_duration_ms": 150
            },
            {
                "target_type": "url",
                "target_value": "https://suspicious-site.com",
                "status": "completed",
                "final_attack_type": "phishing",
                "final_confidence": 0.85,
                "severity": "high",
                "risk_score": 0.9,
                "analysis_duration_ms": 200
            },
            {
                "target_type": "ip",
                "target_value": "192.168.1.100",
                "status": "completed",
                "final_attack_type": "probe",
                "final_confidence": 0.7,
                "severity": "medium",
                "risk_score": 0.6,
                "analysis_duration_ms": 120
            }
        ]
        
        # Insert sample analyses
        analysis_collection = db_manager.get_collection("analysis_results")
        for analysis_data in sample_analyses:
            analysis = AnalysisResult(**analysis_data)
            await analysis_collection.insert_one(analysis.dict(by_alias=True))
        
        logger.info(f"Inserted {len(sample_analyses)} sample analyses")
        
        # Sample alerts
        sample_alerts = [
            {
                "analysis_id": "sample_analysis_1",
                "alert_type": "threat_detected",
                "severity": "high",
                "title": "Phishing URL Detected",
                "description": "High-confidence phishing URL detected",
                "attack_type": "phishing",
                "is_acknowledged": False
            },
            {
                "analysis_id": "sample_analysis_2",
                "alert_type": "anomaly_detected",
                "severity": "medium",
                "title": "Suspicious IP Activity",
                "description": "Unusual activity pattern detected from IP",
                "attack_type": "probe",
                "is_acknowledged": True,
                "acknowledged_by": "admin",
                "acknowledged_at": "2024-01-01T12:00:00Z"
            }
        ]
        
        # Insert sample alerts
        alerts_collection = db_manager.get_collection("alerts")
        for alert_data in sample_alerts:
            alert = Alert(**alert_data)
            await alerts_collection.insert_one(alert.dict(by_alias=True))
        
        logger.info(f"Inserted {len(sample_alerts)} sample alerts")
        
        # Sample model metrics
        sample_metrics = [
            {
                "model_name": "random_forest",
                "model_version": "1.0.0",
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.97,
                "f1_score": 0.95,
                "auc_roc": 0.98,
                "training_samples": 10000,
                "validation_samples": 2000,
                "training_duration_seconds": 300,
                "feature_count": 50,
                "model_size_mb": 2.5,
                "training_start": "2024-01-01T10:00:00Z",
                "training_end": "2024-01-01T10:05:00Z"
            },
            {
                "model_name": "xgboost",
                "model_version": "1.0.0",
                "accuracy": 0.97,
                "precision": 0.95,
                "recall": 0.99,
                "f1_score": 0.97,
                "auc_roc": 0.99,
                "training_samples": 10000,
                "validation_samples": 2000,
                "training_duration_seconds": 180,
                "feature_count": 50,
                "model_size_mb": 1.8,
                "training_start": "2024-01-01T10:00:00Z",
                "training_end": "2024-01-01T10:03:00Z"
            }
        ]
        
        # Insert sample model metrics
        metrics_collection = db_manager.get_collection("model_metrics")
        for metrics_data in sample_metrics:
            metrics = ModelMetrics(**metrics_data)
            await metrics_collection.insert_one(metrics.dict(by_alias=True))
        
        logger.info(f"Inserted {len(sample_metrics)} sample model metrics")
        
    except Exception as e:
        logger.error("Error inserting sample data", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(init_database())

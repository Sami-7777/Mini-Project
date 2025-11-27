"""
Database connection and management for the cyberattack detection system.
"""
import asyncio
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
import redis.asyncio as redis
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import structlog

from ..core.config import settings
from .models import (
    AnalysisResult, Alert, ModelMetrics, SystemMetrics,
    ANALYSIS_INDEXES, ALERT_INDEXES, MODEL_METRICS_INDEXES, SYSTEM_METRICS_INDEXES
)

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Database connection manager for MongoDB and Redis."""
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_db: Optional[AsyncIOMotorDatabase] = None
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connections to MongoDB and Redis."""
        try:
            # Connect to MongoDB
            self.mongo_client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=10
            )
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            self.mongo_db = self.mongo_client.get_default_database()
            
            # Connect to Redis
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            self._connected = True
            logger.info("Database connections established successfully")
            
            # Create indexes
            await self._create_indexes()
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Close database connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
        self._connected = False
        logger.info("Database connections closed")
    
    async def _create_indexes(self) -> None:
        """Create database indexes for optimal performance."""
        try:
            # Analysis results indexes
            analysis_collection = self.get_collection("analysis_results")
            await analysis_collection.create_indexes(ANALYSIS_INDEXES)
            
            # Alerts indexes
            alerts_collection = self.get_collection("alerts")
            await alerts_collection.create_indexes(ALERT_INDEXES)
            
            # Model metrics indexes
            model_metrics_collection = self.get_collection("model_metrics")
            await model_metrics_collection.create_indexes(MODEL_METRICS_INDEXES)
            
            # System metrics indexes
            system_metrics_collection = self.get_collection("system_metrics")
            await system_metrics_collection.create_indexes(SYSTEM_METRICS_INDEXES)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error("Failed to create database indexes", error=str(e))
            raise
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a MongoDB collection."""
        if not self._connected or not self.mongo_db:
            raise RuntimeError("Database not connected")
        return self.mongo_db[collection_name]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connections."""
        health_status = {
            "mongodb": {"status": "disconnected", "latency_ms": None},
            "redis": {"status": "disconnected", "latency_ms": None}
        }
        
        # Check MongoDB
        try:
            import time
            start_time = time.time()
            await self.mongo_client.admin.command('ping')
            health_status["mongodb"] = {
                "status": "connected",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        except Exception as e:
            health_status["mongodb"]["error"] = str(e)
        
        # Check Redis
        try:
            import time
            start_time = time.time()
            await self.redis_client.ping()
            health_status["redis"] = {
                "status": "connected",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        except Exception as e:
            health_status["redis"]["error"] = str(e)
        
        return health_status
    
    @property
    def is_connected(self) -> bool:
        """Check if database connections are established."""
        return self._connected


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> DatabaseManager:
    """Get the database manager instance."""
    if not db_manager.is_connected:
        await db_manager.connect()
    return db_manager


async def close_database() -> None:
    """Close database connections."""
    await db_manager.disconnect()


# Dependency for FastAPI
async def get_db() -> DatabaseManager:
    """FastAPI dependency to get database connection."""
    return await get_database()


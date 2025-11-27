"""
Health check API endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import time
import structlog

from ...database.connection import db_manager
from ...models.model_manager import model_manager
from ...engine.hybrid_engine import hybrid_engine

logger = structlog.get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    uptime: float
    services: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        # Get database health
        db_health = await db_manager.health_check()
        
        # Get model manager status
        model_info = await model_manager.get_model_info()
        
        # Get engine statistics
        engine_stats = hybrid_engine.get_engine_statistics()
        
        # Determine overall status
        overall_status = "healthy"
        if db_health["mongodb"]["status"] != "connected":
            overall_status = "degraded"
        if db_health["redis"]["status"] != "connected":
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            version="1.0.0",
            uptime=time.time(),  # Would calculate actual uptime
            services={
                "database": db_health,
                "models": {
                    "total_models": model_info["total_models"],
                    "trained_models": model_info["trained_models"],
                    "total_ensembles": model_info["total_ensembles"],
                    "trained_ensembles": model_info["trained_ensembles"]
                },
                "engine": engine_stats
            }
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check endpoint."""
    try:
        # Get database health
        db_health = await db_manager.health_check()
        
        # Get model manager info
        model_info = await model_manager.get_model_info()
        
        # Get engine statistics
        engine_stats = hybrid_engine.get_engine_statistics()
        
        # Get threat intelligence stats
        from ...intelligence.threat_intelligence import threat_intelligence_manager
        ti_stats = await threat_intelligence_manager.get_threat_intelligence_stats()
        
        # Check each service
        services_status = {
            "database": {
                "status": "healthy" if db_health["mongodb"]["status"] == "connected" else "unhealthy",
                "details": db_health
            },
            "models": {
                "status": "healthy" if model_info["trained_models"] > 0 else "degraded",
                "details": model_info
            },
            "engine": {
                "status": "healthy",
                "details": engine_stats
            },
            "threat_intelligence": {
                "status": "healthy",
                "details": ti_stats
            }
        }
        
        # Determine overall status
        overall_status = "healthy"
        for service, status_info in services_status.items():
            if status_info["status"] == "unhealthy":
                overall_status = "unhealthy"
                break
            elif status_info["status"] == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": services_status
        }
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check if all required services are ready
        db_health = await db_manager.health_check()
        model_info = await model_manager.get_model_info()
        
        # Check database connectivity
        if db_health["mongodb"]["status"] != "connected":
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # Check if at least one model is trained
        if model_info["trained_models"] == 0:
            raise HTTPException(status_code=503, detail="No trained models available")
        
        return {
            "status": "ready",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/live")
async def liveness_check():
    """Liveness check endpoint."""
    return {
        "status": "alive",
        "timestamp": time.time()
    }


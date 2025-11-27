"""
FastAPI main application for the cyberattack detection system.
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog
import time
from typing import Dict, List, Optional, Any

from ..core.config import settings
from ..core.logger import logger
from ..database.connection import db_manager, get_database
from ..models.model_manager import model_manager
from ..engine.hybrid_engine import hybrid_engine
from ..intelligence.threat_intelligence import threat_intelligence_manager
from .routers import analysis, models, intelligence, health, alerts, ai_orchestrator
from .middleware import RateLimitMiddleware, SecurityMiddleware
from .dependencies import get_current_user, get_api_key

# Configure structured logging
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting cyberattack detection system API")
    
    try:
        # Initialize database connections
        await db_manager.connect()
        logger.info("Database connections established")
        
        # Initialize model manager
        await model_manager.initialize()
        logger.info("Model manager initialized")
        
        # Initialize threat intelligence manager
        logger.info("Threat intelligence manager initialized")
        
        yield
        
    except Exception as e:
        logger.error("Error during startup", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down cyberattack detection system API")
        await db_manager.disconnect()
        logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Cyberattack Detection System API",
    description="Real-time cyberattack detection system with hybrid ML and rule-based intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(intelligence.router, prefix="/api/v1", tags=["intelligence"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(ai_orchestrator.router, prefix="/api/v1", tags=["ai_orchestrator"])


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning("HTTP exception", 
                  status_code=exc.status_code, 
                  detail=exc.detail,
                  path=request.url.path)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", 
                error=str(exc), 
                path=request.url.path,
                method=request.method)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cyberattack Detection System API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": time.time()
    }


@app.get("/api/v1/status")
async def get_system_status():
    """Get system status."""
    try:
        # Get database health
        db_health = await db_manager.health_check()
        
        # Get model manager info
        model_info = await model_manager.get_model_info()
        
        # Get engine statistics
        engine_stats = hybrid_engine.get_engine_statistics()
        
        # Get threat intelligence stats
        ti_stats = await threat_intelligence_manager.get_threat_intelligence_stats()
        
        return {
            "status": "operational",
            "timestamp": time.time(),
            "database": db_health,
            "models": model_info,
            "engine": engine_stats,
            "threat_intelligence": ti_stats
        }
        
    except Exception as e:
        logger.error("Error getting system status", error=str(e))
        raise HTTPException(status_code=500, detail="Error getting system status")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True,
        log_level=settings.log_level.lower()
    )


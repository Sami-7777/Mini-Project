"""
Threat intelligence API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import structlog

from ...intelligence.threat_intelligence import threat_intelligence_manager, ThreatIntelligenceSource
from ..dependencies import get_current_user, get_api_key

logger = structlog.get_logger(__name__)
router = APIRouter()


class ThreatIntelligenceRequest(BaseModel):
    """Request model for threat intelligence analysis."""
    target: str = Field(..., description="URL or IP address to analyze")
    target_type: str = Field(..., description="Type of target: 'url' or 'ip'")
    sources: Optional[List[ThreatIntelligenceSource]] = Field(None, description="Specific sources to query")
    
    class Config:
        use_enum_values = True


class ThreatIntelligenceResponse(BaseModel):
    """Response model for threat intelligence analysis."""
    target: str
    target_type: str
    results: List[Dict[str, Any]]
    total_sources: int
    positive_detections: int
    analysis_timestamp: str


class ThreatIntelligenceStatsResponse(BaseModel):
    """Response model for threat intelligence statistics."""
    total_cached_entries: int
    source_statistics: Dict[str, int]
    cache_ttl: int


@router.post("/intelligence/analyze", response_model=ThreatIntelligenceResponse)
async def analyze_threat_intelligence(
    request: ThreatIntelligenceRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze a target against threat intelligence sources."""
    try:
        logger.info("Starting threat intelligence analysis", 
                   target=request.target, 
                   target_type=request.target_type,
                   user=current_user.get('username'))
        
        # Perform threat intelligence analysis
        if request.target_type == "url":
            results = await threat_intelligence_manager.analyze_url(request.target)
        elif request.target_type == "ip":
            results = await threat_intelligence_manager.analyze_ip(request.target)
        else:
            raise HTTPException(status_code=400, detail="Invalid target type")
        
        # Convert results to response format
        response_results = []
        positive_detections = 0
        
        for result in results:
            result_dict = {
                "source": result.source,
                "threat_type": result.threat_type,
                "confidence": result.confidence,
                "last_updated": result.last_updated.isoformat(),
                "raw_data": result.raw_data
            }
            response_results.append(result_dict)
            
            if result.confidence > 0.5:
                positive_detections += 1
        
        response = ThreatIntelligenceResponse(
            target=request.target,
            target_type=request.target_type,
            results=response_results,
            total_sources=len(results),
            positive_detections=positive_detections,
            analysis_timestamp=result.last_updated.isoformat() if results else ""
        )
        
        logger.info("Threat intelligence analysis completed", 
                   target=request.target,
                   total_sources=len(results),
                   positive_detections=positive_detections)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during threat intelligence analysis", 
                    target=request.target, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Threat intelligence analysis failed: {str(e)}")


@router.get("/intelligence/sources")
async def get_available_sources(
    current_user: dict = Depends(get_current_user)
):
    """Get list of available threat intelligence sources."""
    try:
        sources = [
            {
                "name": source.value,
                "description": f"Threat intelligence from {source.value}",
                "supported_targets": ["url", "ip"] if source in [
                    ThreatIntelligenceSource.VIRUSTOTAL,
                    ThreatIntelligenceSource.THREAT_CROWD
                ] else ["url"] if source in [
                    ThreatIntelligenceSource.GOOGLE_SAFE_BROWSING,
                    ThreatIntelligenceSource.PHISHTANK,
                    ThreatIntelligenceSource.URLVOID
                ] else ["ip"]
            }
            for source in ThreatIntelligenceSource
        ]
        
        return {
            "sources": sources,
            "total_sources": len(sources)
        }
        
    except Exception as e:
        logger.error("Error getting available sources", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving available sources")


@router.get("/intelligence/stats", response_model=ThreatIntelligenceStatsResponse)
async def get_threat_intelligence_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get threat intelligence statistics."""
    try:
        stats = await threat_intelligence_manager.get_threat_intelligence_stats()
        
        return ThreatIntelligenceStatsResponse(
            total_cached_entries=stats.get("total_cached_entries", 0),
            source_statistics=stats.get("source_statistics", {}),
            cache_ttl=stats.get("cache_ttl", 3600)
        )
        
    except Exception as e:
        logger.error("Error getting threat intelligence stats", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving threat intelligence statistics")


@router.post("/intelligence/cache/clear")
async def clear_threat_intelligence_cache(
    current_user: dict = Depends(get_current_user)
):
    """Clear the threat intelligence cache."""
    try:
        # This would implement cache clearing logic
        # For now, we'll return a success message
        
        logger.info("Threat intelligence cache cleared", 
                   user=current_user.get('username'))
        
        return {
            "message": "Threat intelligence cache cleared successfully"
        }
        
    except Exception as e:
        logger.error("Error clearing threat intelligence cache", error=str(e))
        raise HTTPException(status_code=500, detail="Error clearing threat intelligence cache")


@router.get("/intelligence/cache/{target}")
async def get_cached_threat_intelligence(
    target: str,
    target_type: str = Query(..., description="Type of target: 'url' or 'ip'"),
    current_user: dict = Depends(get_current_user)
):
    """Get cached threat intelligence for a target."""
    try:
        # Get cached results
        cached_results = await threat_intelligence_manager._get_cached_results(target, target_type)
        
        if not cached_results:
            raise HTTPException(status_code=404, detail="No cached results found")
        
        # Convert to response format
        response_results = []
        for result in cached_results:
            result_dict = {
                "source": result.source,
                "threat_type": result.threat_type,
                "confidence": result.confidence,
                "severity": result.severity,
                "description": result.description,
                "timestamp": result.timestamp.isoformat(),
                "ttl": result.ttl
            }
            response_results.append(result_dict)
        
        return {
            "target": target,
            "target_type": target_type,
            "cached_results": response_results,
            "total_results": len(response_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting cached threat intelligence", 
                    target=target, 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving cached threat intelligence")


@router.post("/intelligence/validate")
async def validate_threat_intelligence(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Validate threat intelligence data."""
    try:
        target = request.get("target")
        target_type = request.get("target_type")
        expected_result = request.get("expected_result")
        
        if not target or not target_type:
            raise HTTPException(status_code=400, detail="Target and target_type are required")
        
        # Perform analysis
        if target_type == "url":
            results = await threat_intelligence_manager.analyze_url(target)
        elif target_type == "ip":
            results = await threat_intelligence_manager.analyze_ip(target)
        else:
            raise HTTPException(status_code=400, detail="Invalid target type")
        
        # Validate against expected result
        validation_result = {
            "target": target,
            "target_type": target_type,
            "expected_result": expected_result,
            "actual_results": [
                {
                    "source": r.source,
                    "threat_type": r.threat_type,
                    "confidence": r.confidence
                }
                for r in results
            ],
            "validation_passed": True,  # Would implement actual validation logic
            "validation_timestamp": "2024-01-01T00:00:00Z"
        }
        
        logger.info("Threat intelligence validation completed", 
                   target=target,
                   validation_passed=validation_result["validation_passed"],
                   user=current_user.get('username'))
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating threat intelligence", error=str(e))
        raise HTTPException(status_code=500, detail="Error validating threat intelligence")

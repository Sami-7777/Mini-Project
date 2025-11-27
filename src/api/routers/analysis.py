"""
Analysis API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import structlog

from ...database.models import AnalysisResult, AnalysisStatus, AttackType, SeverityLevel
from ...database.connection import get_database
from ...engine.hybrid_engine import hybrid_engine
from ...features.feature_engine import feature_engine
from ..dependencies import get_current_user, get_api_key

logger = structlog.get_logger(__name__)
router = APIRouter()


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    target: str = Field(..., description="URL or IP address to analyze")
    target_type: str = Field(..., description="Type of target: 'url' or 'ip'")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    
    @validator('target_type')
    def validate_target_type(cls, v):
        if v not in ['url', 'ip']:
            raise ValueError('target_type must be either "url" or "ip"')
        return v


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    analysis_id: str
    target: str
    target_type: str
    status: AnalysisStatus
    attack_type: Optional[AttackType] = None
    confidence: Optional[float] = None
    severity: Optional[SeverityLevel] = None
    risk_score: Optional[float] = None
    analysis_duration_ms: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    targets: List[AnalysisRequest] = Field(..., description="List of targets to analyze")
    max_concurrent: int = Field(10, description="Maximum concurrent analyses")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    batch_id: str
    total_targets: int
    completed_analyses: int
    failed_analyses: int
    results: List[AnalysisResponse]
    created_at: datetime


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_target(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Analyze a single target (URL or IP)."""
    try:
        logger.info("Starting analysis", 
                   target=request.target, 
                   target_type=request.target_type,
                   user=current_user.get('username'))
        
        # Perform analysis
        analysis_result = await hybrid_engine.analyze(
            target=request.target,
            target_type=request.target_type,
            context=request.context
        )
        
        # Convert to response model
        response = AnalysisResponse(
            analysis_id=str(analysis_result.id),
            target=analysis_result.target_value,
            target_type=analysis_result.target_type,
            status=analysis_result.status,
            attack_type=analysis_result.final_attack_type,
            confidence=analysis_result.final_confidence,
            severity=analysis_result.severity,
            risk_score=analysis_result.risk_score,
            analysis_duration_ms=analysis_result.analysis_duration_ms,
            created_at=analysis_result.created_at,
            updated_at=analysis_result.updated_at
        )
        
        logger.info("Analysis completed", 
                   analysis_id=str(analysis_result.id),
                   attack_type=analysis_result.final_attack_type,
                   confidence=analysis_result.final_confidence)
        
        return response
        
    except Exception as e:
        logger.error("Error during analysis", 
                    target=request.target, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Analyze multiple targets in batch."""
    try:
        logger.info("Starting batch analysis", 
                   total_targets=len(request.targets),
                   user=current_user.get('username'))
        
        import asyncio
        from datetime import datetime
        
        # Create batch ID
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Process analyses with concurrency limit
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def analyze_with_semaphore(target_request: AnalysisRequest) -> AnalysisResponse:
            async with semaphore:
                try:
                    analysis_result = await hybrid_engine.analyze(
                        target=target_request.target,
                        target_type=target_request.target_type,
                        context=target_request.context
                    )
                    
                    return AnalysisResponse(
                        analysis_id=str(analysis_result.id),
                        target=analysis_result.target_value,
                        target_type=analysis_result.target_type,
                        status=analysis_result.status,
                        attack_type=analysis_result.final_attack_type,
                        confidence=analysis_result.final_confidence,
                        severity=analysis_result.severity,
                        risk_score=analysis_result.risk_score,
                        analysis_duration_ms=analysis_result.analysis_duration_ms,
                        created_at=analysis_result.created_at,
                        updated_at=analysis_result.updated_at
                    )
                except Exception as e:
                    logger.error("Error in batch analysis", 
                                target=target_request.target, 
                                error=str(e))
                    # Return failed analysis
                    return AnalysisResponse(
                        analysis_id="failed",
                        target=target_request.target,
                        target_type=target_request.target_type,
                        status=AnalysisStatus.FAILED,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
        
        # Execute batch analysis
        tasks = [analyze_with_semaphore(target) for target in request.targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed_analyses = 0
        failed_analyses = 0
        analysis_results = []
        
        for result in results:
            if isinstance(result, AnalysisResponse):
                analysis_results.append(result)
                if result.status == AnalysisStatus.COMPLETED:
                    completed_analyses += 1
                else:
                    failed_analyses += 1
            else:
                failed_analyses += 1
        
        response = BatchAnalysisResponse(
            batch_id=batch_id,
            total_targets=len(request.targets),
            completed_analyses=completed_analyses,
            failed_analyses=failed_analyses,
            results=analysis_results,
            created_at=datetime.utcnow()
        )
        
        logger.info("Batch analysis completed", 
                   batch_id=batch_id,
                   completed=completed_analyses,
                   failed=failed_analyses)
        
        return response
        
    except Exception as e:
        logger.error("Error during batch analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get analysis result by ID."""
    try:
        db = await get_database()
        collection = db.get_collection("analysis_results")
        
        analysis_doc = await collection.find_one({"_id": analysis_id})
        
        if not analysis_doc:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis_result = AnalysisResult(**analysis_doc)
        
        return AnalysisResponse(
            analysis_id=str(analysis_result.id),
            target=analysis_result.target_value,
            target_type=analysis_result.target_type,
            status=analysis_result.status,
            attack_type=analysis_result.final_attack_type,
            confidence=analysis_result.final_confidence,
            severity=analysis_result.severity,
            risk_score=analysis_result.risk_score,
            analysis_duration_ms=analysis_result.analysis_duration_ms,
            created_at=analysis_result.created_at,
            updated_at=analysis_result.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving analysis")


@router.get("/analyze/history/{target}")
async def get_analysis_history(
    target: str,
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get analysis history for a target."""
    try:
        history = await hybrid_engine.get_analysis_history(target, limit)
        
        responses = []
        for analysis_result in history:
            response = AnalysisResponse(
                analysis_id=str(analysis_result.id),
                target=analysis_result.target_value,
                target_type=analysis_result.target_type,
                status=analysis_result.status,
                attack_type=analysis_result.final_attack_type,
                confidence=analysis_result.final_confidence,
                severity=analysis_result.severity,
                risk_score=analysis_result.risk_score,
                analysis_duration_ms=analysis_result.analysis_duration_ms,
                created_at=analysis_result.created_at,
                updated_at=analysis_result.updated_at
            )
            responses.append(response)
        
        return {
            "target": target,
            "total_results": len(responses),
            "results": responses
        }
        
    except Exception as e:
        logger.error("Error getting analysis history", target=target, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving analysis history")


@router.post("/analyze/feedback/{analysis_id}")
async def submit_feedback(
    analysis_id: str,
    feedback: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for an analysis result."""
    try:
        db = await get_database()
        collection = db.get_collection("analysis_results")
        
        # Update analysis with feedback
        update_result = await collection.update_one(
            {"_id": analysis_id},
            {
                "$set": {
                    "user_feedback": feedback.get("feedback"),
                    "feedback_timestamp": datetime.utcnow()
                }
            }
        )
        
        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        logger.info("Feedback submitted", 
                   analysis_id=analysis_id,
                   feedback=feedback.get("feedback"),
                   user=current_user.get('username'))
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting feedback", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error submitting feedback")


@router.get("/analyze/stats")
async def get_analysis_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get analysis statistics."""
    try:
        db = await get_database()
        collection = db.get_collection("analysis_results")
        
        # Get total analyses
        total_analyses = await collection.count_documents({})
        
        # Get analyses by status
        status_pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_stats = {}
        async for doc in collection.aggregate(status_pipeline):
            status_stats[doc["_id"]] = doc["count"]
        
        # Get analyses by attack type
        attack_type_pipeline = [
            {"$group": {"_id": "$final_attack_type", "count": {"$sum": 1}}}
        ]
        attack_type_stats = {}
        async for doc in collection.aggregate(attack_type_pipeline):
            attack_type_stats[doc["_id"]] = doc["count"]
        
        # Get analyses by severity
        severity_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
        ]
        severity_stats = {}
        async for doc in collection.aggregate(severity_pipeline):
            severity_stats[doc["_id"]] = doc["count"]
        
        return {
            "total_analyses": total_analyses,
            "status_distribution": status_stats,
            "attack_type_distribution": attack_type_stats,
            "severity_distribution": severity_stats
        }
        
    except Exception as e:
        logger.error("Error getting analysis stats", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving analysis statistics")


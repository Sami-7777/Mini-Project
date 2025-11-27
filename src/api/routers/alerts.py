"""
Alerts API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

from ...database.models import Alert, SeverityLevel, AttackType
from ...database.connection import get_database
from ..dependencies import get_current_user, get_api_key

logger = structlog.get_logger(__name__)
router = APIRouter()


class AlertResponse(BaseModel):
    """Response model for alerts."""
    id: str
    analysis_id: str
    alert_type: str
    severity: SeverityLevel
    title: str
    description: str
    attack_type: Optional[AttackType] = None
    is_acknowledged: bool
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class AlertAcknowledgmentRequest(BaseModel):
    """Request model for alert acknowledgment."""
    acknowledged_by: str = Field(..., description="Username of the person acknowledging the alert")


class AlertFilter(BaseModel):
    """Filter model for alerts."""
    severity: Optional[List[SeverityLevel]] = None
    attack_type: Optional[List[AttackType]] = None
    alert_type: Optional[List[str]] = None
    is_acknowledged: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    severity: Optional[List[SeverityLevel]] = Query(None),
    attack_type: Optional[List[AttackType]] = Query(None),
    is_acknowledged: Optional[bool] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get alerts with filtering and pagination."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        # Build query filter
        query_filter = {}
        
        if severity:
            query_filter["severity"] = {"$in": severity}
        
        if attack_type:
            query_filter["attack_type"] = {"$in": attack_type}
        
        if is_acknowledged is not None:
            query_filter["is_acknowledged"] = is_acknowledged
        
        # Get alerts
        cursor = collection.find(query_filter).sort("created_at", -1).skip(offset).limit(limit)
        
        alerts = []
        async for doc in cursor:
            alert = Alert(**doc)
            alerts.append(AlertResponse(
                id=str(alert.id),
                analysis_id=str(alert.analysis_id),
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                attack_type=alert.attack_type,
                is_acknowledged=alert.is_acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
                created_at=alert.created_at,
                updated_at=alert.updated_at
            ))
        
        return alerts
        
    except Exception as e:
        logger.error("Error getting alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alerts")


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific alert by ID."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        alert_doc = await collection.find_one({"_id": alert_id})
        
        if not alert_doc:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert = Alert(**alert_doc)
        
        return AlertResponse(
            id=str(alert.id),
            analysis_id=str(alert.analysis_id),
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            description=alert.description,
            attack_type=alert.attack_type,
            is_acknowledged=alert.is_acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            created_at=alert.created_at,
            updated_at=alert.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alert")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an alert."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        # Update alert
        update_result = await collection.update_one(
            {"_id": alert_id},
            {
                "$set": {
                    "is_acknowledged": True,
                    "acknowledged_by": request.acknowledged_by,
                    "acknowledged_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        logger.info("Alert acknowledged", 
                   alert_id=alert_id,
                   acknowledged_by=request.acknowledged_by,
                   user=current_user.get('username'))
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error acknowledging alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error acknowledging alert")


@router.get("/alerts/stats")
async def get_alert_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get alert statistics."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        # Get total alerts
        total_alerts = await collection.count_documents({})
        
        # Get alerts by severity
        severity_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
        ]
        severity_stats = {}
        async for doc in collection.aggregate(severity_pipeline):
            severity_stats[doc["_id"]] = doc["count"]
        
        # Get alerts by attack type
        attack_type_pipeline = [
            {"$group": {"_id": "$attack_type", "count": {"$sum": 1}}}
        ]
        attack_type_stats = {}
        async for doc in collection.aggregate(attack_type_pipeline):
            attack_type_stats[doc["_id"]] = doc["count"]
        
        # Get alerts by acknowledgment status
        ack_pipeline = [
            {"$group": {"_id": "$is_acknowledged", "count": {"$sum": 1}}}
        ]
        ack_stats = {}
        async for doc in collection.aggregate(ack_pipeline):
            ack_stats[doc["_id"]] = doc["count"]
        
        # Get recent alerts (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = await collection.count_documents({
            "created_at": {"$gte": recent_cutoff}
        })
        
        return {
            "total_alerts": total_alerts,
            "recent_alerts_24h": recent_alerts,
            "severity_distribution": severity_stats,
            "attack_type_distribution": attack_type_stats,
            "acknowledgment_distribution": ack_stats
        }
        
    except Exception as e:
        logger.error("Error getting alert stats", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alert statistics")


@router.delete("/alerts/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an alert."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        delete_result = await collection.delete_one({"_id": alert_id})
        
        if delete_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        logger.info("Alert deleted", 
                   alert_id=alert_id,
                   user=current_user.get('username'))
        
        return {"message": "Alert deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error deleting alert")


@router.get("/alerts/unacknowledged")
async def get_unacknowledged_alerts(
    limit: int = Query(50, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get unacknowledged alerts."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        # Get unacknowledged alerts
        cursor = collection.find({"is_acknowledged": False}).sort("created_at", -1).limit(limit)
        
        alerts = []
        async for doc in cursor:
            alert = Alert(**doc)
            alerts.append(AlertResponse(
                id=str(alert.id),
                analysis_id=str(alert.analysis_id),
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                attack_type=alert.attack_type,
                is_acknowledged=alert.is_acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
                created_at=alert.created_at,
                updated_at=alert.updated_at
            ))
        
        return {
            "unacknowledged_alerts": alerts,
            "total_count": len(alerts)
        }
        
    except Exception as e:
        logger.error("Error getting unacknowledged alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving unacknowledged alerts")


@router.post("/alerts/bulk/acknowledge")
async def bulk_acknowledge_alerts(
    alert_ids: List[str],
    request: AlertAcknowledgmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Bulk acknowledge multiple alerts."""
    try:
        db = await get_database()
        collection = db.get_collection("alerts")
        
        # Update multiple alerts
        update_result = await collection.update_many(
            {"_id": {"$in": alert_ids}},
            {
                "$set": {
                    "is_acknowledged": True,
                    "acknowledged_by": request.acknowledged_by,
                    "acknowledged_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info("Bulk alert acknowledgment completed", 
                   acknowledged_count=update_result.modified_count,
                   acknowledged_by=request.acknowledged_by,
                   user=current_user.get('username'))
        
        return {
            "message": "Bulk acknowledgment completed",
            "acknowledged_count": update_result.modified_count
        }
        
    except Exception as e:
        logger.error("Error bulk acknowledging alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Error bulk acknowledging alerts")

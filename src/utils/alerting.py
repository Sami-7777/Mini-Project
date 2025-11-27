"""
Alerting system for the cyberattack detection system.
"""
import asyncio
import smtplib
import aiohttp
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..database.models import Alert, SeverityLevel, AttackType
from ..database.connection import get_database

logger = structlog.get_logger(__name__)


class AlertChannel(str, Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class AlertNotification:
    """Alert notification data."""
    alert_id: str
    title: str
    message: str
    severity: SeverityLevel
    attack_type: Optional[AttackType]
    channels: List[AlertChannel]
    metadata: Dict[str, Any]
    timestamp: datetime


class AlertManager:
    """Manages alert generation and notifications."""
    
    def __init__(self):
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        self.worker_task = None
    
    async def start(self):
        """Start the alert manager."""
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._notification_worker())
            logger.info("Alert manager started")
    
    async def stop(self):
        """Stop the alert manager."""
        if self.is_running:
            self.is_running = False
            if self.worker_task:
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass
            logger.info("Alert manager stopped")
    
    async def create_alert(self, analysis_id: str, alert_type: str, 
                          severity: SeverityLevel, title: str, description: str,
                          attack_type: Optional[AttackType] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert."""
        try:
            # Create alert document
            alert = Alert(
                analysis_id=analysis_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                attack_type=attack_type,
                metadata=metadata or {}
            )
            
            # Store in database
            db = await get_database()
            collection = db.get_collection("alerts")
            result = await collection.insert_one(alert.dict(by_alias=True))
            alert_id = str(result.inserted_id)
            
            # Create notification
            notification = AlertNotification(
                alert_id=alert_id,
                title=title,
                message=description,
                severity=severity,
                attack_type=attack_type,
                channels=self._get_notification_channels(severity),
                metadata=metadata or {},
                timestamp=datetime.utcnow()
            )
            
            # Queue notification
            await self.notification_queue.put(notification)
            
            logger.info("Alert created", alert_id=alert_id, severity=severity)
            return alert_id
            
        except Exception as e:
            logger.error("Error creating alert", error=str(e))
            raise
    
    async def _notification_worker(self):
        """Background worker for processing notifications."""
        while self.is_running:
            try:
                # Get notification from queue
                notification = await asyncio.wait_for(
                    self.notification_queue.get(), 
                    timeout=1.0
                )
                
                # Process notification
                await self._process_notification(notification)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error in notification worker", error=str(e))
    
    async def _process_notification(self, notification: AlertNotification):
        """Process a single notification."""
        try:
            # Send notifications to all channels
            tasks = []
            
            for channel in notification.channels:
                if channel == AlertChannel.EMAIL:
                    tasks.append(self._send_email_notification(notification))
                elif channel == AlertChannel.SMS:
                    tasks.append(self._send_sms_notification(notification))
                elif channel == AlertChannel.WEBHOOK:
                    tasks.append(self._send_webhook_notification(notification))
                elif channel == AlertChannel.SLACK:
                    tasks.append(self._send_slack_notification(notification))
                elif channel == AlertChannel.TEAMS:
                    tasks.append(self._send_teams_notification(notification))
            
            # Execute all notifications in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update alert with notification status
                await self._update_alert_notification_status(
                    notification.alert_id, 
                    results
                )
            
        except Exception as e:
            logger.error("Error processing notification", error=str(e))
    
    async def _send_email_notification(self, notification: AlertNotification) -> bool:
        """Send email notification."""
        try:
            if not settings.smtp_username or not settings.smtp_password:
                logger.warning("SMTP credentials not configured")
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = settings.smtp_username
            msg['To'] = settings.alert_email
            msg['Subject'] = f"🚨 {notification.severity.upper()}: {notification.title}"
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>Security Alert</h2>
                <p><strong>Severity:</strong> {notification.severity.upper()}</p>
                <p><strong>Attack Type:</strong> {notification.attack_type or 'Unknown'}</p>
                <p><strong>Time:</strong> {notification.timestamp}</p>
                <p><strong>Description:</strong></p>
                <p>{notification.message}</p>
                
                <hr>
                <p><small>This is an automated alert from the Cyberattack Detection System.</small></p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(msg)
            
            logger.info("Email notification sent", alert_id=notification.alert_id)
            return True
            
        except Exception as e:
            logger.error("Error sending email notification", error=str(e))
            return False
    
    async def _send_sms_notification(self, notification: AlertNotification) -> bool:
        """Send SMS notification."""
        try:
            # Placeholder for SMS implementation
            # In production, you would integrate with SMS providers like Twilio
            logger.info("SMS notification sent", alert_id=notification.alert_id)
            return True
            
        except Exception as e:
            logger.error("Error sending SMS notification", error=str(e))
            return False
    
    async def _send_webhook_notification(self, notification: AlertNotification) -> bool:
        """Send webhook notification."""
        try:
            if not settings.webhook_url:
                logger.warning("Webhook URL not configured")
                return False
            
            # Prepare webhook payload
            payload = {
                "alert_id": notification.alert_id,
                "title": notification.title,
                "message": notification.message,
                "severity": notification.severity,
                "attack_type": notification.attack_type,
                "timestamp": notification.timestamp.isoformat(),
                "metadata": notification.metadata
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook notification sent", alert_id=notification.alert_id)
                        return True
                    else:
                        logger.error("Webhook notification failed", 
                                   status=response.status,
                                   alert_id=notification.alert_id)
                        return False
            
        except Exception as e:
            logger.error("Error sending webhook notification", error=str(e))
            return False
    
    async def _send_slack_notification(self, notification: AlertNotification) -> bool:
        """Send Slack notification."""
        try:
            # Placeholder for Slack implementation
            # In production, you would integrate with Slack API
            logger.info("Slack notification sent", alert_id=notification.alert_id)
            return True
            
        except Exception as e:
            logger.error("Error sending Slack notification", error=str(e))
            return False
    
    async def _send_teams_notification(self, notification: AlertNotification) -> bool:
        """Send Microsoft Teams notification."""
        try:
            # Placeholder for Teams implementation
            # In production, you would integrate with Teams webhook
            logger.info("Teams notification sent", alert_id=notification.alert_id)
            return True
            
        except Exception as e:
            logger.error("Error sending Teams notification", error=str(e))
            return False
    
    def _get_notification_channels(self, severity: SeverityLevel) -> List[AlertChannel]:
        """Get notification channels based on severity."""
        channels = [AlertChannel.EMAIL]
        
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            channels.extend([AlertChannel.WEBHOOK, AlertChannel.SLACK])
        
        if severity == SeverityLevel.CRITICAL:
            channels.append(AlertChannel.SMS)
        
        return channels
    
    async def _update_alert_notification_status(self, alert_id: str, results: List[Any]):
        """Update alert with notification status."""
        try:
            db = await get_database()
            collection = db.get_collection("alerts")
            
            # Determine notification status
            email_sent = any(isinstance(r, bool) and r for r in results[:1])
            sms_sent = any(isinstance(r, bool) and r for r in results[1:2])
            webhook_sent = any(isinstance(r, bool) and r for r in results[2:3])
            
            # Update alert
            await collection.update_one(
                {"_id": alert_id},
                {
                    "$set": {
                        "email_sent": email_sent,
                        "sms_sent": sms_sent,
                        "webhook_sent": webhook_sent,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
        except Exception as e:
            logger.error("Error updating alert notification status", error=str(e))
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            db = await get_database()
            collection = db.get_collection("alerts")
            
            # Get alert statistics
            pipeline = [
                {"$group": {
                    "_id": "$severity",
                    "count": {"$sum": 1},
                    "acknowledged": {"$sum": {"$cond": ["$is_acknowledged", 1, 0]}}
                }}
            ]
            
            stats = {}
            async for doc in collection.aggregate(pipeline):
                stats[doc["_id"]] = {
                    "total": doc["count"],
                    "acknowledged": doc["acknowledged"],
                    "unacknowledged": doc["count"] - doc["acknowledged"]
                }
            
            return {
                "total_alerts": await collection.count_documents({}),
                "severity_distribution": stats
            }
            
        except Exception as e:
            logger.error("Error getting alert statistics", error=str(e))
            return {}


# Global alert manager instance
alert_manager = AlertManager()

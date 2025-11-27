"""
Threat intelligence integration for cyberattack detection.
"""
import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..database.models import ThreatIntelligence
from ..database.connection import get_database

logger = structlog.get_logger(__name__)


class ThreatIntelligenceSource(str, Enum):
    """Available threat intelligence sources."""
    VIRUSTOTAL = "virustotal"
    GOOGLE_SAFE_BROWSING = "google_safe_browsing"
    ABUSEIPDB = "abuseipdb"
    SHODAN = "shodan"
    MALWARE_DOMAIN_LIST = "malware_domain_list"
    PHISHTANK = "phishtank"
    URLVOID = "urlvoid"
    THREAT_CROWD = "threat_crowd"


@dataclass
class ThreatIntelligenceResult:
    """Result from threat intelligence analysis."""
    source: ThreatIntelligenceSource
    target: str
    target_type: str
    threat_type: str
    confidence: float
    severity: str
    description: str
    raw_data: Dict[str, Any]
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds


class ThreatIntelligenceManager:
    """Manages threat intelligence from multiple sources."""
    
    def __init__(self):
        self.sources = {
            ThreatIntelligenceSource.VIRUSTOTAL: self._query_virustotal,
            ThreatIntelligenceSource.GOOGLE_SAFE_BROWSING: self._query_google_safe_browsing,
            ThreatIntelligenceSource.ABUSEIPDB: self._query_abuseipdb,
            ThreatIntelligenceSource.SHODAN: self._query_shodan,
            ThreatIntelligenceSource.MALWARE_DOMAIN_LIST: self._query_malware_domain_list,
            ThreatIntelligenceSource.PHISHTANK: self._query_phishtank,
            ThreatIntelligenceSource.URLVOID: self._query_urlvoid,
            ThreatIntelligenceSource.THREAT_CROWD: self._query_threat_crowd
        }
        
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour default TTL
    
    async def analyze_url(self, url: str) -> List[ThreatIntelligence]:
        """Analyze URL against all threat intelligence sources."""
        results = []
        
        # Get cached results first
        cached_results = await self._get_cached_results(url, "url")
        if cached_results:
            results.extend(cached_results)
        
        # Query sources that don't have cached results
        sources_to_query = [
            ThreatIntelligenceSource.VIRUSTOTAL,
            ThreatIntelligenceSource.GOOGLE_SAFE_BROWSING,
            ThreatIntelligenceSource.PHISHTANK,
            ThreatIntelligenceSource.URLVOID,
            ThreatIntelligenceSource.THREAT_CROWD
        ]
        
        tasks = []
        for source in sources_to_query:
            if source not in [r.source for r in cached_results]:
                task = self._query_source(source, url, "url")
                tasks.append(task)
        
        if tasks:
            source_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in source_results:
                if isinstance(result, ThreatIntelligenceResult):
                    results.append(result)
                    # Cache the result
                    await self._cache_result(result)
        
        # Convert to ThreatIntelligence objects
        threat_intel_objects = []
        for result in results:
            threat_intel = ThreatIntelligence(
                source=result.source,
                source_id=result.target,
                threat_type=result.threat_type,
                confidence=result.confidence,
                last_updated=result.timestamp,
                raw_data=result.raw_data
            )
            threat_intel_objects.append(threat_intel)
        
        return threat_intel_objects
    
    async def analyze_ip(self, ip_address: str) -> List[ThreatIntelligence]:
        """Analyze IP address against all threat intelligence sources."""
        results = []
        
        # Get cached results first
        cached_results = await self._get_cached_results(ip_address, "ip")
        if cached_results:
            results.extend(cached_results)
        
        # Query sources that don't have cached results
        sources_to_query = [
            ThreatIntelligenceSource.VIRUSTOTAL,
            ThreatIntelligenceSource.ABUSEIPDB,
            ThreatIntelligenceSource.SHODAN,
            ThreatIntelligenceSource.THREAT_CROWD
        ]
        
        tasks = []
        for source in sources_to_query:
            if source not in [r.source for r in cached_results]:
                task = self._query_source(source, ip_address, "ip")
                tasks.append(task)
        
        if tasks:
            source_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in source_results:
                if isinstance(result, ThreatIntelligenceResult):
                    results.append(result)
                    # Cache the result
                    await self._cache_result(result)
        
        # Convert to ThreatIntelligence objects
        threat_intel_objects = []
        for result in results:
            threat_intel = ThreatIntelligence(
                source=result.source,
                source_id=result.target,
                threat_type=result.threat_type,
                confidence=result.confidence,
                last_updated=result.timestamp,
                raw_data=result.raw_data
            )
            threat_intel_objects.append(threat_intel)
        
        return threat_intel_objects
    
    async def _query_source(self, source: ThreatIntelligenceSource, 
                           target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query a specific threat intelligence source."""
        try:
            query_func = self.sources.get(source)
            if query_func:
                return await query_func(target, target_type)
            else:
                logger.warning("Unknown threat intelligence source", source=source)
                return None
        except Exception as e:
            logger.error("Error querying threat intelligence source", 
                        source=source, target=target, error=str(e))
            return None
    
    async def _query_virustotal(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query VirusTotal API."""
        if not settings.virustotal_api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                if target_type == "url":
                    url = "https://www.virustotal.com/vtapi/v2/url/report"
                    params = {
                        'apikey': settings.virustotal_api_key,
                        'resource': target
                    }
                elif target_type == "ip":
                    url = "https://www.virustotal.com/vtapi/v2/ip-address/report"
                    params = {
                        'apikey': settings.virustotal_api_key,
                        'ip': target
                    }
                else:
                    return None
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('response_code') == 1:
                            # Parse results
                            positives = data.get('positives', 0)
                            total = data.get('total', 0)
                            confidence = positives / total if total > 0 else 0.0
                            
                            # Determine threat type
                            threat_type = "malware"
                            if positives > 0:
                                if target_type == "url":
                                    threat_type = "malicious_url"
                                else:
                                    threat_type = "malicious_ip"
                            
                            return ThreatIntelligenceResult(
                                source=ThreatIntelligenceSource.VIRUSTOTAL,
                                target=target,
                                target_type=target_type,
                                threat_type=threat_type,
                                confidence=confidence,
                                severity="high" if confidence > 0.5 else "medium",
                                description=f"VirusTotal detected {positives}/{total} positives",
                                raw_data=data,
                                timestamp=datetime.utcnow(),
                                ttl=3600
                            )
        
        except Exception as e:
            logger.error("Error querying VirusTotal", target=target, error=str(e))
        
        return None
    
    async def _query_google_safe_browsing(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query Google Safe Browsing API."""
        if not settings.google_safe_browsing_api_key or target_type != "url":
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={settings.google_safe_browsing_api_key}"
                
                payload = {
                    "client": {
                        "clientId": "cyberattack-detection-system",
                        "clientVersion": "1.0.0"
                    },
                    "threatInfo": {
                        "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                        "platformTypes": ["ANY_PLATFORM"],
                        "threatEntryTypes": ["URL"],
                        "threatEntries": [{"url": target}]
                    }
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'matches' in data and data['matches']:
                            match = data['matches'][0]
                            threat_type = match.get('threatType', 'unknown')
                            confidence = 1.0  # Google Safe Browsing is binary
                            
                            return ThreatIntelligenceResult(
                                source=ThreatIntelligenceSource.GOOGLE_SAFE_BROWSING,
                                target=target,
                                target_type=target_type,
                                threat_type=threat_type.lower(),
                                confidence=confidence,
                                severity="high",
                                description=f"Google Safe Browsing detected {threat_type}",
                                raw_data=data,
                                timestamp=datetime.utcnow(),
                                ttl=3600
                            )
        
        except Exception as e:
            logger.error("Error querying Google Safe Browsing", target=target, error=str(e))
        
        return None
    
    async def _query_abuseipdb(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query AbuseIPDB API."""
        if not settings.abuseipdb_api_key or target_type != "ip":
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.abuseipdb.com/api/v2/check"
                headers = {
                    'Key': settings.abuseipdb_api_key,
                    'Accept': 'application/json'
                }
                params = {
                    'ipAddress': target,
                    'maxAgeInDays': 90,
                    'verbose': ''
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            ip_data = data['data']
                            abuse_confidence = ip_data.get('abuseConfidencePercentage', 0)
                            confidence = abuse_confidence / 100.0
                            
                            if confidence > 0:
                                threat_type = "abusive_ip"
                                severity = "high" if confidence > 0.8 else "medium"
                                
                                return ThreatIntelligenceResult(
                                    source=ThreatIntelligenceSource.ABUSEIPDB,
                                    target=target,
                                    target_type=target_type,
                                    threat_type=threat_type,
                                    confidence=confidence,
                                    severity=severity,
                                    description=f"AbuseIPDB confidence: {abuse_confidence}%",
                                    raw_data=data,
                                    timestamp=datetime.utcnow(),
                                    ttl=3600
                                )
        
        except Exception as e:
            logger.error("Error querying AbuseIPDB", target=target, error=str(e))
        
        return None
    
    async def _query_shodan(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query Shodan API."""
        if not settings.shodan_api_key or target_type != "ip":
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.shodan.io/shodan/host/{target}"
                params = {'key': settings.shodan_api_key}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Analyze Shodan data for threats
                        threat_indicators = []
                        confidence = 0.0
                        
                        # Check for known malicious services
                        if 'vulns' in data:
                            threat_indicators.append("vulnerabilities")
                            confidence += 0.3
                        
                        if 'tags' in data:
                            tags = data['tags']
                            if any(tag in ['malware', 'botnet', 'c2'] for tag in tags):
                                threat_indicators.append("malicious_tags")
                                confidence += 0.5
                        
                        if confidence > 0:
                            return ThreatIntelligenceResult(
                                source=ThreatIntelligenceSource.SHODAN,
                                target=target,
                                target_type=target_type,
                                threat_type="suspicious_ip",
                                confidence=min(confidence, 1.0),
                                severity="medium",
                                description=f"Shodan detected: {', '.join(threat_indicators)}",
                                raw_data=data,
                                timestamp=datetime.utcnow(),
                                ttl=3600
                            )
        
        except Exception as e:
            logger.error("Error querying Shodan", target=target, error=str(e))
        
        return None
    
    async def _query_malware_domain_list(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query Malware Domain List."""
        if target_type != "url":
            return None
        
        try:
            # This would typically query a local database or API
            # For now, we'll return None as a placeholder
            return None
        
        except Exception as e:
            logger.error("Error querying Malware Domain List", target=target, error=str(e))
        
        return None
    
    async def _query_phishtank(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query PhishTank API."""
        if target_type != "url":
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "http://checkurl.phishtank.com/checkurl/"
                data = {
                    'url': target,
                    'format': 'json',
                    'app_key': 'your_phishtank_api_key'  # Would need to add to config
                }
                
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get('results', {}).get('in_database'):
                            return ThreatIntelligenceResult(
                                source=ThreatIntelligenceSource.PHISHTANK,
                                target=target,
                                target_type=target_type,
                                threat_type="phishing",
                                confidence=1.0,
                                severity="high",
                                description="PhishTank confirmed phishing URL",
                                raw_data=result,
                                timestamp=datetime.utcnow(),
                                ttl=3600
                            )
        
        except Exception as e:
            logger.error("Error querying PhishTank", target=target, error=str(e))
        
        return None
    
    async def _query_urlvoid(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query URLVoid API."""
        if target_type != "url":
            return None
        
        try:
            # Placeholder for URLVoid integration
            return None
        
        except Exception as e:
            logger.error("Error querying URLVoid", target=target, error=str(e))
        
        return None
    
    async def _query_threat_crowd(self, target: str, target_type: str) -> ThreatIntelligenceResult:
        """Query ThreatCrowd API."""
        try:
            async with aiohttp.ClientSession() as session:
                if target_type == "url":
                    url = f"https://www.threatcrowd.org/searchApi/v2/domain/report/?domain={target}"
                elif target_type == "ip":
                    url = f"https://www.threatcrowd.org/searchApi/v2/ip/report/?ip={target}"
                else:
                    return None
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('response_code') == '1':
                            # Analyze ThreatCrowd data
                            confidence = 0.0
                            threat_type = "suspicious"
                            
                            if data.get('malware_samples'):
                                confidence += 0.4
                                threat_type = "malware"
                            
                            if data.get('resolutions'):
                                confidence += 0.2
                            
                            if confidence > 0:
                                return ThreatIntelligenceResult(
                                    source=ThreatIntelligenceSource.THREAT_CROWD,
                                    target=target,
                                    target_type=target_type,
                                    threat_type=threat_type,
                                    confidence=min(confidence, 1.0),
                                    severity="medium",
                                    description="ThreatCrowd detected suspicious activity",
                                    raw_data=data,
                                    timestamp=datetime.utcnow(),
                                    ttl=3600
                                )
        
        except Exception as e:
            logger.error("Error querying ThreatCrowd", target=target, error=str(e))
        
        return None
    
    async def _get_cached_results(self, target: str, target_type: str) -> List[ThreatIntelligenceResult]:
        """Get cached threat intelligence results."""
        try:
            db = await get_database()
            collection = db.get_collection("threat_intelligence_cache")
            
            # Generate cache key
            cache_key = hashlib.md5(f"{target}:{target_type}".encode()).hexdigest()
            
            # Check cache
            cached_doc = await collection.find_one({"cache_key": cache_key})
            
            if cached_doc:
                # Check if cache is still valid
                cached_time = cached_doc['timestamp']
                ttl = cached_doc.get('ttl', self.cache_ttl)
                
                if datetime.utcnow() - cached_time < timedelta(seconds=ttl):
                    return cached_doc['results']
                else:
                    # Remove expired cache
                    await collection.delete_one({"cache_key": cache_key})
        
        except Exception as e:
            logger.error("Error getting cached results", target=target, error=str(e))
        
        return []
    
    async def _cache_result(self, result: ThreatIntelligenceResult) -> None:
        """Cache a threat intelligence result."""
        try:
            db = await get_database()
            collection = db.get_collection("threat_intelligence_cache")
            
            # Generate cache key
            cache_key = hashlib.md5(f"{result.target}:{result.target_type}".encode()).hexdigest()
            
            # Store in cache
            cache_doc = {
                "cache_key": cache_key,
                "target": result.target,
                "target_type": result.target_type,
                "results": [result],
                "timestamp": result.timestamp,
                "ttl": result.ttl or self.cache_ttl
            }
            
            await collection.replace_one(
                {"cache_key": cache_key},
                cache_doc,
                upsert=True
            )
        
        except Exception as e:
            logger.error("Error caching result", target=result.target, error=str(e))
    
    async def get_threat_intelligence_stats(self) -> Dict[str, Any]:
        """Get threat intelligence statistics."""
        try:
            db = await get_database()
            collection = db.get_collection("threat_intelligence_cache")
            
            # Get cache statistics
            total_cached = await collection.count_documents({})
            
            # Get source statistics
            pipeline = [
                {"$unwind": "$results"},
                {"$group": {
                    "_id": "$results.source",
                    "count": {"$sum": 1}
                }}
            ]
            
            source_stats = {}
            async for doc in collection.aggregate(pipeline):
                source_stats[doc["_id"]] = doc["count"]
            
            return {
                "total_cached_entries": total_cached,
                "source_statistics": source_stats,
                "cache_ttl": self.cache_ttl
            }
        
        except Exception as e:
            logger.error("Error getting threat intelligence stats", error=str(e))
            return {}


# Global threat intelligence manager instance
threat_intelligence_manager = ThreatIntelligenceManager()


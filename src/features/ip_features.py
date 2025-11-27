"""
IP address feature extraction for cyberattack detection.
"""
import ipaddress
import socket
import struct
import math
from typing import Dict, List, Optional, Tuple
import geoip2.database
import geoip2.errors
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import structlog

from ..database.models import IPFeatures
from ..core.config import settings

logger = structlog.get_logger(__name__)


class IPFeatureExtractor:
    """Extract comprehensive features from IP addresses for attack detection."""
    
    def __init__(self):
        self.geoip_reader = None
        self._initialize_geoip()
        
        # Known malicious IP ranges and patterns
        self.malicious_ranges = [
            # Add known malicious IP ranges here
            "10.0.0.0/8",  # Private range (example)
            "192.168.0.0/16",  # Private range (example)
        ]
        
        # High-risk countries (configurable)
        self.high_risk_countries = {
            'CN', 'RU', 'KP', 'IR', 'SY'  # Add more as needed
        }
    
    def _initialize_geoip(self):
        """Initialize GeoIP database reader."""
        try:
            # Try to load MaxMind GeoLite2 database
            self.geoip_reader = geoip2.database.Reader('data/GeoLite2-City.mmdb')
        except Exception as e:
            logger.warning("Could not load GeoIP database", error=str(e))
            self.geoip_reader = None
    
    async def extract_features(self, ip_address: str) -> IPFeatures:
        """Extract all IP features."""
        try:
            # Validate IP address
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Basic IP features
            basic_features = self._extract_basic_features(ip_obj)
            
            # Geolocation features
            geo_features = await self._extract_geolocation_features(ip_address)
            
            # Network analysis
            network_features = await self._extract_network_features(ip_address)
            
            # Reputation analysis
            reputation_features = await self._extract_reputation_features(ip_address)
            
            # Behavioral patterns
            behavioral_features = await self._extract_behavioral_features(ip_address)
            
            return IPFeatures(
                ip_address=ip_address,
                **basic_features,
                **geo_features,
                **network_features,
                **reputation_features,
                **behavioral_features
            )
            
        except Exception as e:
            logger.error("Error extracting IP features", ip=ip_address, error=str(e))
            return self._get_minimal_features(ip_address)
    
    def _extract_basic_features(self, ip_obj) -> Dict:
        """Extract basic IP address features."""
        return {
            'is_private': ip_obj.is_private,
            'is_reserved': ip_obj.is_reserved,
        }
    
    async def _extract_geolocation_features(self, ip_address: str) -> Dict:
        """Extract geolocation features."""
        features = {
            'country': None,
            'region': None,
            'city': None,
            'latitude': None,
            'longitude': None,
            'timezone': None
        }
        
        if self.geoip_reader:
            try:
                response = self.geoip_reader.city(ip_address)
                features.update({
                    'country': response.country.iso_code,
                    'region': response.subdivisions.most_specific.name,
                    'city': response.city.name,
                    'latitude': float(response.location.latitude),
                    'longitude': float(response.location.longitude),
                    'timezone': response.location.time_zone
                })
            except geoip2.errors.AddressNotFoundError:
                logger.debug("IP address not found in GeoIP database", ip=ip_address)
            except Exception as e:
                logger.warning("Error querying GeoIP database", ip=ip_address, error=str(e))
        
        return features
    
    async def _extract_network_features(self, ip_address: str) -> Dict:
        """Extract network-related features."""
        features = {
            'asn': None,
            'asn_organization': None,
            'isp': None
        }
        
        try:
            # Use Shodan API if available
            if settings.shodan_api_key:
                shodan_data = await self._query_shodan(ip_address)
                if shodan_data:
                    features.update({
                        'asn': shodan_data.get('asn'),
                        'asn_organization': shodan_data.get('org'),
                        'isp': shodan_data.get('isp')
                    })
        except Exception as e:
            logger.warning("Error extracting network features", ip=ip_address, error=str(e))
        
        return features
    
    async def _extract_reputation_features(self, ip_address: str) -> Dict:
        """Extract reputation and threat intelligence features."""
        features = {
            'reputation_score': None,
            'abuse_score': None,
            'threat_types': []
        }
        
        try:
            # Query multiple threat intelligence sources
            tasks = [
                self._query_abuseipdb(ip_address),
                self._query_virustotal_ip(ip_address),
                self._check_malicious_ranges(ip_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process AbuseIPDB results
            if not isinstance(results[0], Exception) and results[0]:
                abuse_data = results[0]
                features['abuse_score'] = abuse_data.get('abuseConfidencePercentage', 0) / 100.0
                features['threat_types'].extend(abuse_data.get('usageTypes', []))
            
            # Process VirusTotal results
            if not isinstance(results[1], Exception) and results[1]:
                vt_data = results[1]
                features['reputation_score'] = vt_data.get('reputation', 0) / 100.0
                features['threat_types'].extend(vt_data.get('detected_urls', []))
            
            # Process malicious range check
            if not isinstance(results[2], Exception) and results[2]:
                features['threat_types'].append('known_malicious_range')
                
        except Exception as e:
            logger.warning("Error extracting reputation features", ip=ip_address, error=str(e))
        
        return features
    
    async def _extract_behavioral_features(self, ip_address: str) -> Dict:
        """Extract behavioral pattern features."""
        features = {
            'request_frequency': None,
            'unique_user_agents': None,
            'common_ports': []
        }
        
        try:
            # This would typically query your own logs/database
            # For now, we'll use placeholder values
            features.update({
                'request_frequency': 0.0,  # requests per minute
                'unique_user_agents': 0,
                'common_ports': []
            })
            
        except Exception as e:
            logger.warning("Error extracting behavioral features", ip=ip_address, error=str(e))
        
        return features
    
    async def _query_shodan(self, ip_address: str) -> Optional[Dict]:
        """Query Shodan API for IP information."""
        if not settings.shodan_api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.shodan.io/shodan/host/{ip_address}"
                params = {'key': settings.shodan_api_key}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.warning("Error querying Shodan", ip=ip_address, error=str(e))
        
        return None
    
    async def _query_abuseipdb(self, ip_address: str) -> Optional[Dict]:
        """Query AbuseIPDB API for IP reputation."""
        if not settings.abuseipdb_api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.abuseipdb.com/api/v2/check"
                headers = {
                    'Key': settings.abuseipdb_api_key,
                    'Accept': 'application/json'
                }
                params = {
                    'ipAddress': ip_address,
                    'maxAgeInDays': 90,
                    'verbose': ''
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {})
        except Exception as e:
            logger.warning("Error querying AbuseIPDB", ip=ip_address, error=str(e))
        
        return None
    
    async def _query_virustotal_ip(self, ip_address: str) -> Optional[Dict]:
        """Query VirusTotal API for IP reputation."""
        if not settings.virustotal_api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.virustotal.com/vtapi/v2/ip-address/report"
                params = {
                    'apikey': settings.virustotal_api_key,
                    'ip': ip_address
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.warning("Error querying VirusTotal", ip=ip_address, error=str(e))
        
        return None
    
    def _check_malicious_ranges(self, ip_address: str) -> bool:
        """Check if IP is in known malicious ranges."""
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            for range_str in self.malicious_ranges:
                if ip_obj in ipaddress.ip_network(range_str):
                    return True
        except Exception as e:
            logger.warning("Error checking malicious ranges", ip=ip_address, error=str(e))
        
        return False
    
    def _get_minimal_features(self, ip_address: str) -> IPFeatures:
        """Return minimal features when extraction fails."""
        return IPFeatures(
            ip_address=ip_address,
            is_private=False,
            is_reserved=False,
            country=None,
            region=None,
            city=None,
            latitude=None,
            longitude=None,
            timezone=None,
            asn=None,
            asn_organization=None,
            isp=None,
            reputation_score=None,
            abuse_score=None,
            threat_types=[],
            request_frequency=None,
            unique_user_agents=None,
            common_ports=[]
        )


class IPRotationDetector:
    """Detect IP rotation and evasion techniques."""
    
    def __init__(self):
        self.rotation_patterns = [
            # Add patterns for detecting IP rotation
        ]
    
    def detect_rotation(self, ip_history: List[str]) -> Dict[str, any]:
        """Detect IP rotation patterns."""
        results = {
            'rotation_detected': False,
            'rotation_frequency': 0.0,
            'unique_ips': len(set(ip_history)),
            'rotation_pattern': None
        }
        
        if len(ip_history) < 2:
            return results
        
        # Calculate rotation frequency
        unique_ips = set(ip_history)
        results['rotation_frequency'] = len(unique_ips) / len(ip_history)
        
        # Detect rotation if frequency is high
        if results['rotation_frequency'] > 0.5:
            results['rotation_detected'] = True
            results['rotation_pattern'] = 'high_frequency'
        
        return results


# Global feature extractor instance
ip_feature_extractor = IPFeatureExtractor()
ip_rotation_detector = IPRotationDetector()


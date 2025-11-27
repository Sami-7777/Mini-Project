"""
Data processing utilities for the cyberattack detection system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import json
import re
from urllib.parse import urlparse, parse_qs
import ipaddress

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database

logger = structlog.get_logger(__name__)


class DataType(str, Enum):
    """Data types for processing."""
    URL = "url"
    IP = "ip"
    EMAIL = "email"
    DOMAIN = "domain"
    USER_AGENT = "user_agent"
    LOG_ENTRY = "log_entry"


@dataclass
class ProcessedData:
    """Processed data structure."""
    data_type: DataType
    original_value: str
    processed_value: str
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime


class DataProcessor:
    """Processes and normalizes data for analysis."""
    
    def __init__(self):
        self.suspicious_patterns = {
            'phishing': [
                r'secure.*login',
                r'verify.*account',
                r'suspended.*account',
                r'urgent.*action',
                r'click.*here'
            ],
            'malware': [
                r'download.*update',
                r'install.*patch',
                r'virus.*scan',
                r'security.*fix'
            ],
            'spam': [
                r'free.*money',
                r'win.*prize',
                r'limited.*time',
                r'act.*now'
            ]
        }
        
        self.brand_keywords = [
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter',
            'linkedin', 'instagram', 'youtube', 'netflix', 'spotify', 'paypal'
        ]
    
    async def process_data(self, data: Union[str, Dict[str, Any]], 
                          data_type: DataType) -> ProcessedData:
        """Process data based on type."""
        try:
            if data_type == DataType.URL:
                return await self._process_url(data)
            elif data_type == DataType.IP:
                return await self._process_ip(data)
            elif data_type == DataType.EMAIL:
                return await self._process_email(data)
            elif data_type == DataType.DOMAIN:
                return await self._process_domain(data)
            elif data_type == DataType.USER_AGENT:
                return await self._process_user_agent(data)
            elif data_type == DataType.LOG_ENTRY:
                return await self._process_log_entry(data)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            logger.error("Error processing data", data_type=data_type, error=str(e))
            raise
    
    async def _process_url(self, url: str) -> ProcessedData:
        """Process URL data."""
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Extract features
            features = {
                'scheme': parsed.scheme,
                'netloc': parsed.netloc,
                'path': parsed.path,
                'query': parsed.query,
                'fragment': parsed.fragment,
                'domain': parsed.netloc.split(':')[0] if parsed.netloc else '',
                'port': parsed.port,
                'path_depth': len([p for p in parsed.path.split('/') if p]),
                'query_params': len(parse_qs(parsed.query)),
                'has_ip': self._is_ip_address(parsed.netloc),
                'is_shortener': self._is_url_shortener(parsed.netloc),
                'suspicious_keywords': self._extract_suspicious_keywords(url),
                'brand_keywords': self._extract_brand_keywords(url),
                'entropy': self._calculate_entropy(url),
                'length': len(url)
            }
            
            # Normalize URL
            normalized_url = self._normalize_url(url)
            
            return ProcessedData(
                data_type=DataType.URL,
                original_value=url,
                processed_value=normalized_url,
                features=features,
                metadata={'parsed': parsed._asdict()},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing URL", url=url, error=str(e))
            raise
    
    async def _process_ip(self, ip: str) -> ProcessedData:
        """Process IP address data."""
        try:
            # Parse IP
            ip_obj = ipaddress.ip_address(ip)
            
            # Extract features
            features = {
                'version': ip_obj.version,
                'is_private': ip_obj.is_private,
                'is_reserved': ip_obj.is_reserved,
                'is_multicast': ip_obj.is_multicast,
                'is_loopback': ip_obj.is_loopback,
                'is_link_local': ip_obj.is_link_local,
                'is_global': ip_obj.is_global,
                'is_unspecified': ip_obj.is_unspecified
            }
            
            # Add geolocation if available
            geo_features = await self._get_geo_features(ip)
            features.update(geo_features)
            
            return ProcessedData(
                data_type=DataType.IP,
                original_value=ip,
                processed_value=ip,
                features=features,
                metadata={'ip_object': str(ip_obj)},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing IP", ip=ip, error=str(e))
            raise
    
    async def _process_email(self, email: str) -> ProcessedData:
        """Process email address data."""
        try:
            # Parse email
            local, domain = email.split('@', 1)
            
            # Extract features
            features = {
                'local_part': local,
                'domain': domain,
                'local_length': len(local),
                'domain_length': len(domain),
                'total_length': len(email),
                'has_numbers': any(c.isdigit() for c in local),
                'has_special_chars': any(c in '._-+' for c in local),
                'suspicious_patterns': self._check_email_suspicious_patterns(email),
                'is_disposable': await self._is_disposable_email(domain)
            }
            
            # Normalize email
            normalized_email = f"{local.lower()}@{domain.lower()}"
            
            return ProcessedData(
                data_type=DataType.EMAIL,
                original_value=email,
                processed_value=normalized_email,
                features=features,
                metadata={'parsed': {'local': local, 'domain': domain}},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing email", email=email, error=str(e))
            raise
    
    async def _process_domain(self, domain: str) -> ProcessedData:
        """Process domain data."""
        try:
            # Extract features
            features = {
                'domain': domain,
                'length': len(domain),
                'subdomain_count': len(domain.split('.')) - 2,
                'tld': domain.split('.')[-1] if '.' in domain else '',
                'has_numbers': any(c.isdigit() for c in domain),
                'has_hyphens': '-' in domain,
                'suspicious_patterns': self._check_domain_suspicious_patterns(domain),
                'is_suspicious_tld': self._is_suspicious_tld(domain.split('.')[-1] if '.' in domain else '')
            }
            
            # Normalize domain
            normalized_domain = domain.lower()
            
            return ProcessedData(
                data_type=DataType.DOMAIN,
                original_value=domain,
                processed_value=normalized_domain,
                features=features,
                metadata={},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing domain", domain=domain, error=str(e))
            raise
    
    async def _process_user_agent(self, user_agent: str) -> ProcessedData:
        """Process user agent data."""
        try:
            # Extract features
            features = {
                'user_agent': user_agent,
                'length': len(user_agent),
                'browser': self._extract_browser(user_agent),
                'os': self._extract_os(user_agent),
                'device_type': self._extract_device_type(user_agent),
                'is_bot': self._is_bot_user_agent(user_agent),
                'suspicious_patterns': self._check_user_agent_suspicious_patterns(user_agent)
            }
            
            return ProcessedData(
                data_type=DataType.USER_AGENT,
                original_value=user_agent,
                processed_value=user_agent,
                features=features,
                metadata={},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing user agent", user_agent=user_agent, error=str(e))
            raise
    
    async def _process_log_entry(self, log_entry: Dict[str, Any]) -> ProcessedData:
        """Process log entry data."""
        try:
            # Extract features
            features = {
                'timestamp': log_entry.get('timestamp', ''),
                'level': log_entry.get('level', ''),
                'message': log_entry.get('message', ''),
                'source': log_entry.get('source', ''),
                'user_id': log_entry.get('user_id', ''),
                'ip_address': log_entry.get('ip_address', ''),
                'user_agent': log_entry.get('user_agent', ''),
                'request_method': log_entry.get('request_method', ''),
                'request_path': log_entry.get('request_path', ''),
                'status_code': log_entry.get('status_code', 0),
                'response_time': log_entry.get('response_time', 0),
                'suspicious_patterns': self._check_log_suspicious_patterns(log_entry)
            }
            
            # Normalize log entry
            normalized_entry = json.dumps(log_entry, sort_keys=True)
            
            return ProcessedData(
                data_type=DataType.LOG_ENTRY,
                original_value=json.dumps(log_entry),
                processed_value=normalized_entry,
                features=features,
                metadata={'original': log_entry},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Error processing log entry", error=str(e))
            raise
    
    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address."""
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False
    
    def _is_url_shortener(self, hostname: str) -> bool:
        """Check if hostname is a URL shortener."""
        shorteners = [
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
            'short.link', 'tiny.cc', 'buff.ly', 'adf.ly'
        ]
        return any(shortener in hostname.lower() for shortener in shorteners)
    
    def _extract_suspicious_keywords(self, text: str) -> List[str]:
        """Extract suspicious keywords from text."""
        found_keywords = []
        text_lower = text.lower()
        
        for category, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found_keywords.append(f"{category}:{pattern}")
        
        return found_keywords
    
    def _extract_brand_keywords(self, text: str) -> List[str]:
        """Extract brand keywords from text."""
        found_brands = []
        text_lower = text.lower()
        
        for brand in self.brand_keywords:
            if brand in text_lower:
                found_brands.append(brand)
        
        return found_brands
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate entropy of text."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        try:
            parsed = urlparse(url)
            
            # Normalize scheme
            scheme = parsed.scheme.lower()
            
            # Normalize hostname
            hostname = parsed.netloc.lower()
            if ':' in hostname:
                hostname = hostname.split(':')[0]
            
            # Normalize path
            path = parsed.path.lower()
            
            # Normalize query (sort parameters)
            query_params = parse_qs(parsed.query)
            sorted_params = sorted(query_params.items())
            query = '&'.join(f"{k}={v[0]}" for k, v in sorted_params)
            
            # Reconstruct URL
            normalized = f"{scheme}://{hostname}{path}"
            if query:
                normalized += f"?{query}"
            
            return normalized
            
        except Exception as e:
            logger.error("Error normalizing URL", error=str(e))
            return url
    
    async def _get_geo_features(self, ip: str) -> Dict[str, Any]:
        """Get geolocation features for IP."""
        try:
            # This would integrate with a geolocation service
            # For now, return placeholder data
            return {
                'country': 'Unknown',
                'region': 'Unknown',
                'city': 'Unknown',
                'latitude': 0.0,
                'longitude': 0.0,
                'timezone': 'Unknown'
            }
            
        except Exception as e:
            logger.error("Error getting geo features", error=str(e))
            return {}
    
    def _check_email_suspicious_patterns(self, email: str) -> List[str]:
        """Check for suspicious patterns in email."""
        suspicious = []
        
        # Check for suspicious patterns
        patterns = [
            r'\d{10,}',  # Long number sequences
            r'[a-z]{20,}',  # Long letter sequences
            r'[^a-zA-Z0-9@._-]',  # Special characters
        ]
        
        for pattern in patterns:
            if re.search(pattern, email):
                suspicious.append(pattern)
        
        return suspicious
    
    async def _is_disposable_email(self, domain: str) -> bool:
        """Check if domain is a disposable email service."""
        # This would check against a list of disposable email domains
        disposable_domains = [
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com'
        ]
        return domain.lower() in disposable_domains
    
    def _check_domain_suspicious_patterns(self, domain: str) -> List[str]:
        """Check for suspicious patterns in domain."""
        suspicious = []
        
        # Check for suspicious patterns
        patterns = [
            r'\d{4,}',  # Long number sequences
            r'[a-z]{15,}',  # Long letter sequences
            r'[^a-zA-Z0-9.-]',  # Special characters
        ]
        
        for pattern in patterns:
            if re.search(pattern, domain):
                suspicious.append(pattern)
        
        return suspicious
    
    def _is_suspicious_tld(self, tld: str) -> bool:
        """Check if TLD is suspicious."""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.top', '.click']
        return f'.{tld}' in suspicious_tlds
    
    def _extract_browser(self, user_agent: str) -> str:
        """Extract browser from user agent."""
        browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'Internet Explorer']
        
        for browser in browsers:
            if browser in user_agent:
                return browser
        
        return 'Unknown'
    
    def _extract_os(self, user_agent: str) -> str:
        """Extract OS from user agent."""
        os_patterns = {
            'Windows': r'Windows',
            'Mac OS': r'Mac OS',
            'Linux': r'Linux',
            'Android': r'Android',
            'iOS': r'iPhone|iPad'
        }
        
        for os_name, pattern in os_patterns.items():
            if re.search(pattern, user_agent):
                return os_name
        
        return 'Unknown'
    
    def _extract_device_type(self, user_agent: str) -> str:
        """Extract device type from user agent."""
        if 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent:
            return 'Mobile'
        elif 'Tablet' in user_agent or 'iPad' in user_agent:
            return 'Tablet'
        else:
            return 'Desktop'
    
    def _is_bot_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is a bot."""
        bot_patterns = [
            r'bot', r'crawler', r'spider', r'scraper', r'curl', r'wget',
            r'python', r'java', r'go-http-client'
        ]
        
        return any(re.search(pattern, user_agent, re.IGNORECASE) for pattern in bot_patterns)
    
    def _check_user_agent_suspicious_patterns(self, user_agent: str) -> List[str]:
        """Check for suspicious patterns in user agent."""
        suspicious = []
        
        # Check for suspicious patterns
        patterns = [
            r'[^a-zA-Z0-9\s().,;:/-]',  # Special characters
            r'\d{10,}',  # Long number sequences
        ]
        
        for pattern in patterns:
            if re.search(pattern, user_agent):
                suspicious.append(pattern)
        
        return suspicious
    
    def _check_log_suspicious_patterns(self, log_entry: Dict[str, Any]) -> List[str]:
        """Check for suspicious patterns in log entry."""
        suspicious = []
        
        # Check for suspicious patterns
        if log_entry.get('status_code', 0) >= 400:
            suspicious.append('error_status')
        
        if log_entry.get('response_time', 0) > 5000:  # 5 seconds
            suspicious.append('slow_response')
        
        if log_entry.get('request_path', '').count('/') > 10:
            suspicious.append('deep_path')
        
        return suspicious
    
    async def batch_process(self, data_list: List[Union[str, Dict[str, Any]]], 
                           data_type: DataType) -> List[ProcessedData]:
        """Process multiple data items in batch."""
        try:
            processed_data = []
            
            for data in data_list:
                try:
                    processed = await self.process_data(data, data_type)
                    processed_data.append(processed)
                except Exception as e:
                    logger.error("Error processing data item", error=str(e))
                    continue
            
            return processed_data
            
        except Exception as e:
            logger.error("Error in batch processing", error=str(e))
            return []
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            return {
                'supported_data_types': [dt.value for dt in DataType],
                'suspicious_patterns': len(self.suspicious_patterns),
                'brand_keywords': len(self.brand_keywords)
            }
            
        except Exception as e:
            logger.error("Error getting processing statistics", error=str(e))
            return {}


# Global data processor instance
data_processor = DataProcessor()

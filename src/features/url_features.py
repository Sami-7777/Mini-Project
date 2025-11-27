"""
URL feature extraction for cyberattack detection.
"""
import re
import math
import hashlib
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple
import tldextract
import whois
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import structlog

from ..database.models import URLFeatures

logger = structlog.get_logger(__name__)


class URLFeatureExtractor:
    """Extract comprehensive features from URLs for attack detection."""
    
    def __init__(self):
        self.suspicious_keywords = {
            'phishing': [
                'secure', 'verify', 'update', 'suspended', 'locked', 'expired',
                'urgent', 'immediate', 'action', 'required', 'click', 'login',
                'account', 'bank', 'paypal', 'amazon', 'microsoft', 'apple'
            ],
            'malware': [
                'download', 'install', 'update', 'patch', 'fix', 'repair',
                'virus', 'scan', 'clean', 'remove', 'security', 'protection'
            ],
            'suspicious_patterns': [
                'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
                'short.link', 'tiny.cc', 'buff.ly', 'adf.ly'
            ]
        }
        
        self.brand_keywords = [
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter',
            'linkedin', 'instagram', 'youtube', 'netflix', 'spotify', 'paypal',
            'ebay', 'walmart', 'target', 'bestbuy', 'homedepot', 'lowes'
        ]
    
    def extract_features(self, url: str) -> URLFeatures:
        """Extract all URL features."""
        try:
            parsed_url = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Basic lexical features
            lexical_features = self._extract_lexical_features(url, parsed_url)
            
            # Character analysis
            char_features = self._extract_character_features(url)
            
            # Domain analysis
            domain_features = self._extract_domain_features(extracted, parsed_url)
            
            # Keyword analysis
            keyword_features = self._extract_keyword_features(url)
            
            # URL structure analysis
            structure_features = self._extract_structure_features(parsed_url)
            
            # Semantic features
            semantic_features = self._extract_semantic_features(url)
            
            return URLFeatures(
                **lexical_features,
                **char_features,
                **domain_features,
                **keyword_features,
                **structure_features,
                **semantic_features
            )
            
        except Exception as e:
            logger.error("Error extracting URL features", url=url, error=str(e))
            # Return minimal features on error
            return self._get_minimal_features(url)
    
    def _extract_lexical_features(self, url: str, parsed_url) -> Dict:
        """Extract lexical features from URL."""
        return {
            'url_length': len(url),
            'domain_length': len(parsed_url.netloc),
            'path_length': len(parsed_url.path),
            'query_length': len(parsed_url.query),
            'fragment_length': len(parsed_url.fragment)
        }
    
    def _extract_character_features(self, url: str) -> Dict:
        """Extract character-based features."""
        digits = sum(1 for c in url if c.isdigit())
        letters = sum(1 for c in url if c.isalpha())
        special_chars = len(url) - digits - letters
        
        # Calculate entropy
        char_counts = {}
        for char in url.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0
        url_len = len(url)
        for count in char_counts.values():
            probability = count / url_len
            entropy -= probability * math.log2(probability)
        
        return {
            'digit_count': digits,
            'letter_count': letters,
            'special_char_count': special_chars,
            'entropy': round(entropy, 4)
        }
    
    def _extract_domain_features(self, extracted, parsed_url) -> Dict:
        """Extract domain-related features."""
        subdomain_count = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
        
        # Try to get domain age (with timeout)
        domain_age = None
        try:
            domain_info = whois.whois(extracted.domain + '.' + extracted.suffix)
            if domain_info.creation_date:
                if isinstance(domain_info.creation_date, list):
                    creation_date = domain_info.creation_date[0]
                else:
                    creation_date = domain_info.creation_date
                
                if isinstance(creation_date, datetime):
                    domain_age = (datetime.now() - creation_date).days
        except Exception as e:
            logger.debug("Could not get domain age", domain=extracted.domain, error=str(e))
        
        return {
            'subdomain_count': subdomain_count,
            'tld': extracted.suffix,
            'domain_age_days': domain_age
        }
    
    def _extract_keyword_features(self, url: str) -> Dict:
        """Extract keyword-based features."""
        url_lower = url.lower()
        
        suspicious_keywords = []
        for category, keywords in self.suspicious_keywords.items():
            for keyword in keywords:
                if keyword in url_lower:
                    suspicious_keywords.append(f"{category}:{keyword}")
        
        brand_keywords = []
        for brand in self.brand_keywords:
            if brand in url_lower:
                brand_keywords.append(brand)
        
        return {
            'suspicious_keywords': suspicious_keywords,
            'brand_keywords': brand_keywords
        }
    
    def _extract_structure_features(self, parsed_url) -> Dict:
        """Extract URL structure features."""
        # Check if domain is an IP address
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        has_ip_address = bool(re.match(ip_pattern, parsed_url.netloc))
        
        # Check for URL shorteners
        shortener_domains = {
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
            'short.link', 'tiny.cc', 'buff.ly', 'adf.ly', 'bitly.com'
        }
        has_shortener = any(shortener in parsed_url.netloc.lower() for shortener in shortener_domains)
        
        # Check for redirect patterns
        has_redirect = any(keyword in parsed_url.query.lower() for keyword in ['redirect', 'url', 'link'])
        
        # Extract port number
        port_number = None
        if ':' in parsed_url.netloc:
            try:
                port_number = int(parsed_url.netloc.split(':')[-1])
            except ValueError:
                pass
        
        return {
            'has_ip_address': has_ip_address,
            'has_shortener': has_shortener,
            'has_redirect': has_redirect,
            'port_number': port_number
        }
    
    def _extract_semantic_features(self, url: str) -> Dict:
        """Extract semantic features."""
        # URL similarity score (placeholder - would use more sophisticated methods)
        url_similarity_score = self._calculate_url_similarity(url)
        
        # Brand similarity score
        brand_similarity_score = self._calculate_brand_similarity(url)
        
        return {
            'url_similarity_score': url_similarity_score,
            'brand_similarity_score': brand_similarity_score
        }
    
    def _calculate_url_similarity(self, url: str) -> float:
        """Calculate URL similarity to known malicious patterns."""
        # This is a simplified implementation
        # In production, you'd use more sophisticated similarity measures
        
        malicious_patterns = [
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
            r'[a-z0-9]{8,}',  # Long random strings
            r'[^a-zA-Z0-9.-]{3,}',  # Multiple special characters
        ]
        
        similarity_score = 0.0
        for pattern in malicious_patterns:
            if re.search(pattern, url):
                similarity_score += 0.3
        
        return min(similarity_score, 1.0)
    
    def _calculate_brand_similarity(self, url: str) -> float:
        """Calculate similarity to known brand domains."""
        url_lower = url.lower()
        
        # Check for typosquatting patterns
        for brand in self.brand_keywords:
            if brand in url_lower:
                # Check for common typosquatting techniques
                if any(technique in url_lower for technique in [
                    brand + 's', brand + 'z', brand + 'x',
                    brand.replace('o', '0'), brand.replace('i', '1'),
                    brand.replace('e', '3'), brand.replace('a', '4')
                ]):
                    return 0.8  # High similarity but suspicious
        
        return 0.0
    
    def _get_minimal_features(self, url: str) -> URLFeatures:
        """Return minimal features when extraction fails."""
        return URLFeatures(
            url_length=len(url),
            domain_length=0,
            path_length=0,
            query_length=0,
            fragment_length=0,
            digit_count=0,
            letter_count=0,
            special_char_count=0,
            entropy=0.0,
            subdomain_count=0,
            tld="",
            suspicious_keywords=[],
            brand_keywords=[],
            has_ip_address=False,
            has_shortener=False,
            has_redirect=False,
            port_number=None,
            url_similarity_score=0.0,
            brand_similarity_score=0.0
        )


class URLPolymorphismDetector:
    """Detect URL polymorphism and obfuscation techniques."""
    
    def __init__(self):
        self.obfuscation_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#[0-9]+;',  # HTML entities
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'\\u[0-9a-fA-F]{4}',  # Unicode encoding
        ]
    
    def detect_polymorphism(self, url: str) -> Dict[str, any]:
        """Detect various polymorphism techniques."""
        results = {
            'has_encoding': False,
            'encoding_types': [],
            'suspicious_encoding_ratio': 0.0,
            'homograph_attack': False,
            'punycode_usage': False
        }
        
        # Check for various encodings
        for pattern in self.obfuscation_patterns:
            if re.search(pattern, url):
                results['has_encoding'] = True
                results['encoding_types'].append(pattern)
        
        # Calculate encoding ratio
        encoded_chars = sum(len(re.findall(pattern, url)) for pattern in self.obfuscation_patterns)
        results['suspicious_encoding_ratio'] = encoded_chars / len(url) if url else 0
        
        # Check for homograph attacks (punycode)
        if 'xn--' in url:
            results['punycode_usage'] = True
            results['homograph_attack'] = True
        
        return results


# Global feature extractor instance
url_feature_extractor = URLFeatureExtractor()
polymorphism_detector = URLPolymorphismDetector()


"""
Encryption and security utilities for the cyberattack detection system.
"""
import os
import hashlib
import hmac
import base64
from typing import Optional, Dict, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class EncryptionManager:
    """Manages encryption and decryption of sensitive data."""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key."""
        try:
            # Try to get key from environment
            key_str = settings.encryption_key
            
            if key_str and len(key_str) >= 32:
                # Use existing key
                return base64.urlsafe_b64encode(key_str.encode()[:32].ljust(32, b'0'))
            else:
                # Generate new key
                key = Fernet.generate_key()
                logger.warning("Generated new encryption key. Store this securely!")
                logger.info("Encryption key", key=base64.urlsafe_b64encode(key).decode())
                return key
                
        except Exception as e:
            logger.error("Error getting encryption key", error=str(e))
            # Generate fallback key
            return Fernet.generate_key()
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error("Error encrypting data", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error("Error decrypting data", error=str(e))
            raise
    
    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Encrypt dictionary values."""
        try:
            encrypted_dict = {}
            
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    encrypted_value = self.encrypt(str(value))
                    encrypted_dict[key] = encrypted_value
                else:
                    # For complex types, convert to JSON string first
                    import json
                    json_str = json.dumps(value)
                    encrypted_value = self.encrypt(json_str)
                    encrypted_dict[key] = encrypted_value
            
            return encrypted_dict
            
        except Exception as e:
            logger.error("Error encrypting dictionary", error=str(e))
            raise
    
    def decrypt_dict(self, encrypted_data: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt dictionary values."""
        try:
            decrypted_dict = {}
            
            for key, encrypted_value in encrypted_data.items():
                decrypted_value = self.decrypt(encrypted_value)
                decrypted_dict[key] = decrypted_value
            
            return decrypted_dict
            
        except Exception as e:
            logger.error("Error decrypting dictionary", error=str(e))
            raise


class HashManager:
    """Manages hashing operations for data integrity and privacy."""
    
    @staticmethod
    def sha256_hash(data: Union[str, bytes]) -> str:
        """Generate SHA-256 hash."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            logger.error("Error generating SHA-256 hash", error=str(e))
            raise
    
    @staticmethod
    def md5_hash(data: Union[str, bytes]) -> str:
        """Generate MD5 hash."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            return hashlib.md5(data).hexdigest()
            
        except Exception as e:
            logger.error("Error generating MD5 hash", error=str(e))
            raise
    
    @staticmethod
    def hmac_hash(data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """Generate HMAC hash."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            if isinstance(key, str):
                key = key.encode('utf-8')
            
            return hmac.new(key, data, hashlib.sha256).hexdigest()
            
        except Exception as e:
            logger.error("Error generating HMAC hash", error=str(e))
            raise
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data with optional salt."""
        try:
            if salt is None:
                salt = os.urandom(16).hex()
            
            # Combine data with salt
            combined = f"{data}:{salt}"
            
            # Generate hash
            hash_value = HashManager.sha256_hash(combined)
            
            # Return hash with salt
            return f"{hash_value}:{salt}"
            
        except Exception as e:
            logger.error("Error hashing sensitive data", error=str(e))
            raise
    
    @staticmethod
    def verify_hash(data: str, hash_with_salt: str) -> bool:
        """Verify hash against data."""
        try:
            hash_value, salt = hash_with_salt.split(':')
            
            # Recreate hash
            combined = f"{data}:{salt}"
            computed_hash = HashManager.sha256_hash(combined)
            
            return hmac.compare_digest(hash_value, computed_hash)
            
        except Exception as e:
            logger.error("Error verifying hash", error=str(e))
            return False


class DataAnonymizer:
    """Anonymizes sensitive data for privacy protection."""
    
    @staticmethod
    def anonymize_ip(ip_address: str, mask_bits: int = 24) -> str:
        """Anonymize IP address by masking bits."""
        try:
            import ipaddress
            
            ip = ipaddress.ip_address(ip_address)
            
            if ip.version == 4:
                # IPv4
                mask = ipaddress.IPv4Network(f"0.0.0.0/{mask_bits}", strict=False)
                masked_ip = ipaddress.IPv4Address(int(ip) & int(mask.network_address))
                return str(masked_ip)
            else:
                # IPv6
                mask = ipaddress.IPv6Network(f"::/{mask_bits}", strict=False)
                masked_ip = ipaddress.IPv6Address(int(ip) & int(mask.network_address))
                return str(masked_ip)
                
        except Exception as e:
            logger.error("Error anonymizing IP address", error=str(e))
            return "0.0.0.0"
    
    @staticmethod
    def anonymize_email(email: str) -> str:
        """Anonymize email address."""
        try:
            if '@' not in email:
                return email
            
            local, domain = email.split('@', 1)
            
            # Keep first and last character of local part
            if len(local) > 2:
                anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
            else:
                anonymized_local = '*' * len(local)
            
            return f"{anonymized_local}@{domain}"
            
        except Exception as e:
            logger.error("Error anonymizing email", error=str(e))
            return "***@***.***"
    
    @staticmethod
    def anonymize_url(url: str) -> str:
        """Anonymize URL while preserving structure."""
        try:
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            
            parsed = urlparse(url)
            
            # Anonymize domain
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) >= 2:
                # Keep TLD, anonymize domain
                anonymized_domain = '*.' + '.'.join(domain_parts[-2:])
            else:
                anonymized_domain = parsed.netloc
            
            # Anonymize query parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                anonymized_params = {}
                
                for key, values in query_params.items():
                    if key.lower() in ['email', 'user', 'id', 'token']:
                        anonymized_params[key] = ['***']
                    else:
                        anonymized_params[key] = values
                
                anonymized_query = urlencode(anonymized_params, doseq=True)
            else:
                anonymized_query = parsed.query
            
            # Reconstruct URL
            anonymized_url = urlunparse((
                parsed.scheme,
                anonymized_domain,
                parsed.path,
                parsed.params,
                anonymized_query,
                parsed.fragment
            ))
            
            return anonymized_url
            
        except Exception as e:
            logger.error("Error anonymizing URL", error=str(e))
            return "***://***.***"
    
    @staticmethod
    def anonymize_user_agent(user_agent: str) -> str:
        """Anonymize user agent string."""
        try:
            # Keep browser and OS info, remove version details
            parts = user_agent.split()
            
            anonymized_parts = []
            for part in parts:
                if '/' in part:
                    # Version info
                    name, version = part.split('/', 1)
                    anonymized_parts.append(f"{name}/***")
                else:
                    anonymized_parts.append(part)
            
            return ' '.join(anonymized_parts)
            
        except Exception as e:
            logger.error("Error anonymizing user agent", error=str(e))
            return "***"
    
    @staticmethod
    def anonymize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data in dictionary."""
        try:
            anonymized_data = {}
            
            for key, value in data.items():
                if isinstance(value, str):
                    key_lower = key.lower()
                    
                    if 'ip' in key_lower:
                        anonymized_data[key] = DataAnonymizer.anonymize_ip(value)
                    elif 'email' in key_lower:
                        anonymized_data[key] = DataAnonymizer.anonymize_email(value)
                    elif 'url' in key_lower:
                        anonymized_data[key] = DataAnonymizer.anonymize_url(value)
                    elif 'user_agent' in key_lower:
                        anonymized_data[key] = DataAnonymizer.anonymize_user_agent(value)
                    else:
                        anonymized_data[key] = value
                else:
                    anonymized_data[key] = value
            
            return anonymized_data
            
        except Exception as e:
            logger.error("Error anonymizing data", error=str(e))
            return data


class SecurityValidator:
    """Validates security-related data and configurations."""
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format and strength."""
        try:
            # Check minimum length
            if len(api_key) < 32:
                return False
            
            # Check for required prefix
            if not api_key.startswith('sk-'):
                return False
            
            # Check for sufficient entropy
            import string
            charset = string.ascii_letters + string.digits + '-_'
            
            if not all(c in charset for c in api_key[3:]):
                return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating API key", error=str(e))
            return False
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        try:
            result = {
                'is_strong': False,
                'score': 0,
                'issues': []
            }
            
            # Check length
            if len(password) < 8:
                result['issues'].append('Password too short (minimum 8 characters)')
            else:
                result['score'] += 1
            
            # Check for uppercase
            if not any(c.isupper() for c in password):
                result['issues'].append('Missing uppercase letters')
            else:
                result['score'] += 1
            
            # Check for lowercase
            if not any(c.islower() for c in password):
                result['issues'].append('Missing lowercase letters')
            else:
                result['score'] += 1
            
            # Check for digits
            if not any(c.isdigit() for c in password):
                result['issues'].append('Missing digits')
            else:
                result['score'] += 1
            
            # Check for special characters
            special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
            if not any(c in special_chars for c in password):
                result['issues'].append('Missing special characters')
            else:
                result['score'] += 1
            
            # Determine if strong
            result['is_strong'] = result['score'] >= 4 and len(result['issues']) == 0
            
            return result
            
        except Exception as e:
            logger.error("Error validating password strength", error=str(e))
            return {'is_strong': False, 'score': 0, 'issues': ['Validation error']}
    
    @staticmethod
    def validate_url_safety(url: str) -> Dict[str, Any]:
        """Validate URL safety."""
        try:
            result = {
                'is_safe': True,
                'issues': [],
                'risk_score': 0.0
            }
            
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                result['issues'].append('Invalid or unsafe scheme')
                result['risk_score'] += 0.3
            
            # Check for IP address in hostname
            try:
                import ipaddress
                ipaddress.ip_address(parsed.hostname)
                result['issues'].append('IP address in hostname')
                result['risk_score'] += 0.2
            except ValueError:
                pass  # Not an IP address
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
            if any(tld in parsed.hostname for tld in suspicious_tlds):
                result['issues'].append('Suspicious TLD')
                result['risk_score'] += 0.1
            
            # Check for URL shorteners
            shortener_domains = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl']
            if any(shortener in parsed.hostname for shortener in shortener_domains):
                result['issues'].append('URL shortener detected')
                result['risk_score'] += 0.1
            
            # Determine if safe
            result['is_safe'] = result['risk_score'] < 0.5
            
            return result
            
        except Exception as e:
            logger.error("Error validating URL safety", error=str(e))
            return {'is_safe': False, 'issues': ['Validation error'], 'risk_score': 1.0}


# Global instances
encryption_manager = EncryptionManager()
hash_manager = HashManager()
data_anonymizer = DataAnonymizer()
security_validator = SecurityValidator()

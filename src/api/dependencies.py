"""
API dependencies for authentication and authorization.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer()


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get and validate API key from Authorization header."""
    try:
        api_key = credentials.credentials
        
        # Validate API key format
        if not api_key or len(api_key) < 10:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # In production, you would validate against a database
        # For now, we'll use a simple validation
        if not await validate_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating API key", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(api_key: str = Depends(get_api_key)) -> Dict[str, str]:
    """Get current user information from API key."""
    try:
        # In production, you would look up user info from database
        user_info = await get_user_from_api_key(api_key)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting current user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_admin_user(current_user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:
    """Get current user and verify admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def validate_api_key(api_key: str) -> bool:
    """Validate API key against database or configuration."""
    try:
        # Placeholder implementation
        # In production, you would validate against a database
        
        # For demo purposes, accept any key that starts with 'sk-' and has length > 10
        if api_key.startswith("sk-") and len(api_key) > 10:
            return True
        
        # You could also check against a list of valid keys in settings
        valid_keys = getattr(settings, 'valid_api_keys', [])
        if api_key in valid_keys:
            return True
        
        return False
        
    except Exception as e:
        logger.error("Error validating API key", error=str(e))
        return False


async def get_user_from_api_key(api_key: str) -> Optional[Dict[str, str]]:
    """Get user information from API key."""
    try:
        # Placeholder implementation
        # In production, you would look up user info from database
        
        # For demo purposes, return a default user
        if api_key.startswith("sk-"):
            return {
                "username": "api_user",
                "role": "user",
                "api_key": api_key,
                "permissions": ["read", "write"]
            }
        
        return None
        
    except Exception as e:
        logger.error("Error getting user from API key", error=str(e))
        return None


async def check_permission(permission: str, current_user: Dict[str, str] = Depends(get_current_user)) -> bool:
    """Check if user has specific permission."""
    try:
        user_permissions = current_user.get("permissions", [])
        
        if permission in user_permissions:
            return True
        
        # Admin users have all permissions
        if current_user.get("role") == "admin":
            return True
        
        return False
        
    except Exception as e:
        logger.error("Error checking permission", error=str(e))
        return False


def require_permission(permission: str):
    """Dependency factory for requiring specific permission."""
    async def permission_checker(current_user: Dict[str, str] = Depends(get_current_user)):
        if not await check_permission(permission, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker


async def get_rate_limit_info(current_user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, int]:
    """Get rate limit information for current user."""
    try:
        # In production, you would get rate limits from user profile or database
        user_role = current_user.get("role", "user")
        
        # Define rate limits by role
        rate_limits = {
            "admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
            "user": {"requests_per_minute": 100, "requests_per_hour": 1000},
            "guest": {"requests_per_minute": 10, "requests_per_hour": 100}
        }
        
        return rate_limits.get(user_role, rate_limits["guest"])
        
    except Exception as e:
        logger.error("Error getting rate limit info", error=str(e))
        return {"requests_per_minute": 10, "requests_per_hour": 100}

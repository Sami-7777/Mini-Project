"""
Custom middleware for the FastAPI application.
"""
import time
import json
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import structlog
from datetime import datetime, timedelta
import redis.asyncio as redis
from collections import defaultdict

from ..core.config import settings
from ..core.logger import logger

# Configure structured logging
logger = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        self.app = app
        self.calls = calls
        self.period = period
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client for rate limiting."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.warning("Could not initialize Redis for rate limiting", error=str(e))
            self.redis_client = None
    
    async def __call__(self, scope, receive, send):
        """Process request with rate limiting."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            await self.app(scope, receive, send)
            return
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if await self._is_rate_limited(client_ip):
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.calls} requests per {self.period} seconds",
                    "retry_after": self.period
                }
            )
            await response(scope, receive, send)
            return
        
        # Process request
        await self.app(scope, receive, send)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        if not self.redis_client:
            return False
        
        try:
            key = f"rate_limit:{client_ip}"
            current_time = int(time.time())
            window_start = current_time - self.period
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, self.period)
            
            results = await pipe.execute()
            current_count = results[1]
            
            return current_count >= self.calls
            
        except Exception as e:
            logger.warning("Error checking rate limit", client_ip=client_ip, error=str(e))
            return False


class SecurityMiddleware:
    """Security middleware for adding security headers."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request with security headers."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                
                # Add security headers
                security_headers = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
                    (b"content-security-policy", b"default-src 'self'"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"geolocation=(), microphone=(), camera=()")
                ]
                
                # Add headers if they don't already exist
                existing_headers = [h[0].lower() for h in headers]
                for header_name, header_value in security_headers:
                    if header_name.lower() not in existing_headers:
                        headers.append((header_name, header_value))
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class LoggingMiddleware:
    """Logging middleware for request/response logging."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request with logging."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Log request
        logger.info("Request started",
                   method=request.method,
                   path=request.url.path,
                   query_params=str(request.query_params),
                   client_ip=request.client.host if request.client else "unknown",
                   user_agent=request.headers.get("user-agent", "unknown"))
        
        # Process request
        response = None
        
        async def send_wrapper(message):
            nonlocal response
            if message["type"] == "http.response.start":
                response = message
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
        
        # Log response
        if response:
            duration = time.time() - start_time
            logger.info("Request completed",
                       method=request.method,
                       path=request.url.path,
                       status_code=response["status"],
                       duration_ms=round(duration * 1000, 2),
                       client_ip=request.client.host if request.client else "unknown")


class MetricsMiddleware:
    """Metrics middleware for collecting request metrics."""
    
    def __init__(self, app):
        self.app = app
        self.metrics = defaultdict(int)
        self.response_times = []
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client for metrics."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            logger.warning("Could not initialize Redis for metrics", error=str(e))
            self.redis_client = None
    
    async def __call__(self, scope, receive, send):
        """Process request with metrics collection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Process request
        response = None
        
        async def send_wrapper(message):
            nonlocal response
            if message["type"] == "http.response.start":
                response = message
            await send(message)
        
        await self.app(scope, receive, send)
        
        # Collect metrics
        if response:
            duration = time.time() - start_time
            await self._collect_metrics(request, response, duration)
    
    async def _collect_metrics(self, request: Request, response: dict, duration: float):
        """Collect request metrics."""
        try:
            # Update in-memory metrics
            self.metrics["total_requests"] += 1
            self.metrics[f"status_{response['status']}"] += 1
            self.metrics[f"method_{request.method}"] += 1
            
            # Store response time
            self.response_times.append(duration)
            if len(self.response_times) > 1000:  # Keep only last 1000 requests
                self.response_times = self.response_times[-1000:]
            
            # Store in Redis if available
            if self.redis_client:
                current_minute = int(time.time() // 60)
                key = f"metrics:{current_minute}"
                
                pipe = self.redis_client.pipeline()
                pipe.hincrby(key, "total_requests", 1)
                pipe.hincrby(key, f"status_{response['status']}", 1)
                pipe.hincrby(key, f"method_{request.method}", 1)
                pipe.expire(key, 3600)  # Expire after 1 hour
                
                await pipe.execute()
        
        except Exception as e:
            logger.warning("Error collecting metrics", error=str(e))
    
    async def get_metrics(self) -> dict:
        """Get collected metrics."""
        try:
            # Calculate response time statistics
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                max_response_time = max(self.response_times)
                min_response_time = min(self.response_times)
            else:
                avg_response_time = max_response_time = min_response_time = 0
            
            return {
                "metrics": dict(self.metrics),
                "response_times": {
                    "average": round(avg_response_time * 1000, 2),  # Convert to ms
                    "max": round(max_response_time * 1000, 2),
                    "min": round(min_response_time * 1000, 2),
                    "count": len(self.response_times)
                }
            }
        
        except Exception as e:
            logger.error("Error getting metrics", error=str(e))
            return {}


class AuthenticationMiddleware:
    """Authentication middleware for API key validation."""
    
    def __init__(self, app, excluded_paths: list = None):
        self.app = app
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json",
            "/health", "/health/live", "/health/ready"
        ]
    
    async def __call__(self, scope, receive, send):
        """Process request with authentication."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            await self.app(scope, receive, send)
            return
        
        # Check for API key
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        
        if not api_key:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "message": "API key is required"
                }
            )
            await response(scope, receive, send)
            return
        
        # Validate API key (placeholder - implement actual validation)
        if not await self._validate_api_key(api_key):
            response = JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid"
                }
            )
            await response(scope, receive, send)
            return
        
        # Add user info to request scope
        scope["user"] = await self._get_user_from_api_key(api_key)
        
        await self.app(scope, receive, send)
    
    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        # Placeholder implementation
        # In production, you would validate against a database
        return api_key.startswith("sk-") and len(api_key) > 10
    
    async def _get_user_from_api_key(self, api_key: str) -> dict:
        """Get user information from API key."""
        # Placeholder implementation
        # In production, you would look up user info from database
        return {
            "username": "api_user",
            "role": "user",
            "api_key": api_key
        }

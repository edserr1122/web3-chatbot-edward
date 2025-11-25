"""
Cache management for API responses and analysis results.
Uses Redis for distributed caching with TTL support.
"""

import json
import logging
from typing import Any, Optional
from datetime import timedelta
import redis
from src.utils.config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of API responses and analysis results."""
    
    def __init__(self):
        """Initialize cache manager with Redis or in-memory fallback."""
        self.enabled = config.REDIS_ENABLED
        self.ttl = config.CACHE_TTL_SECONDS
        self.redis_client = None
        self._memory_cache = {}  # Fallback in-memory cache
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(**config.get_redis_connection_kwargs())
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"⚠️  Redis connection failed: {e}. Falling back to in-memory cache.")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            if self.redis_client:
                # Redis cache
                value = self.redis_client.get(key)
                if value:
                    logger.info(f"♻️  Redis cache HIT: {key}")
                    return json.loads(value)
                else:
                    logger.info(f"💨 Redis cache MISS: {key}")
                    return None
            else:
                # In-memory cache
                cached = self._memory_cache.get(key)
                if cached is not None:
                    logger.info(f"♻️  In-memory cache HIT: {key}")
                else:
                    logger.info(f"💨 In-memory cache MISS: {key}")
                return cached
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: from config)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        ttl = ttl or self.ttl
        
        try:
            if self.redis_client:
                # Redis cache
                serialized = json.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
                logger.info(f"🗂️  Redis cache SET: {key} (TTL: {ttl}s)")
                return True
            else:
                # In-memory cache (no TTL for simplicity)
                self._memory_cache[key] = value
                logger.info(f"🗂️  In-memory cache SET: {key} (no TTL)")
                return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            if self.redis_client:
                self.redis_client.delete(key)
                logger.info(f"🗑️  Redis cache DELETE: {key}")
            else:
                self._memory_cache.pop(key, None)
                logger.info(f"🗑️  In-memory cache DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                self.redis_client.flushdb()
            else:
                self._memory_cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """
        Get value from cache or compute and cache it.
        
        Args:
            key: Cache key
            factory_func: Function to call if cache miss
            ttl: Time to live in seconds
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute value
        value = factory_func()
        
        # Cache it
        self.set(key, value, ttl)
        
        return value
    
    def make_key(self, *parts: str) -> str:
        """
        Create a cache key from parts.
        
        Args:
            *parts: Key components
            
        Returns:
            Cache key string
        """
        return ":".join(str(p) for p in parts)


# Singleton instance
cache_manager = CacheManager()


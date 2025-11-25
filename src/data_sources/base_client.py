"""
Base client class for all API data sources.
Provides common functionality for API requests, error handling, and rate limiting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """Abstract base class for all API clients."""
    
    # Class-level circuit breaker state (shared across all instances of same class)
    _circuit_breaker_until = {}  # {class_name: timestamp}
    _circuit_breaker_cooldown = 60  # seconds to wait after 429
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        """
        Initialize the base API client.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum seconds between requests
        
    def _rate_limit(self):
        """Implement basic rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_circuit_breaker_open(self) -> bool:
        """
        Check if circuit breaker is open (API is temporarily disabled due to rate limit).
        
        Returns:
            bool: True if circuit breaker is open (should not make requests)
        """
        client_class = self.__class__.__name__
        cooldown_until = self._circuit_breaker_until.get(client_class, 0)
        current_time = time.time()
        
        if current_time < cooldown_until:
            remaining = int(cooldown_until - current_time)
            return True
        
        # Circuit breaker has cooled down, reset it
        if client_class in self._circuit_breaker_until:
            del self._circuit_breaker_until[client_class]
            logger.info(f"🔓 [{client_class}] Circuit breaker reset - API calls resuming")
        
        return False
    
    def _open_circuit_breaker(self, cooldown_seconds: Optional[int] = None):
        """
        Open circuit breaker (temporarily disable API due to rate limit).
        
        Args:
            cooldown_seconds: How long to wait before retry (default: 60s)
        """
        client_class = self.__class__.__name__
        cooldown = cooldown_seconds if cooldown_seconds else self._circuit_breaker_cooldown
        cooldown_until = time.time() + cooldown
        self._circuit_breaker_until[client_class] = cooldown_until
        
        logger.warning(f"🔒 [{client_class}] Circuit breaker OPEN - Rate limit hit! Cooling down for {cooldown}s")
        logger.info(f"💡 [{client_class}] Other data sources will be used during cooldown")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        Can be overridden by subclasses for custom authentication.
        
        Returns:
            dict: Headers dictionary
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "Web3-Crypto-Chatbot/1.0"
        }
        return headers
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Make an API request with error handling and circuit breaker pattern.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            timeout: Request timeout in seconds
            
        Returns:
            dict: API response data
            
        Raises:
            Exception: If request fails or circuit breaker is open
        """
        api_name = self.__class__.__name__.replace("Client", "")
        
        # Check if circuit breaker is open (rate limited)
        if self._is_circuit_breaker_open():
            cooldown_until = self._circuit_breaker_until.get(self.__class__.__name__, 0)
            remaining = int(cooldown_until - time.time())
            logger.warning(f"⏸️  [{api_name}] Skipping request - Circuit breaker open ({remaining}s cooldown remaining)")
            raise Exception(f"{api_name} temporarily unavailable due to rate limiting (retry in {remaining}s)")
        
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            # Log API call with details
            logger.info(f"📡 [{api_name}] Calling API: {method} {url}")
            if params:
                # Mask sensitive parameters in logs
                masked_params = {
                    k: '***' if any(sensitive in k.lower() for sensitive in ['api_key', 'auth_token', 'key', 'token', 'secret']) 
                    else v 
                    for k, v in params.items()
                }
                logger.info(f"   Parameters: {masked_params}")
            
            if method.upper() == "GET":
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=timeout
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url, 
                    json=params, 
                    headers=headers, 
                    timeout=timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            result = response.json()
            
            # Log successful response
            if isinstance(result, dict):
                if "data" in result:
                    data = result["data"]
                    if isinstance(data, list):
                        logger.info(f"✅ [{api_name}] Success - Retrieved {len(data)} items")
                    else:
                        logger.info(f"✅ [{api_name}] Success - Retrieved data object")
                elif "results" in result:
                    results = result["results"]
                    if isinstance(results, list):
                        logger.info(f"✅ [{api_name}] Success - Retrieved {len(results)} items")
                    else:
                        logger.info(f"✅ [{api_name}] Success - Retrieved results object")
                else:
                    logger.info(f"✅ [{api_name}] Success - Retrieved response")
            else:
                logger.info(f"✅ [{api_name}] Success")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ [{api_name}] Request timeout for {url}")
            raise Exception(f"Request timeout: {api_name}")
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            # Handle 429 Rate Limit - Open circuit breaker
            if status_code == 429:
                logger.error(f"❌ [{api_name}] HTTP error 429 for {url}: Rate limit exceeded")
                self._open_circuit_breaker(cooldown_seconds=120)  # 2 min cooldown for rate limits
                raise Exception(f"{api_name} rate limit exceeded - using fallback data sources")
            
            # Handle 451 Geo-Restriction (Binance, etc.)
            elif status_code == 451:
                logger.warning(f"⚠️  [{api_name}] HTTP error 451 for {url}: Geo-restricted (service unavailable in your region)")
                # Don't open circuit breaker - this is permanent for this region
                raise Exception(f"{api_name} unavailable due to geo-restrictions - using fallback data sources")
            
            # Handle 5xx Server Errors (502, 503, 504) - Temporary server issues
            elif status_code in [502, 503, 504]:
                error_names = {502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}
                error_name = error_names.get(status_code, "Server Error")
                logger.error(f"❌ [{api_name}] HTTP error {status_code} ({error_name}) for {url}: Server temporarily unavailable")
                self._open_circuit_breaker(cooldown_seconds=30)  # 30 sec cooldown for server errors
                raise Exception(f"{api_name} temporarily unavailable ({error_name}) - using fallback data sources")
            
            # Handle other HTTP errors (4xx, etc.)
            logger.error(f"❌ [{api_name}] HTTP error {status_code} for {url}: {e}")
            
            # Don't log full HTML error pages in the exception
            error_text = e.response.text
            if len(error_text) > 500 or '<!DOCTYPE html>' in error_text:
                error_text = f"{error_text[:200]}... (truncated HTML response)"
            
            raise Exception(f"API error: {status_code} - {error_text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ [{api_name}] Request error for {url}: {e}")
            raise Exception(f"Request failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ [{api_name}] Unexpected error for {url}: {e}")
            raise
    
    @abstractmethod
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get token data. Must be implemented by subclasses.
        
        Args:
            symbol: Token symbol (e.g., "BTC", "ETH")
            
        Returns:
            dict: Token data
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test API connection. Must be implemented by subclasses.
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
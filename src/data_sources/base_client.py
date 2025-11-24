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
        Make an API request with error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            timeout: Request timeout in seconds
            
        Returns:
            dict: API response data
            
        Raises:
            Exception: If request fails
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
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
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise Exception(f"Request timeout: {self.__class__.__name__}")
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise Exception(f"API error: {e.response.status_code} - {e.response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            raise Exception(f"Request failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
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
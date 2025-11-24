"""
Alternative.me Fear & Greed Index client.
Free API, no authentication required.
Documentation: https://alternative.me/crypto/fear-and-greed-index/
"""

from typing import Dict, Any, List
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class FearGreedClient(BaseAPIClient):
    """Client for Alternative.me Fear & Greed Index API."""
    
    def __init__(self, base_url: str = "https://api.alternative.me/fng/"):
        """
        Initialize Fear & Greed client.
        
        Args:
            base_url: API base URL
        """
        super().__init__(api_key=None, base_url=base_url)
        self.min_request_interval = 0.5
        
    def get_token_data(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get current Fear & Greed Index.
        Note: This is a market-wide index, not token-specific.
        
        Args:
            symbol: Token symbol (ignored, kept for interface compatibility)
            
        Returns:
            dict: Fear & Greed Index data
        """
        return self.get_current()
    
    def get_current(self) -> Dict[str, Any]:
        """
        Get current Fear & Greed Index value.
        
        Returns:
            dict: Current index data
        """
        params = {"limit": 1}
        response = self._make_request("", params)
        
        data = response.get("data", [{}])[0]
        
        value = int(data.get("value", 0))
        classification = data.get("value_classification", "")
        
        return {
            "value": value,
            "classification": classification,
            "timestamp": data.get("timestamp"),
            "time_until_update": data.get("time_until_update"),
            "interpretation": self._interpret_value(value),
        }
    
    def get_historical(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical Fear & Greed Index values.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            list: Historical index data
        """
        params = {"limit": days}
        response = self._make_request("", params)
        
        return response.get("data", [])
    
    def _interpret_value(self, value: int) -> str:
        """
        Interpret Fear & Greed Index value.
        
        Args:
            value: Index value (0-100)
            
        Returns:
            str: Interpretation text
        """
        if value <= 25:
            return "Extreme Fear - Market is very fearful, potential buying opportunity"
        elif value <= 45:
            return "Fear - Market sentiment is fearful"
        elif value <= 55:
            return "Neutral - Market sentiment is balanced"
        elif value <= 75:
            return "Greed - Market sentiment is greedy"
        else:
            return "Extreme Greed - Market is very greedy, potential correction risk"
    
    def test_connection(self) -> bool:
        """
        Test connection to Fear & Greed API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.get_current()
            return True
        except Exception as e:
            logger.error(f"Fear & Greed connection test failed: {e}")
            return False
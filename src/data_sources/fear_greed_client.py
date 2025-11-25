"""
Alternative.me API client (Fear & Greed Index + Crypto Data).
Free API, no authentication required.
Documentation: 
- Fear & Greed: https://alternative.me/crypto/fear-and-greed-index/
- Crypto API v2: https://api.alternative.me/
"""

from typing import Dict, Any, List, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class FearGreedClient(BaseAPIClient):
    """
    Client for Alternative.me API.
    
    Provides:
    - Fear & Greed Index (market sentiment)
    - Global crypto market data
    - Basic crypto ticker data (as fallback)
    """
    
    def __init__(self):
        """
        Initialize Alternative.me client.
        
        Note: No API key required - completely free!
        """
        # Use v2 API base, switch to /fng/ for Fear & Greed
        super().__init__(api_key=None, base_url="https://api.alternative.me/")
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
        response = self._make_request("fng/", params)
        
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
        response = self._make_request("fng/", params)
        
        return response.get("data", [])
    
    def get_global_data(self, convert: str = "USD") -> Dict[str, Any]:
        """
        Get global cryptocurrency market data.
        
        ⚡ USE THIS for global market overview - simpler than CoinGecko/CMC!
        
        Args:
            convert: Currency conversion (USD, EUR, GBP, etc.)
            
        Returns:
            dict: Global market statistics
            
        Reference: https://api.alternative.me/v2/global/
        """
        params = {}
        if convert != "USD":
            params["convert"] = convert
        
        response = self._make_request("v2/global/", params)
        data = response.get("data", {})
        quotes = data.get("quotes", {}).get(convert, {})
        
        return {
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "active_markets": data.get("active_markets"),
            "bitcoin_dominance": data.get("bitcoin_percentage_of_market_cap"),
            "total_market_cap": quotes.get("total_market_cap"),
            "total_volume_24h": quotes.get("total_volume_24h"),
            "last_updated": data.get("last_updated"),
        }
    
    def get_ticker(self, limit: int = 100, convert: str = "USD") -> List[Dict[str, Any]]:
        """
        Get ticker data for multiple cryptocurrencies.
        
        ⚠️ USE AS FALLBACK ONLY - CoinGecko/CMC provide better data.
        Only use this if other sources fail.
        
        Args:
            limit: Number of results (0 for all)
            convert: Currency conversion (USD, EUR, BTC, ETH, etc.)
            
        Returns:
            list: Ticker data for multiple coins
            
        Reference: https://api.alternative.me/v2/ticker/
        """
        params = {
            "limit": limit,
            "structure": "array"
        }
        if convert != "USD":
            params["convert"] = convert
        
        response = self._make_request("v2/ticker/", params)
        return response.get("data", [])
    
    def get_ticker_specific(self, symbol_or_id: str, convert: str = "USD") -> Optional[Dict[str, Any]]:
        """
        Get ticker data for a specific cryptocurrency.
        
        ⚠️ USE AS FALLBACK ONLY - CoinGecko/CMC provide better data.
        
        Args:
            symbol_or_id: Coin ID or website slug (e.g., "bitcoin", "1")
            convert: Currency conversion (USD, EUR, BTC, ETH, etc.)
            
        Returns:
            dict: Ticker data for specific coin
            
        Reference: https://api.alternative.me/v2/ticker/(id,name)/
        """
        params = {}
        if convert != "USD":
            params["convert"] = convert
        
        response = self._make_request(f"v2/ticker/{symbol_or_id}/", params)
        data = response.get("data", {})
        
        # Extract first item from dictionary
        if data:
            return list(data.values())[0]
        return None
    
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
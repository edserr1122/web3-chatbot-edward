"""
LunarCrush API client for social analytics and sentiment data.
Documentation: https://lunarcrush.com/developers/api
"""

from typing import Dict, Any, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class LunarCrushClient(BaseAPIClient):
    """Client for LunarCrush API."""
    
    def __init__(self, api_key: str):
        """
        Initialize LunarCrush client.
        
        Args:
            api_key: LunarCrush API key (required)
        """
        super().__init__(api_key=api_key, base_url="https://api.lunarcrush.com/v2/")
        self.min_request_interval = 1.0
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get social and sentiment data for a token.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Social analytics data
        """
        endpoint = "assets"
        params = {
            "symbol": symbol.upper(),
            "data_points": 1,  # Latest data point
            "interval": "day"
        }
        
        response = self._make_request(endpoint, params)
        
        # Extract data
        data_list = response.get("data", [])
        if not data_list:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        data = data_list[0]
        
        return {
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            "price": data.get("price"),
            "volume_24h": data.get("volume_24h"),
            "market_cap": data.get("market_cap"),
            
            # Social metrics
            "galaxy_score": data.get("galaxy_score"),  # Overall score (0-100)
            "alt_rank": data.get("alt_rank"),  # Alternative rank
            "social_volume": data.get("social_volume"),  # Social mentions
            "social_engagement": data.get("social_engagement"),  # Total engagement
            "social_dominance": data.get("social_dominance"),  # % of total crypto social
            "sentiment": data.get("sentiment"),  # Sentiment score
            
            # Social contributors
            "num_contributors": data.get("num_contributors"),
            "tweet_volume": data.get("tweet_volume"),
            "reddit_posts": data.get("reddit_posts"),
            "reddit_engagement": data.get("reddit_engagement"),
            
            # Sentiment breakdown
            "sentiment_absolute": data.get("sentiment_absolute"),
            "sentiment_relative": data.get("sentiment_relative"),
            
            # Volatility
            "volatility": data.get("volatility"),
            
            # Time data
            "timestamp": data.get("time"),
        }
    
    def get_social_metrics(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical social metrics.
        
        Args:
            symbol: Token symbol
            days: Number of days of historical data
            
        Returns:
            dict: Historical social metrics
        """
        endpoint = "assets"
        params = {
            "symbol": symbol.upper(),
            "data_points": days,
            "interval": "day"
        }
        
        return self._make_request(endpoint, params)
    
    def test_connection(self) -> bool:
        """
        Test connection to LunarCrush API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.get_token_data("BTC")
            return True
        except Exception as e:
            logger.error(f"LunarCrush connection test failed: {e}")
            return False
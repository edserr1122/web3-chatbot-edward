"""
CoinMarketCap API client for cryptocurrency market data.
Documentation: https://coinmarketcap.com/api/documentation/v1/
"""

from typing import Dict, Any, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class CoinMarketCapClient(BaseAPIClient):
    """Client for CoinMarketCap API."""
    
    def __init__(self, api_key: str):
        """
        Initialize CoinMarketCap client.
        
        Args:
            api_key: CoinMarketCap API key (required)
        """
        super().__init__(api_key=api_key, base_url="https://pro-api.coinmarketcap.com/v1/")
        self.min_request_interval = 1.0  # Basic plan rate limit
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["X-CMC_PRO_API_KEY"] = self.api_key
        return headers
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get token data from CoinMarketCap.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Token data
        """
        endpoint = "cryptocurrency/quotes/latest"
        params = {
            "symbol": symbol.upper(),
            "convert": "USD"
        }
        
        response = self._make_request(endpoint, params)
        
        # Extract data
        data = response.get("data", {}).get(symbol.upper(), {})
        quote = data.get("quote", {}).get("USD", {})
        
        return {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            "price_usd": quote.get("price"),
            "market_cap": quote.get("market_cap"),
            "market_cap_rank": data.get("cmc_rank"),
            "total_volume": quote.get("volume_24h"),
            "price_change_1h": quote.get("percent_change_1h"),
            "price_change_24h": quote.get("percent_change_24h"),
            "price_change_7d": quote.get("percent_change_7d"),
            "price_change_30d": quote.get("percent_change_30d"),
            "circulating_supply": data.get("circulating_supply"),
            "total_supply": data.get("total_supply"),
            "max_supply": data.get("max_supply"),
            "fully_diluted_valuation": quote.get("fully_diluted_market_cap"),
            "market_cap_dominance": quote.get("market_cap_dominance"),
            "last_updated": quote.get("last_updated"),
        }
    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get token metadata (description, website, etc.).
        
        Args:
            symbol: Token symbol
            
        Returns:
            dict: Token metadata
        """
        endpoint = "cryptocurrency/info"
        params = {
            "symbol": symbol.upper()
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", {}).get(symbol.upper(), {})
    
    def test_connection(self) -> bool:
        """
        Test connection to CoinMarketCap API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.get_token_data("BTC")
            return True
        except Exception as e:
            logger.error(f"CoinMarketCap connection test failed: {e}")
            return False
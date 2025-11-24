"""
Messari API client for crypto research and on-chain data.
Documentation: https://messari.io/api/docs
"""

from typing import Dict, Any, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class MessariClient(BaseAPIClient):
    """Client for Messari API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Messari client.
        
        Args:
            api_key: Messari API key (required)
        """
        super().__init__(api_key=api_key, base_url="https://data.messari.io/api/v1/")
        self.min_request_interval = 0.5
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["x-messari-api-key"] = self.api_key
        return headers
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive asset data from Messari.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Asset data including metrics and profile
        """
        # Get both metrics and profile
        metrics = self.get_metrics(symbol)
        profile = self.get_profile(symbol)
        
        return {
            "symbol": symbol.upper(),
            "metrics": metrics,
            "profile": profile,
        }
    
    def get_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset metrics (market data, on-chain data).
        
        Args:
            symbol: Token symbol
            
        Returns:
            dict: Asset metrics
        """
        endpoint = f"assets/{symbol.lower()}/metrics"
        
        response = self._make_request(endpoint)
        data = response.get("data", {})
        
        market_data = data.get("market_data", {})
        marketcap = data.get("marketcap", {})
        supply = data.get("supply", {})
        on_chain = data.get("on_chain_data", {})
        
        return {
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            
            # Market data
            "price_usd": market_data.get("price_usd"),
            "price_btc": market_data.get("price_btc"),
            "volume_last_24h": market_data.get("volume_last_24_hours"),
            "real_volume_last_24h": market_data.get("real_volume_last_24_hours"),
            "percent_change_24h": market_data.get("percent_change_usd_last_24_hours"),
            
            # Market cap
            "market_cap": marketcap.get("current_marketcap_usd"),
            "market_cap_rank": marketcap.get("rank"),
            "marketcap_dominance": marketcap.get("marketcap_dominance_percent"),
            
            # Supply
            "circulating_supply": supply.get("circulating"),
            "total_supply": supply.get("y_2050"),
            "max_supply": supply.get("y_plus10"),
            "annual_inflation": supply.get("annual_inflation_percent"),
            
            # On-chain (if available)
            "active_addresses": on_chain.get("addresses_count") if on_chain else None,
            "transaction_count": on_chain.get("transaction_count") if on_chain else None,
        }
    
    def get_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset profile (description, technology, etc.).
        
        Args:
            symbol: Token symbol
            
        Returns:
            dict: Asset profile
        """
        endpoint = f"assets/{symbol.lower()}/profile"
        
        response = self._make_request(endpoint)
        data = response.get("data", {})
        
        profile = data.get("profile", {})
        
        return {
            "tagline": profile.get("general", {}).get("overview", {}).get("tagline"),
            "category": profile.get("general", {}).get("overview", {}).get("category"),
            "sector": profile.get("general", {}).get("overview", {}).get("sector"),
            "description": profile.get("general", {}).get("overview", {}).get("project_details"),
            "technology": profile.get("technology", {}).get("overview", {}).get("technology_details"),
            "consensus_mechanism": profile.get("technology", {}).get("overview", {}).get("consensus_mechanism"),
        }
    
    def test_connection(self) -> bool:
        """
        Test connection to Messari API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.get_metrics("BTC")
            return True
        except Exception as e:
            logger.error(f"Messari connection test failed: {e}")
            return False
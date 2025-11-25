"""
Messari API client for crypto research and on-chain data.
Documentation: https://developer.messari.io/api-reference
API v2: https://api.messari.io/metrics/v2/

⚠️ FREE TIER LIMITATION:
Only 2 endpoints are available with free API keys:
1. GET /metrics/v2/assets - List of assets with pagination
2. GET /metrics/v2/assets/details - Asset details by assetIDs

All other endpoints (metrics, news, research, ROI, ATH, time-series) require paid plans.
"""

from typing import Dict, Any, Optional, List
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class MessariClient(BaseAPIClient):
    """
    Client for Messari API v2 (Free Tier).
    
    ⚠️ FREE TIER: Only supports /assets and /assets/details endpoints.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Messari client.
        
        Args:
            api_key: Messari API key (free tier)
        """
        # Using Messari API v2 base URL
        super().__init__(api_key=api_key, base_url="https://api.messari.io/")
        self.min_request_interval = 0.5
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key."""
        headers = super()._get_headers()
        headers["X-Messari-API-Key"] = self.api_key
        return headers
    
    def get_assets_list(self, limit: int = 10, page: int = 1) -> List[Dict[str, Any]]:
        """
        Get list of assets (FREE TIER ENDPOINT).
        
        ✅ FREE TIER: This endpoint is available with free API keys.
        
        Args:
            limit: Number of results per page (default: 10)
            page: Page number (default: 1)
            
        Returns:
            list: List of assets
        """
        endpoint = "metrics/v2/assets"
        params = {
            "limit": limit,
            "page": page,
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", [])
    
    def get_asset_details(self, *asset_ids: str) -> List[Dict[str, Any]]:
        """
        Get asset details by asset IDs (FREE TIER ENDPOINT).
        
        ✅ FREE TIER: This endpoint is available with free API keys.
        
        Args:
            *asset_ids: One or more asset IDs (e.g., "bitcoin", "ethereum")
                       Use lowercase full names, not symbols
            
        Returns:
            list: Asset details for requested assets
        """
        endpoint = "metrics/v2/assets/details"
        
        # Join asset IDs with comma
        assets_param = ",".join(asset_ids)
        
        params = {
            "assetIDs": assets_param
        }
        
        response = self._make_request(endpoint, params)
        data_list = response.get("data", [])
        
        # Parse and structure the response
        results = []
        for data in data_list if isinstance(data_list, list) else [data_list]:
            results.append({
                "id": data.get("id"),
                "symbol": data.get("symbol"),
                "name": data.get("name"),
                "slug": data.get("slug"),
                "tagline": data.get("tagline"),
                "description": data.get("overview"),
                "category": data.get("category"),
                "sector": data.get("sector"),
                "tags": data.get("tags", []),
                "technology": data.get("technology"),
                "background": data.get("background"),
                "launch_date": data.get("token_distribution", {}).get("launch_date") if data.get("token_distribution") else None,
            })
        
        return results
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset details by symbol (wrapper for compatibility).
        
        ⚠️ This converts common symbols to Messari asset IDs.
        Only works for major cryptocurrencies.
        
        Args:
            symbol: Token symbol (e.g., "BTC", "ETH")
            
        Returns:
            dict: Asset details or empty dict if not found
        """
        # Map common symbols to Messari asset IDs (lowercase full names)
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDT": "tether",
            "BNB": "binance-coin",
            "SOL": "solana",
            "XRP": "xrp",
            "USDC": "usd-coin",
            "ADA": "cardano",
            "AVAX": "avalanche",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "TRX": "tron",
            "MATIC": "polygon",
            "LTC": "litecoin",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "ATOM": "cosmos",
            "XMR": "monero",
            "ETC": "ethereum-classic",
            "BCH": "bitcoin-cash",
        }
        
        asset_id = symbol_to_id.get(symbol.upper())
        
        if not asset_id:
            logger.warning(f"Messari: Symbol {symbol} not in supported list (free tier limitation)")
            return {}
        
        try:
            results = self.get_asset_details(asset_id)
            if results:
                # Return first result in compatible format
                details = results[0]
                return {
                    "symbol": details.get("symbol"),
                    "name": details.get("name"),
                    "details": details,
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get Messari data for {symbol}: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test connection to Messari API (free tier).
        
        Returns:
            bool: True if connection successful
        """
        try:
            endpoint = "metrics/v2/assets"
            params = {"limit": 1, "page": 1}
            response = self._make_request(
                endpoint,
                params,
                force_refresh=True,
                use_cache=False,
            )
            return bool(response.get("data"))
        except Exception as e:
            logger.error(f"Messari connection test failed: {e}")
            return False
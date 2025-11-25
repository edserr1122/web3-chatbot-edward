"""
CoinGecko API client for cryptocurrency market data.
Documentation: https://www.coingecko.com/en/api/documentation
"""

from typing import Dict, Any, Optional, List
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class CoinGeckoClient(BaseAPIClient):
    """Client for CoinGecko API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: CoinGecko API key (optional for free tier)
        """
        # Use Pro API if key provided, otherwise free API
        base_url = "https://api.coingecko.com/api/v3/" if not api_key else "https://pro-api.coingecko.com/api/v3/"
        super().__init__(api_key=api_key, base_url=base_url)
        self.min_request_interval = 1.5 if not api_key else 0.5  # Free tier rate limit
        self._circuit_breaker_cooldown = 120  # 2 minutes cooldown for CoinGecko rate limits
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key if available."""
        headers = super()._get_headers()
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        return headers
    
    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """
        Convert symbol to CoinGecko coin ID.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            str: CoinGecko coin ID (e.g., "bitcoin")
        """
        # Common mappings
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "SOL": "solana",
            "ADA": "cardano",
            "XRP": "ripple",
            "DOT": "polkadot",
            "AVAX": "avalanche-2",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "ATOM": "cosmos",
            "UNI": "uniswap",
            "LTC": "litecoin",
            "ETC": "ethereum-classic",
            "XLM": "stellar",
            "ALGO": "algorand",
            "VET": "vechain",
            "ICP": "internet-computer",
            "FIL": "filecoin",
            "TRX": "tron",
            "AAVE": "aave",
            "MKR": "maker",
            "SNX": "synthetix-network-token",
            "TAO": "bittensor",  # Bittensor
        }
        
        return symbol_to_id.get(symbol.upper())
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive token data from CoinGecko.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Token data including price, market cap, volume, etc.
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"Unknown token symbol: {symbol}")
        
        endpoint = f"coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }
        
        data = self._make_request(endpoint, params)
        
        # Extract and normalize data
        market_data = data.get("market_data", {})
        
        return {
            "id": data.get("id"),
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name"),
            "price_usd": market_data.get("current_price", {}).get("usd"),
            "market_cap": market_data.get("market_cap", {}).get("usd"),
            "market_cap_rank": data.get("market_cap_rank"),
            "total_volume": market_data.get("total_volume", {}).get("usd"),
            "high_24h": market_data.get("high_24h", {}).get("usd"),
            "low_24h": market_data.get("low_24h", {}).get("usd"),
            "price_change_24h": market_data.get("price_change_percentage_24h"),
            "price_change_7d": market_data.get("price_change_percentage_7d"),
            "price_change_30d": market_data.get("price_change_percentage_30d"),
            "circulating_supply": market_data.get("circulating_supply"),
            "total_supply": market_data.get("total_supply"),
            "max_supply": market_data.get("max_supply"),
            "ath": market_data.get("ath", {}).get("usd"),
            "ath_date": market_data.get("ath_date", {}).get("usd"),
            "atl": market_data.get("atl", {}).get("usd"),
            "atl_date": market_data.get("atl_date", {}).get("usd"),
            "fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get("usd"),
            "community_data": data.get("community_data", {}),
            "developer_data": data.get("developer_data", {}),
        }
    
    def get_market_chart(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical market data (price, volume, market cap).
        
        Args:
            symbol: Token symbol
            days: Number of days of historical data
            
        Returns:
            dict: Historical data with prices, volumes, market caps
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"Unknown token symbol: {symbol}")
        
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 1 else "hourly"
        }
        
        return self._make_request(endpoint, params)
    
    def get_ohlc(self, symbol: str, days: int = 7) -> List[List[float]]:
        """
        Get OHLC (Open, High, Low, Close) data.
        
        Args:
            symbol: Token symbol
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            
        Returns:
            list: OHLC data [[timestamp, open, high, low, close], ...]
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"Unknown token symbol: {symbol}")
        
        endpoint = f"coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        return self._make_request(endpoint, params)
    
    def get_simple_price(self, symbols: List[str], include_market_cap: bool = True, 
                         include_24h_vol: bool = True, include_24h_change: bool = True) -> Dict[str, Any]:
        """
        Get simple price data for multiple coins (LIGHTWEIGHT - faster than /coins/{id}).
        
        ⚡ Use this for:
        - Quick price checks
        - Comparative analysis (multiple tokens)
        - When you don't need full metadata
        
        Args:
            symbols: List of token symbols (e.g., ["BTC", "ETH"])
            include_market_cap: Include market cap data
            include_24h_vol: Include 24h volume
            include_24h_change: Include 24h price change
            
        Returns:
            dict: Price data for each coin
        """
        # Convert symbols to coin IDs
        coin_ids = []
        for symbol in symbols:
            coin_id = self._get_coin_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
        
        if not coin_ids:
            raise ValueError(f"No valid symbols provided: {symbols}")
        
        endpoint = "simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_market_cap": str(include_market_cap).lower(),
            "include_24hr_vol": str(include_24h_vol).lower(),
            "include_24hr_change": str(include_24h_change).lower(),
        }
        
        return self._make_request(endpoint, params)
    
    def get_coins_markets(self, symbols: Optional[List[str]] = None, 
                          per_page: int = 250, page: int = 1) -> List[Dict[str, Any]]:
        """
        Get market data for multiple coins in one call (EFFICIENT for comparative analysis).
        
        ⚡ Use this for:
        - Comparative analysis
        - Top N coins by market cap
        - Getting multiple coins data efficiently
        
        Args:
            symbols: Optional list of symbols to filter (if None, returns top coins)
            per_page: Results per page (max 250)
            page: Page number
            
        Returns:
            list: Market data for each coin
        """
        endpoint = "coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(per_page, 250),
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d"
        }
        
        # Filter by specific coin IDs if symbols provided
        if symbols:
            coin_ids = []
            for symbol in symbols:
                coin_id = self._get_coin_id(symbol)
                if coin_id:
                    coin_ids.append(coin_id)
            
            if coin_ids:
                params["ids"] = ",".join(coin_ids)
        
        return self._make_request(endpoint, params)
    
    def get_global_data(self) -> Dict[str, Any]:
        """
        Get global cryptocurrency market data.
        
        Returns:
            dict: Global market stats (total market cap, volume, BTC dominance, etc.)
        """
        endpoint = "global"
        response = self._make_request(endpoint)
        return response.get("data", {})
    
    def get_trending(self) -> Dict[str, Any]:
        """
        Get trending coins in the last 24 hours.
        
        Returns:
            dict: Trending coins, NFTs, and categories
        """
        endpoint = "search/trending"
        return self._make_request(endpoint)
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all coin categories with market data.
        
        Returns:
            list: Categories with market cap, volume, market cap change
        """
        endpoint = "coins/categories"
        return self._make_request(endpoint)
    
    def test_connection(self) -> bool:
        """
        Test connection to CoinGecko API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            endpoint = "ping"
            response = self._make_request(endpoint)
            return response.get("gecko_says") == "(V3) To the Moon!"
        except Exception as e:
            logger.error(f"CoinGecko connection test failed: {e}")
            return False
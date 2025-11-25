"""
CoinMarketCap API client for cryptocurrency market data.
Documentation: https://coinmarketcap.com/api/documentation/v1/

FREE TIER (Basic Plan) - 11 Endpoints, 10K credits/mo:
✅ /v1/cryptocurrency/quotes/latest - Current prices
✅ /v1/cryptocurrency/listings/latest - Top coins by market cap
✅ /v1/cryptocurrency/info - Metadata
✅ /v1/cryptocurrency/map - Symbol-to-ID mapping
✅ /v1/global-metrics/quotes/latest - Global market stats
✅ /v1/cryptocurrency/ohlcv/latest - Latest OHLC only

PAID TIER ONLY (Requires Hobbyist+ plan):
❌ /v2/cryptocurrency/ohlcv/historical - Historical OHLC (requires paid plan!)
❌ /v2/cryptocurrency/price-performance-stats/latest - May require paid plan
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
        # Base URL without version - endpoints include their own version (v1 or v2)
        super().__init__(api_key=api_key, base_url="https://pro-api.coinmarketcap.com/")
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
        endpoint = "v1/cryptocurrency/quotes/latest"
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
        endpoint = "v1/cryptocurrency/info"
        params = {
            "symbol": symbol.upper()
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", {}).get(symbol.upper(), {})
    
    def get_quotes_latest(self, symbols: list) -> Dict[str, Any]:
        """
        Get latest market quotes for multiple cryptocurrencies (BATCH OPERATION).
        
        ⚡ EFFICIENT: Fetch multiple tokens in ONE API call.
        Recommended over individual get_token_data() calls.
        
        Args:
            symbols: List of token symbols (e.g., ["BTC", "ETH", "SOL"])
            
        Returns:
            dict: Market data for all requested tokens
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyQuotesLatest
        """
        endpoint = "v1/cryptocurrency/quotes/latest"
        params = {
            "symbol": ",".join([s.upper() for s in symbols]),
            "convert": "USD"
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", {})
    
    def get_listings_latest(self, start: int = 1, limit: int = 100, convert: str = "USD") -> list:
        """
        Get latest cryptocurrency listings (top N by market cap).
        
        ⚡ Use for:
        - Getting top cryptocurrencies by market cap
        - Market overview
        - Comparative analysis of top coins
        
        Args:
            start: Starting rank (default: 1)
            limit: Number of results (1-5000, default: 100)
            convert: Currency for price conversion (default: USD)
            
        Returns:
            list: List of cryptocurrency data sorted by market cap
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest
        """
        endpoint = "v1/cryptocurrency/listings/latest"
        params = {
            "start": start,
            "limit": min(limit, 5000),
            "convert": convert
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", [])
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get global cryptocurrency market metrics.
        
        Returns:
            dict: Global market data including:
                - Total market cap
                - Total 24h volume
                - BTC dominance
                - ETH dominance
                - Active cryptocurrencies
                - Active exchanges
                - DeFi volume/market cap
                
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV1GlobalmetricsQuotesLatest
        """
        endpoint = "v1/global-metrics/quotes/latest"
        params = {"convert": "USD"}
        
        response = self._make_request(endpoint, params)
        data = response.get("data", {})
        quote = data.get("quote", {}).get("USD", {})
        
        return {
            "total_market_cap": quote.get("total_market_cap"),
            "total_volume_24h": quote.get("total_volume_24h"),
            "btc_dominance": data.get("btc_dominance"),
            "eth_dominance": data.get("eth_dominance"),
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "active_exchanges": data.get("active_exchanges"),
            "active_market_pairs": data.get("active_market_pairs"),
            "defi_volume_24h": data.get("defi_volume_24h"),
            "defi_market_cap": data.get("defi_market_cap"),
            "stablecoin_volume_24h": data.get("stablecoin_volume_24h"),
            "stablecoin_market_cap": data.get("stablecoin_market_cap"),
            "derivatives_volume_24h": data.get("derivatives_volume_24h"),
            "last_updated": data.get("last_updated"),
        }
    
    def get_ohlcv_latest(self, symbol: str, convert: str = "USD") -> Dict[str, Any]:
        """
        Get latest OHLCV (Open, High, Low, Close, Volume) data.
        
        ⚡ Use for quick OHLCV check without historical data.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            convert: Currency for price conversion (default: USD)
            
        Returns:
            dict: Latest OHLCV data
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyOhlcvLatest
        """
        endpoint = "v1/cryptocurrency/ohlcv/latest"
        params = {
            "symbol": symbol.upper(),
            "convert": convert
        }
        
        response = self._make_request(endpoint, params)
        data = response.get("data", {}).get(symbol.upper(), {})
        quote = data.get("quote", {}).get(convert, {})
        
        return {
            "symbol": data.get("symbol"),
            "time_open": quote.get("time_open"),
            "time_close": quote.get("time_close"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "market_cap": quote.get("market_cap"),
            "timestamp": quote.get("timestamp"),
        }
    
    def get_ohlcv_historical(self, symbol: str, time_period: str = "daily", 
                             count: int = 90, convert: str = "USD") -> list:
        """
        Get historical OHLCV (Open, High, Low, Close, Volume) data.
        
        ⚠️ PAID SUBSCRIPTION REQUIRED - NOT AVAILABLE IN FREE (BASIC) PLAN!
        Requires: Hobbyist plan or higher
        
        This endpoint is kept for reference but will fail with 403 error on free tier.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            time_period: Time period (hourly, daily, weekly, monthly, yearly)
            count: Number of periods to return (default: 90)
            convert: Currency for price conversion (default: USD)
            
        Returns:
            list: Historical OHLCV data (empty on free tier)
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV2CryptocurrencyOhlcvHistorical
        """
        endpoint = "v2/cryptocurrency/ohlcv/historical"
        params = {
            "symbol": symbol.upper(),
            "time_period": time_period,
            "count": count,
            "convert": convert
        }
        
        response = self._make_request(endpoint, params)
        data = response.get("data", {})
        
        # Extract OHLCV data
        if not data:
            return []
        
        # Get the first symbol data (in case of multiple)
        symbol_data = data.get(symbol.upper(), {})
        quotes = symbol_data.get("quotes", [])
        
        # Format data for technical analysis
        ohlcv_data = []
        for quote in quotes:
            quote_data = quote.get("quote", {}).get(convert, {})
            ohlcv_data.append({
                "time_open": quote.get("time_open"),
                "time_close": quote.get("time_close"),
                "open": quote_data.get("open"),
                "high": quote_data.get("high"),
                "low": quote_data.get("low"),
                "close": quote_data.get("close"),
                "volume": quote_data.get("volume"),
                "market_cap": quote_data.get("market_cap"),
                "timestamp": quote_data.get("timestamp"),
            })
        
        return ohlcv_data
    
    def get_price_performance_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get price performance statistics (all-time high, all-time low, etc.).
        
        ⚠️ MAY REQUIRE PAID SUBSCRIPTION - Check your plan limits.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Price performance statistics
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV2CryptocurrencyPriceperformancestatsLatest
        """
        endpoint = "v2/cryptocurrency/price-performance-stats/latest"
        params = {"symbol": symbol.upper()}
        
        response = self._make_request(endpoint, params)
        data = response.get("data", {}).get(symbol.upper(), {})
        
        return {
            "symbol": data.get("symbol"),
            "all_time_high": data.get("periods", {}).get("all_time", {}).get("quote", {}).get("USD", {}).get("high"),
            "all_time_high_timestamp": data.get("periods", {}).get("all_time", {}).get("high_timestamp"),
            "all_time_low": data.get("periods", {}).get("all_time", {}).get("quote", {}).get("USD", {}).get("low"),
            "all_time_low_timestamp": data.get("periods", {}).get("all_time", {}).get("low_timestamp"),
            "periods": data.get("periods", {}),
        }
    
    def get_map(self, listing_status: str = "active", start: int = 1, limit: int = 5000) -> list:
        """
        Get CoinMarketCap ID map for symbol-to-ID conversion.
        
        ⚡ Use for:
        - Converting symbols to CMC IDs
        - Getting list of all available cryptocurrencies
        
        Args:
            listing_status: active, inactive, or untracked (default: active)
            start: Starting rank (default: 1)
            limit: Number of results (default: 5000)
            
        Returns:
            list: List of cryptocurrency mappings
            
        Reference: https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyMap
        """
        endpoint = "v1/cryptocurrency/map"
        params = {
            "listing_status": listing_status,
            "start": start,
            "limit": limit
        }
        
        response = self._make_request(endpoint, params)
        return response.get("data", [])
    
    def test_connection(self) -> bool:
        """
        Test connection to CoinMarketCap API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            endpoint = "v1/cryptocurrency/map"
            params = {"limit": 1}
            response = self._make_request(
                endpoint,
                params,
                force_refresh=True,
                use_cache=False
            )
            return bool(response.get("data"))
        except Exception as e:
            logger.error(f"CoinMarketCap connection test failed: {e}")
            return False
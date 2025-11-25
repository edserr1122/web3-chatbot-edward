"""
CoinCap API v3 Client
Cryptocurrency data API with real-time pricing and historical data.
Base URL: https://rest.coincap.io/v3/
Documentation: https://rest.coincap.io/api-docs.json
API Dashboard: https://pro.coincap.io/dashboard
"""

from typing import Dict, Any, List, Optional
import logging
from src.data_sources.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class CoinCapClient(BaseAPIClient):
    """
    Client for CoinCap API v3.
    
    ✅ REQUIRES API KEY: Get yours at https://pro.coincap.io/dashboard
    
    API Tiers:
    - Free: 4,000 credits/month (1 credit per API call base + 1 per 2.5KB data)
    - Basic: 75,000 credits/month
    - Growth: 225,000 credits/month
    - Professional: 675,000 credits/month
    - Enterprise: 5,000,000 credits/month
    
    All tiers limited to 600 API calls per minute.
    
    Endpoints Used:
    - /price/bysymbol/{symbol} - Current price
    - /assets/{slug}/history - Historical data
    
    Documentation: https://rest.coincap.io/api-docs.json
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinCap v3 client.
        
        Args:
            api_key: CoinCap API key (required - get from https://pro.coincap.io/dashboard)
        """
        super().__init__(api_key=api_key, base_url="https://rest.coincap.io/v3/")
        self.min_request_interval = 0.1  # 600 calls/min = ~10 calls/sec = 0.1s interval
        self._circuit_breaker_cooldown = 120  # 2 minutes cooldown for rate limits
        self.slug_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "BNB": "binance-coin",
            "ADA": "cardano",
            "XRP": "xrp",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "MATIC": "polygon",
            "LTC": "litecoin",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "ATOM": "cosmos",
            "TAO": "bittensor",
            "TRX": "tron",
            "AVAX": "avalanche",
            "USDT": "tether",
            "USDC": "usd-coin",
        }
        
        if not api_key:
            logger.warning("⚠️  CoinCap v3 requires API key - requests will fail without it")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current token data by symbol.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Token data (NOTE: CoinCap v3 /price/bysymbol only returns price)
        """
        endpoint = f"price/bysymbol/{symbol.upper()}"
        
        response = self._make_request(endpoint)
        
        # CoinCap v3 /price/bysymbol response format:
        # {"timestamp": 1764064128814, "data": ["86959.000000000000000000"]}
        # data[0] is the price as a string
        
        if "data" not in response or not isinstance(response["data"], list) or len(response["data"]) == 0:
            logger.error(f"Unexpected CoinCap response structure: {response}")
            raise Exception(f"No price data returned for symbol {symbol}")
        
        price_str = response["data"][0]
        
        # Parse the price string to float
        try:
            price = float(price_str)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse price from CoinCap: {price_str}")
            raise Exception(f"Invalid price data for {symbol}: {e}")
        
        # CoinCap v3 /price/bysymbol endpoint ONLY returns price
        # Other fields (market cap, volume, etc.) are not available from this endpoint
        return {
            "symbol": symbol.upper(),
            "price_usd": price,
            "market_cap": 0,  # Not available from /price/bysymbol endpoint
            "total_volume": 0,  # Not available from /price/bysymbol endpoint
            "price_change_24h": 0,  # Not available from /price/bysymbol endpoint
        }
    
    def get_history(
        self, 
        symbol: str, 
        interval: str = "d1",  # Valid: m1, m5, m15, m30, h1, h2, h6, h12, d1
        start: Optional[int] = None,  # Unix timestamp in milliseconds
        end: Optional[int] = None      # Unix timestamp in milliseconds
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data using CoinCap v3 API.
        
        Endpoint: GET /v3/assets/{slug}/history
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            interval: Valid intervals: m1, m5, m15, m30, h1, h2, h6, h12, d1
            start: Start time (Unix timestamp in milliseconds)
            end: End time (Unix timestamp in milliseconds)
            
        Returns:
            list: Historical price data points
            
        Documentation: https://rest.coincap.io/v3/assets/{slug}/history
        """
        slug = self._get_slug(symbol)
        
        # CoinCap v3: /assets/{slug}/history
        endpoint = f"assets/{slug}/history"
        
        params = {"interval": interval}
        if start:
            params["start"] = int(start)
        if end:
            params["end"] = int(end)
        
        response = self._make_request(endpoint, params)
        
        # v3 API returns data array
        return response.get("data", [])

    def get_candlesticks(
        self,
        symbol: str,
        interval: str = "d1",
        limit: int = 180
    ) -> List[List[float]]:
        """
        Get candlestick data from CoinCap /ta/{slug}/candlesticks endpoint.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            interval: Valid intervals: m1, m5, m15, m30, h1, h2, h6, h12, d1
            limit: Number of points (CoinCap allows up to 2000 per page)
        
        Returns:
            list: List of [timestamp, open, high, low, close]
        """
        slug = self._get_slug(symbol)
        endpoint = f"ta/{slug}/candlesticks"
        params = {
            "interval": interval,
            "limit": min(limit, 2000)
        }
        
        response = self._make_request(endpoint, params)
        candles = response.get("data", [])
        
        ohlc_data = []
        for candle in candles:
            try:
                timestamp = int(candle.get("timestamp", 0))
                open_price = float(candle.get("open", candle.get("o", 0)))
                high_price = float(candle.get("high", candle.get("h", 0)))
                low_price = float(candle.get("low", candle.get("l", 0)))
                close_price = float(candle.get("close", candle.get("c", 0)))
                ohlc_data.append([timestamp, open_price, high_price, low_price, close_price])
            except (TypeError, ValueError):
                continue
        
        return ohlc_data

    def _get_slug(self, symbol: str) -> str:
        """Convert symbol to CoinCap slug/asset id."""
        return self.slug_map.get(symbol.upper(), symbol.lower())
    
    def get_market_chart(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical market data (compatible with CoinGecko format).
        
        Uses CoinCap v3 API: /assets/{slug}/history
        
        Args:
            symbol: Token symbol
            days: Number of days back
            
        Returns:
            dict: Market chart data in CoinGecko-compatible format
        """
        import time
        
        try:
            # Calculate timestamps (in milliseconds for CoinCap v3 API)
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # N days ago
            
            # Get historical data from v3 API: /assets/{slug}/history
            # Note: CoinCap v3 uses "d1" for daily (not "1d")
            history = self.get_history(symbol, interval="d1", start=start_time, end=end_time)
            
            # Convert CoinCap v3 format to CoinGecko-compatible format
            # v3 response format may vary - typically: [{"timestamp": 1234567890000, "price": "123.45", ...}, ...]
            prices = []
            market_caps = []
            total_volumes = []
            
            for point in history:
                # v3 API may return 'timestamp' or 'time', and 'price' or 'priceUsd'
                timestamp_ms = int(point.get("timestamp", point.get("time", 0)))
                price_value = point.get("price", point.get("priceUsd", "0"))
                
                try:
                    price = float(price_value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid price in CoinCap response: {price_value}")
                    continue
                
                prices.append([timestamp_ms, price])
                
                # CoinCap v3 /assets/{slug}/history does not provide market cap/volume
                market_caps.append([timestamp_ms, 0])  # Not available
                total_volumes.append([timestamp_ms, 0])  # Not available
            
            return {
                "prices": prices,
                "market_caps": market_caps,  # Not available from CoinCap v3
                "total_volumes": total_volumes  # Not available from CoinCap v3
            }
        except Exception as e:
            logger.warning(f"CoinCap historical data fetch failed for {symbol}: {e}")
            # Return empty data structure
            return {
                "prices": [],
                "market_caps": [],
                "total_volumes": []
            }
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.api_key:
                logger.warning("⚠️  CoinCap API key not configured - skipping connection test")
                return False
            
            self.get_token_data("BTC")
            logger.info("✅ CoinCap API connection test successful")
            return True
        except Exception as e:
            logger.warning(f"⚠️  CoinCap API connection test failed: {e}")
            return False


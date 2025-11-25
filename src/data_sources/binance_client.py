"""
Binance Public API Client
Free API for cryptocurrency market data with excellent historical price support.
Documentation: https://binance-docs.github.io/apidocs/spot/en/
"""

from typing import Dict, Any, List, Optional
import logging
from src.data_sources.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class BinanceClient(BaseAPIClient):
    """
    Client for Binance Public API.
    
    ✅ FREE TIER: No API key required
    - Historical OHLC (candlestick) data
    - Market data for all Binance-listed tokens
    - Rate limit: 1200 requests/minute (very high!)
    
    ⚠️ LIMITATION: Only supports tokens listed on Binance
    
    Documentation: https://binance-docs.github.io/apidocs/spot/en/
    """
    
    def __init__(self):
        """Initialize Binance client (no API key needed)."""
        super().__init__(api_key=None, base_url="https://api.binance.com/api/v3/")
        self.min_request_interval = 0.05  # 1200 requests/min = 0.05s between requests
        self._circuit_breaker_cooldown = 60  # 1 minute cooldown for rate limits
        
        # Symbol mapping (Binance uses trading pairs like BTCUSDT)
        self.symbol_to_pair = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "USDT": "USDTUSDT",
            "BNB": "BNBUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "USDC": "USDCUSDT",
            "ADA": "ADAUSDT",
            "AVAX": "AVAXUSDT",
            "DOGE": "DOGEUSDT",
            "DOT": "DOTUSDT",
            "TRX": "TRXUSDT",
            "MATIC": "MATICUSDT",
            "LTC": "LTCUSDT",
            "LINK": "LINKUSDT",
            "UNI": "UNIUSDT",
            "ATOM": "ATOMUSDT",
            "XMR": "XMRUSDT",
            "ETC": "ETCUSDT",
            "BCH": "BCHUSDT",
            "TAO": "TAOUSDT",  # Bittensor (may be geo-restricted)
        }
    
    def _get_trading_pair(self, symbol: str) -> str:
        """Convert symbol to Binance trading pair."""
        return self.symbol_to_pair.get(symbol.upper(), f"{symbol.upper()}USDT")
    
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current token data.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Token data
        """
        pair = self._get_trading_pair(symbol)
        
        # Get 24hr ticker data
        endpoint = "ticker/24hr"
        params = {"symbol": pair}
        
        data = self._make_request(endpoint, params)
        
        return {
            "symbol": symbol.upper(),
            "trading_pair": pair,
            "price_usd": float(data.get("lastPrice", 0)),
            "price_change_24h": float(data.get("priceChangePercent", 0)),
            "high_24h": float(data.get("highPrice", 0)),
            "low_24h": float(data.get("lowPrice", 0)),
            "total_volume": float(data.get("quoteVolume", 0)),  # Volume in USDT
            "volume_24h": float(data.get("volume", 0)),  # Volume in base currency
            "number_of_trades": int(data.get("count", 0)),
        }
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str = "1d",  # 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
        limit: int = 100,
        start_time: Optional[int] = None,  # Unix timestamp in milliseconds
        end_time: Optional[int] = None      # Unix timestamp in milliseconds
    ) -> List[List]:
        """
        Get historical candlestick/kline data (OHLCV).
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of data points (max 1000)
            start_time: Start time (Unix timestamp in milliseconds)
            end_time: End time (Unix timestamp in milliseconds)
            
        Returns:
            list: Kline data - each entry is [timestamp, open, high, low, close, volume, ...]
            
        Reference: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        """
        pair = self._get_trading_pair(symbol)
        endpoint = "klines"
        
        params = {
            "symbol": pair,
            "interval": interval,
            "limit": min(limit, 1000)  # Max 1000
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        return self._make_request(endpoint, params)
    
    def get_market_chart(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical market data (compatible with CoinGecko format).
        
        Args:
            symbol: Token symbol
            days: Number of days back
            
        Returns:
            dict: Market chart data in CoinGecko-compatible format
        """
        import time
        
        # Calculate timestamps
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # N days ago
        
        # Determine interval based on days
        if days <= 1:
            interval = "1h"  # Hourly for 1 day
        else:
            interval = "1d"  # Daily for longer periods
        
        # Get klines data
        try:
            klines = self.get_klines(
                symbol, 
                interval=interval, 
                start_time=start_time,
                end_time=end_time,
                limit=days if interval == "1d" else 24 * days
            )
        except Exception as e:
            logger.warning(f"Binance klines fetch failed for {symbol}: {e}")
            return {"prices": [], "market_caps": [], "total_volumes": []}
        
        # Convert to CoinGecko-compatible format
        prices = []
        total_volumes = []
        
        for kline in klines:
            # Kline format: [timestamp, open, high, low, close, volume, close_time, ...]
            timestamp = int(kline[0])  # Open time
            close_price = float(kline[4])  # Close price
            volume = float(kline[7])  # Quote asset volume (in USDT)
            
            prices.append([timestamp, close_price])
            total_volumes.append([timestamp, volume])
        
        return {
            "prices": prices,
            "market_caps": [],  # Binance doesn't provide historical market cap
            "total_volumes": total_volumes
        }
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Use ping endpoint
            endpoint = "ping"
            self._make_request(
                endpoint,
                force_refresh=True,
                use_cache=False
            )
            logger.info("Binance API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Binance API connection test failed: {e}")
            return False


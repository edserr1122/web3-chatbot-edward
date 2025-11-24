"""
Price Analysis Module
Analyzes historical price trends, volatility, support/resistance levels.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import statistics
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.coinmarketcap_client import CoinMarketCapClient
from src.utils.formatters import OutputFormatter

logger = logging.getLogger(__name__)


class PriceAnalyzer:
    """Performs price analysis on cryptocurrency tokens."""
    
    def __init__(
        self,
        coingecko_client: Optional[CoinGeckoClient] = None,
        coinmarketcap_client: Optional[CoinMarketCapClient] = None
    ):
        """
        Initialize Price Analyzer.
        
        Args:
            coingecko_client: CoinGecko API client
            coinmarketcap_client: CoinMarketCap API client
        """
        self.coingecko = coingecko_client
        self.coinmarketcap = coinmarketcap_client
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive price analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Price analysis results
        """
        try:
            # Get current price data
            current_data = self._get_current_price_data(symbol)
            
            # Get historical data
            historical_data = self._get_historical_data(symbol, days=30)
            
            # Analyze price trends
            trends = self._analyze_trends(current_data, historical_data)
            
            # Analyze volatility
            volatility = self._analyze_volatility(historical_data)
            
            # Find support and resistance
            support_resistance = self._find_support_resistance(historical_data, current_data)
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "current_price": current_data.get("price_usd"),
                "price_changes": self._extract_price_changes(current_data),
                "trends": trends,
                "volatility": volatility,
                "support_resistance": support_resistance,
                "ath_atl": self._analyze_ath_atl(current_data),
                "summary": self._generate_summary(current_data, trends, volatility),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Price analysis failed for {symbol}: {e}")
            raise
    
    def _get_current_price_data(self, symbol: str) -> Dict[str, Any]:
        """Get current price data."""
        # Try CoinGecko first
        if self.coingecko:
            try:
                return self.coingecko.get_token_data(symbol)
            except Exception as e:
                logger.warning(f"CoinGecko fetch failed: {e}")
        
        # Fallback to CoinMarketCap
        if self.coinmarketcap:
            try:
                return self.coinmarketcap.get_token_data(symbol)
            except Exception as e:
                logger.warning(f"CoinMarketCap fetch failed: {e}")
        
        raise Exception("No price data available from any source")
    
    def _get_historical_data(self, symbol: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """Get historical price data."""
        if self.coingecko:
            try:
                return self.coingecko.get_market_chart(symbol, days=days)
            except Exception as e:
                logger.warning(f"Historical data fetch failed: {e}")
        
        return None
    
    def _extract_price_changes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract price change percentages."""
        return {
            "1h": data.get("price_change_1h"),
            "24h": data.get("price_change_24h"),
            "7d": data.get("price_change_7d"),
            "30d": data.get("price_change_30d"),
        }
    
    def _analyze_trends(
        self, 
        current_data: Dict[str, Any], 
        historical_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze price trends."""
        trends = {}
        
        # Short-term trend (24h)
        change_24h = current_data.get("price_change_24h", 0)
        trends["24h"] = {
            "direction": "up" if change_24h > 0 else "down" if change_24h < 0 else "flat",
            "percentage": change_24h,
            "strength": self._assess_trend_strength(abs(change_24h)),
        }
        
        # Medium-term trend (7d)
        change_7d = current_data.get("price_change_7d", 0)
        trends["7d"] = {
            "direction": "up" if change_7d > 0 else "down" if change_7d < 0 else "flat",
            "percentage": change_7d,
            "strength": self._assess_trend_strength(abs(change_7d)),
        }
        
        # Long-term trend (30d)
        change_30d = current_data.get("price_change_30d", 0)
        trends["30d"] = {
            "direction": "up" if change_30d > 0 else "down" if change_30d < 0 else "flat",
            "percentage": change_30d,
            "strength": self._assess_trend_strength(abs(change_30d)),
        }
        
        # Overall trend assessment
        trends["overall"] = self._assess_overall_trend(change_24h, change_7d, change_30d)
        
        return trends
    
    def _assess_trend_strength(self, abs_percentage: float) -> str:
        """Assess the strength of a trend."""
        if abs_percentage > 20:
            return "Very Strong"
        elif abs_percentage > 10:
            return "Strong"
        elif abs_percentage > 5:
            return "Moderate"
        elif abs_percentage > 2:
            return "Weak"
        else:
            return "Minimal"
    
    def _assess_overall_trend(
        self, 
        change_24h: float, 
        change_7d: float, 
        change_30d: float
    ) -> str:
        """Assess overall trend direction."""
        changes = [change_24h, change_7d, change_30d]
        positive = sum(1 for c in changes if c and c > 0)
        
        if positive == 3:
            return "Strong Uptrend"
        elif positive == 2:
            return "Uptrend"
        elif positive == 1:
            return "Downtrend"
        elif positive == 0:
            return "Strong Downtrend"
        else:
            return "Mixed/Sideways"
    
    def _analyze_volatility(self, historical_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze price volatility."""
        if not historical_data or "prices" not in historical_data:
            return {"assessment": "Unable to calculate volatility"}
        
        prices = [p[1] for p in historical_data.get("prices", [])]
        
        if len(prices) < 2:
            return {"assessment": "Insufficient data"}
        
        # Calculate volatility metrics
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                price_changes.append(change)
        
        if not price_changes:
            return {"assessment": "Unable to calculate volatility"}
        
        std_dev = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
        avg_change = statistics.mean([abs(c) for c in price_changes])
        
        # Calculate high/low range
        high = max(prices)
        low = min(prices)
        current = prices[-1]
        range_pct = ((high - low) / low) * 100 if low > 0 else 0
        
        return {
            "standard_deviation": std_dev,
            "average_daily_change": avg_change,
            "price_range_percentage": range_pct,
            "high": high,
            "low": low,
            "assessment": self._assess_volatility(std_dev),
        }
    
    def _assess_volatility(self, std_dev: float) -> str:
        """Assess volatility level."""
        if std_dev > 10:
            return "Extremely High - High risk"
        elif std_dev > 5:
            return "High - Significant price swings"
        elif std_dev > 3:
            return "Moderate - Normal crypto volatility"
        elif std_dev > 1:
            return "Low - Relatively stable"
        else:
            return "Very Low - Minimal price movement"
    
    def _find_support_resistance(
        self, 
        historical_data: Optional[Dict[str, Any]],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        current_price = current_data.get("price_usd")
        
        if not historical_data or "prices" not in historical_data or not current_price:
            return {"support": None, "resistance": None, "note": "Insufficient data"}
        
        prices = [p[1] for p in historical_data.get("prices", [])]
        
        # Simple support/resistance based on recent highs and lows
        recent_high = max(prices)
        recent_low = min(prices)
        
        # Find resistance (recent high above current price)
        resistance = recent_high if recent_high > current_price else None
        
        # Find support (recent low below current price)
        support = recent_low if recent_low < current_price else None
        
        return {
            "support": support,
            "support_formatted": OutputFormatter.format_price(support) if support else "N/A",
            "resistance": resistance,
            "resistance_formatted": OutputFormatter.format_price(resistance) if resistance else "N/A",
            "current_position": self._assess_price_position(current_price, support, resistance),
        }
    
    def _assess_price_position(
        self, 
        current: float, 
        support: Optional[float], 
        resistance: Optional[float]
    ) -> str:
        """Assess current price position relative to support/resistance."""
        if not support or not resistance:
            return "Unable to assess position"
        
        range_size = resistance - support
        position_in_range = ((current - support) / range_size) * 100
        
        if position_in_range > 80:
            return "Near resistance - potential selling pressure"
        elif position_in_range > 60:
            return "Upper range - approaching resistance"
        elif position_in_range > 40:
            return "Mid-range - balanced position"
        elif position_in_range > 20:
            return "Lower range - approaching support"
        else:
            return "Near support - potential buying opportunity"
    
    def _analyze_ath_atl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all-time high and low."""
        ath = data.get("ath")
        atl = data.get("atl")
        current = data.get("price_usd")
        
        analysis = {
            "ath": ath,
            "atl": atl,
        }
        
        if ath and current:
            distance_from_ath = ((current - ath) / ath) * 100
            analysis["distance_from_ath_pct"] = distance_from_ath
            analysis["ath_assessment"] = self._assess_ath_distance(distance_from_ath)
        
        if atl and current:
            gain_from_atl = ((current - atl) / atl) * 100
            analysis["gain_from_atl_pct"] = gain_from_atl
        
        return analysis
    
    def _assess_ath_distance(self, distance_pct: float) -> str:
        """Assess distance from ATH."""
        if distance_pct > -10:
            return "Near ATH - potential resistance"
        elif distance_pct > -30:
            return "Moderate correction from ATH"
        elif distance_pct > -50:
            return "Significant correction from ATH"
        else:
            return "Deep correction from ATH - potential value opportunity"
    
    def _generate_summary(
        self, 
        current_data: Dict[str, Any], 
        trends: Dict[str, Any],
        volatility: Dict[str, Any]
    ) -> str:
        """Generate price analysis summary."""
        symbol = current_data.get("symbol", "").upper()
        price = current_data.get("price_usd")
        change_24h = current_data.get("price_change_24h", 0)
        change_7d = current_data.get("price_change_7d", 0)
        
        summary_parts = []
        
        # Current price
        price_str = OutputFormatter.format_price(price) if price else "N/A"
        change_24h_str = OutputFormatter.format_percentage(change_24h) if change_24h else "0%"
        
        direction = "up" if change_24h > 0 else "down"
        summary_parts.append(f"{symbol} is currently trading at {price_str}, {direction} {change_24h_str} in the last 24 hours")
        
        # 7-day trend
        if change_7d:
            change_7d_str = OutputFormatter.format_percentage(change_7d)
            summary_parts.append(f"Over the past 7 days, the price has changed {change_7d_str}")
        
        # Volatility
        vol_assessment = volatility.get("assessment", "")
        if vol_assessment:
            summary_parts.append(f"Volatility: {vol_assessment}")
        
        # Overall trend
        overall_trend = trends.get("overall", "")
        if overall_trend:
            summary_parts.append(f"Overall trend: {overall_trend}")
        
        return ". ".join(summary_parts) + "."
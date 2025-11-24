"""
Technical Analysis Module
Calculates and analyzes technical indicators (RSI, MACD, MA, etc.).
"""

from typing import Dict, Any, Optional, List
import logging
import pandas as pd
import pandas_ta as ta
from src.data_sources.coingecko_client import CoinGeckoClient

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Performs technical analysis on cryptocurrency tokens."""
    
    def __init__(self, coingecko_client: Optional[CoinGeckoClient] = None):
        """
        Initialize Technical Analyzer.
        
        Args:
            coingecko_client: CoinGecko API client
        """
        self.coingecko = coingecko_client
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Technical analysis results
        """
        try:
            # Get OHLC data
            ohlc_data = self._get_ohlc_data(symbol, days=90)
            
            if not ohlc_data:
                return {"error": "Unable to fetch OHLC data for technical analysis"}
            
            # Convert to DataFrame
            df = self._prepare_dataframe(ohlc_data)
            
            # Calculate indicators
            indicators = {
                "rsi": self._calculate_rsi(df),
                "macd": self._calculate_macd(df),
                "moving_averages": self._calculate_moving_averages(df),
                "bollinger_bands": self._calculate_bollinger_bands(df),
                "volume": self._analyze_volume(df),
            }
            
            # Generate signals
            signals = self._generate_signals(indicators)
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "indicators": indicators,
                "signals": signals,
                "summary": self._generate_summary(indicators, signals),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            raise
    
    def _get_ohlc_data(self, symbol: str, days: int = 90) -> Optional[List]:
        """Get OHLC (Open, High, Low, Close) data."""
        if not self.coingecko:
            return None
        
        try:
            return self.coingecko.get_ohlc(symbol, days=days)
        except Exception as e:
            logger.warning(f"OHLC data fetch failed: {e}")
            return None
    
    def _prepare_dataframe(self, ohlc_data: List) -> pd.DataFrame:
        """Convert OHLC data to pandas DataFrame."""
        df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        rsi = ta.rsi(df["close"], length=period)
        current_rsi = rsi.iloc[-1] if not rsi.empty else None
        
        if current_rsi is None:
            return {"value": None, "signal": "Unknown", "interpretation": "Unable to calculate RSI"}
        
        # Determine signal
        if current_rsi > 70:
            signal = "Overbought"
            interpretation = "RSI above 70 suggests overbought conditions - potential sell signal"
        elif current_rsi > 60:
            signal = "Slightly Overbought"
            interpretation = "RSI above 60 indicates bullish momentum with some overbought pressure"
        elif current_rsi > 40:
            signal = "Neutral"
            interpretation = "RSI in neutral zone - no clear directional signal"
        elif current_rsi > 30:
            signal = "Slightly Oversold"
            interpretation = "RSI below 40 indicates bearish pressure with some oversold conditions"
        else:
            signal = "Oversold"
            interpretation = "RSI below 30 suggests oversold conditions - potential buy signal"
        
        return {
            "value": round(current_rsi, 2),
            "signal": signal,
            "interpretation": interpretation,
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        macd = ta.macd(df["close"])
        
        if macd is None or macd.empty:
            return {"signal": "Unknown", "interpretation": "Unable to calculate MACD"}
        
        current_macd = macd["MACD_12_26_9"].iloc[-1]
        current_signal = macd["MACDs_12_26_9"].iloc[-1]
        current_histogram = macd["MACDh_12_26_9"].iloc[-1]
        
        # Determine signal
        if current_macd > current_signal and current_histogram > 0:
            signal = "Bullish"
            interpretation = "MACD above signal line with positive histogram - bullish momentum"
        elif current_macd < current_signal and current_histogram < 0:
            signal = "Bearish"
            interpretation = "MACD below signal line with negative histogram - bearish momentum"
        else:
            signal = "Neutral"
            interpretation = "MACD showing mixed signals - momentum unclear"
        
        return {
            "macd": round(current_macd, 2) if current_macd else None,
            "signal_line": round(current_signal, 2) if current_signal else None,
            "histogram": round(current_histogram, 2) if current_histogram else None,
            "signal": signal,
            "interpretation": interpretation,
        }
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Simple and Exponential Moving Averages."""
        current_price = df["close"].iloc[-1]
        
        # Calculate MAs
        sma_20 = ta.sma(df["close"], length=20).iloc[-1]
        sma_50 = ta.sma(df["close"], length=50).iloc[-1]
        ema_20 = ta.ema(df["close"], length=20).iloc[-1]
        ema_50 = ta.ema(df["close"], length=50).iloc[-1]
        
        # Determine trend
        if current_price > sma_20 and current_price > sma_50:
            trend = "Bullish"
            interpretation = "Price above both 20 and 50-day SMAs - uptrend confirmed"
        elif current_price < sma_20 and current_price < sma_50:
            trend = "Bearish"
            interpretation = "Price below both 20 and 50-day SMAs - downtrend confirmed"
        else:
            trend = "Mixed"
            interpretation = "Price between moving averages - trend unclear"
        
        return {
            "sma_20": round(sma_20, 2) if sma_20 else None,
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "ema_20": round(ema_20, 2) if ema_20 else None,
            "ema_50": round(ema_50, 2) if ema_50 else None,
            "current_price": round(current_price, 2),
            "trend": trend,
            "interpretation": interpretation,
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        bbands = ta.bbands(df["close"], length=period)
        
        if bbands is None or bbands.empty:
            return {"signal": "Unknown", "interpretation": "Unable to calculate Bollinger Bands"}
        
        current_price = df["close"].iloc[-1]
        upper_band = bbands[f"BBU_{period}_2.0"].iloc[-1]
        middle_band = bbands[f"BBM_{period}_2.0"].iloc[-1]
        lower_band = bbands[f"BBL_{period}_2.0"].iloc[-1]
        
        # Determine position
        band_width = upper_band - lower_band
        position_pct = ((current_price - lower_band) / band_width) * 100 if band_width > 0 else 50
        
        if position_pct > 100:
            signal = "Above Upper Band"
            interpretation = "Price above upper band - potentially overbought"
        elif position_pct > 80:
            signal = "Near Upper Band"
            interpretation = "Price near upper band - strong bullish momentum"
        elif position_pct > 20:
            signal = "Within Bands"
            interpretation = "Price within normal range"
        elif position_pct > 0:
            signal = "Near Lower Band"
            interpretation = "Price near lower band - strong bearish momentum"
        else:
            signal = "Below Lower Band"
            interpretation = "Price below lower band - potentially oversold"
        
        return {
            "upper_band": round(upper_band, 2) if upper_band else None,
            "middle_band": round(middle_band, 2) if middle_band else None,
            "lower_band": round(lower_band, 2) if lower_band else None,
            "current_price": round(current_price, 2),
            "position_percentage": round(position_pct, 2),
            "signal": signal,
            "interpretation": interpretation,
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends (if available)."""
        # CoinGecko OHLC doesn't include volume, so this is a placeholder
        # You could enhance this with market_chart data
        return {
            "note": "Volume data not available in current OHLC dataset",
            "interpretation": "Use market_chart endpoint for volume analysis",
        }
    
    def _generate_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on indicators."""
        signals = []
        
        # RSI signal
        rsi = indicators.get("rsi", {})
        if rsi.get("signal") == "Oversold":
            signals.append("Buy")
        elif rsi.get("signal") == "Overbought":
            signals.append("Sell")
        
        # MACD signal
        macd = indicators.get("macd", {})
        if macd.get("signal") == "Bullish":
            signals.append("Buy")
        elif macd.get("signal") == "Bearish":
            signals.append("Sell")
        
        # MA signal
        ma = indicators.get("moving_averages", {})
        if ma.get("trend") == "Bullish":
            signals.append("Buy")
        elif ma.get("trend") == "Bearish":
            signals.append("Sell")
        
        # Aggregate signals
        buy_count = signals.count("Buy")
        sell_count = signals.count("Sell")
        
        if buy_count > sell_count:
            overall = "Bullish"
            confidence = f"{(buy_count / len(signals)) * 100:.0f}%" if signals else "0%"
        elif sell_count > buy_count:
            overall = "Bearish"
            confidence = f"{(sell_count / len(signals)) * 100:.0f}%" if signals else "0%"
        else:
            overall = "Neutral"
            confidence = "50%"
        
        return {
            "overall": overall,
            "confidence": confidence,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "interpretation": f"{overall} outlook with {confidence} indicator agreement",
        }
    
    def _generate_summary(self, indicators: Dict[str, Any], signals: Dict[str, Any]) -> str:
        """Generate technical analysis summary."""
        rsi = indicators.get("rsi", {})
        macd = indicators.get("macd", {})
        ma = indicators.get("moving_averages", {})
        
        summary_parts = []
        
        # RSI
        if rsi.get("value"):
            summary_parts.append(f"RSI(14) is at {rsi['value']}, indicating {rsi['signal']} conditions")
        
        # MACD
        if macd.get("signal"):
            summary_parts.append(f"MACD shows {macd['signal']} momentum")
        
        # Moving Averages
        if ma.get("trend"):
            summary_parts.append(f"Moving averages indicate a {ma['trend']} trend")
        
        # Overall signal
        overall = signals.get("overall", "Neutral")
        confidence = signals.get("confidence", "0%")
        summary_parts.append(f"Overall technical outlook: {overall} ({confidence} indicator agreement)")
        
        return ". ".join(summary_parts) + "."
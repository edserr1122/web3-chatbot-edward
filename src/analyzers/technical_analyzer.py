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
    
    def __init__(
        self, 
        coingecko_client: Optional[CoinGeckoClient] = None,
        binance_clients: Optional[List[Any]] = None,  # Ordered Binance clients
        coincap_client: Optional[Any] = None  # CoinCapClient fallback
    ):
        """
        Initialize Technical Analyzer.
        
        Args:
            coingecko_client: CoinGecko API client (primary OHLC data source)
            binance_client: Binance API client (fallback for OHLC data - excellent quality)
            
        Note: CoinMarketCap is NOT used because historical OHLCV data 
              requires a paid subscription (Hobbyist or higher).
        """
        self.coingecko = coingecko_client
        self.binance_clients = binance_clients or []
        self.coincap = coincap_client
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Technical analysis results
        """
        try:
            logger.info(f"🔍 [TechnicalAnalyzer] Starting analysis for {symbol}")
            
            # Get OHLC data with source tracking
            ohlc_data, data_source = self._get_ohlc_data(symbol, days=90)
            
            # Minimum 20 points needed for RSI (14-period) + some buffer
            if not ohlc_data or len(ohlc_data) < 20:
                logger.error(f"   ❌ [TechnicalAnalyzer] Insufficient OHLC data for {symbol}")
                return {
                    "error": "Unable to fetch sufficient OHLC data for technical analysis",
                    "note": f"Technical analysis requires at least 20 data points (got {len(ohlc_data) if ohlc_data else 0})",
                    "limitation": "All data sources unavailable - may be due to API rate limits, network issues, or token not listed on exchanges"
                }
            
            # Log data source
            logger.info(f"📊 [TechnicalAnalyzer] Data sources used: {data_source} (OHLC data - {len(ohlc_data)} points)")
            
            # Convert to DataFrame
            df = self._prepare_dataframe(ohlc_data)
            
            # Additional safety check (20 minimum for RSI-14 + buffer)
            if df.empty or len(df) < 20:
                logger.warning(f"⚠️  [TechnicalAnalyzer] Insufficient DataFrame data for {symbol}")
                return {
                    "error": "Insufficient OHLC data for technical analysis",
                    "note": f"Technical analysis requires at least 20 data points (got {len(df)})",
                    "limitation": "CoinGecko free tier may limit OHLC data availability"
                }
            
            # Log if we have limited data
            if len(df) < 50:
                logger.warning(f"⚠️  [TechnicalAnalyzer] Limited data ({len(df)} points) - Some indicators may be less reliable")
            
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
            
            # Add data limitation note if applicable
            data_note = None
            if len(df) < 50:
                data_note = f"Analysis based on {len(df)} data points ({data_source if data_source else 'limited source'}). Some indicators may have reduced accuracy."
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "data_points": len(df),
                "data_source": data_source if data_source else "unknown",
                "indicators": indicators,
                "signals": signals,
                "summary": self._generate_summary(indicators, signals),
            }
            
            if data_note:
                analysis["data_limitation_note"] = data_note
            
            logger.info(f"✅ [TechnicalAnalyzer] Analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ [TechnicalAnalyzer] Analysis failed for {symbol}: {e}")
            raise
    
    def _get_ohlc_data(self, symbol: str, days: int = 90) -> tuple[Optional[List], Optional[str]]:
        """
        Get OHLC (Open, High, Low, Close) data with fallback support.
        
        Priority: Binance (Global → US) → CoinGecko → CoinCap TA
        
        Note: CoinMarketCap historical OHLCV requires paid subscription.
        
        Requires at least 20 data points (minimum for RSI-14 calculation).
        
        Returns:
            tuple: (ohlc_data, source_name) where source_name is the API that provided data
        """
        ohlc_data = None
        data_source = None
        
        # Priority 1: Binance clients (global first -> US)
        if self.binance_clients:
            for binance_client in self.binance_clients:
                binance_name = binance_client.__class__.__name__.replace("Client", "")
                try:
                    logger.info(f"   🔄 [{binance_name}] Using fallback for OHLC data ({symbol})")
                    klines = binance_client.get_klines(symbol, interval="1d", limit=min(days, 1000))
                    
                    # Convert Binance klines to CoinGecko-compatible format
                    ohlc_data = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in klines]
                    
                    if ohlc_data and len(ohlc_data) >= 20:
                        data_source = binance_name
                        logger.info(f"   ✅ [{binance_name}] Returned {len(ohlc_data)} OHLC points (excellent quality)")
                        return ohlc_data, data_source
                    else:
                        logger.warning(f"   ⚠️  [{binance_name}] Insufficient OHLC data ({len(ohlc_data) if ohlc_data else 0} points)")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️  [{binance_name}] OHLC fetch failed: {e}")
        
        # Priority 2: CoinGecko
        if not ohlc_data and self.coingecko:
            try:
                logger.info(f"   📊 [CoinGecko] Fetching OHLC from CoinGecko")
                ohlc_data = self.coingecko.get_ohlc(symbol, days=days)
                
                if ohlc_data and len(ohlc_data) >= 20:
                    data_source = "CoinGecko"
                    if len(ohlc_data) >= 50:
                        logger.info(f"   ✅ [CoinGecko] Returned {len(ohlc_data)} OHLC points (optimal)")
                    else:
                        logger.info(f"   ✅ [CoinGecko] Returned {len(ohlc_data)} OHLC points (limited but usable)")
                        logger.warning(f"   💡 CoinGecko free tier limits: Some indicators may have reduced accuracy")
                    return ohlc_data, data_source
                elif ohlc_data:
                    logger.warning(f"   ⚠️  [CoinGecko] Returned only {len(ohlc_data)} OHLC points (need 20+ minimum)")
                else:
                    logger.warning(f"   ⚠️  [CoinGecko] Returned no OHLC data")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  [CoinGecko] OHLC fetch failed: {e}")
        
        # Priority 3: CoinCap TA candlesticks
        if not ohlc_data and self.coincap:
            try:
                logger.info(f"   🔄 [CoinCap] Using TA candlesticks for OHLC data ({symbol})")
                ohlc_data = self.coincap.get_candlesticks(symbol, interval="d1", limit=min(days, 200))
                
                if ohlc_data and len(ohlc_data) >= 20:
                    data_source = "CoinCap"
                    logger.info(f"   ✅ [CoinCap] Returned {len(ohlc_data)} OHLC points (TA endpoint)")
                    return ohlc_data, data_source
                else:
                    logger.warning(f"   ⚠️  [CoinCap] Insufficient candlestick data ({len(ohlc_data) if ohlc_data else 0} points)")
            except Exception as e:
                logger.warning(f"   ⚠️  [CoinCap] Candlestick fetch failed: {e}")
        
        # No data source worked
        logger.error(f"   ❌ [TechnicalAnalyzer] No OHLC data available from any source for {symbol}")
        return None, None
    
    def _prepare_dataframe(self, ohlc_data: List) -> pd.DataFrame:
        """Convert OHLC data to pandas DataFrame."""
        df = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        rsi = ta.rsi(df["close"], length=period)
        current_rsi = rsi.iloc[-1] if rsi is not None and not rsi.empty else None
        
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
        
        # Additional safety check for None values in columns
        try:
            current_macd = macd["MACD_12_26_9"].iloc[-1]
            current_signal = macd["MACDs_12_26_9"].iloc[-1]
            current_histogram = macd["MACDh_12_26_9"].iloc[-1]
        except (KeyError, IndexError, AttributeError):
            return {"signal": "Unknown", "interpretation": "Unable to calculate MACD"}
        
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
        try:
            current_price = df["close"].iloc[-1]
            data_points = len(df)
            
            # Calculate MAs based on available data
            # Only calculate if we have enough data points
            sma_20_series = ta.sma(df["close"], length=min(20, data_points)) if data_points >= 20 else None
            sma_50_series = ta.sma(df["close"], length=min(50, data_points)) if data_points >= 50 else None
            ema_20_series = ta.ema(df["close"], length=min(20, data_points)) if data_points >= 20 else None
            ema_50_series = ta.ema(df["close"], length=min(50, data_points)) if data_points >= 50 else None
            
            sma_20 = sma_20_series.iloc[-1] if sma_20_series is not None and not sma_20_series.empty else None
            sma_50 = sma_50_series.iloc[-1] if sma_50_series is not None and not sma_50_series.empty else None
            ema_20 = ema_20_series.iloc[-1] if ema_20_series is not None and not ema_20_series.empty else None
            ema_50 = ema_50_series.iloc[-1] if ema_50_series is not None and not ema_50_series.empty else None
        except (IndexError, KeyError, AttributeError):
            return {
                "signal": "Unknown",
                "interpretation": "Unable to calculate moving averages - insufficient data"
            }
        
        # Determine trend based on available MAs
        if sma_20 and sma_50:
            # Full analysis with both MAs
            if current_price > sma_20 and current_price > sma_50:
                trend = "Bullish"
                interpretation = "Price above both 20 and 50-day SMAs - uptrend confirmed"
            elif current_price < sma_20 and current_price < sma_50:
                trend = "Bearish"
                interpretation = "Price below both 20 and 50-day SMAs - downtrend confirmed"
            else:
                trend = "Mixed"
                interpretation = "Price between moving averages - trend unclear"
        elif sma_20:
            # Limited data - use only 20-day MA
            if current_price > sma_20:
                trend = "Bullish"
                interpretation = f"Price above 20-day SMA - short-term uptrend (limited to {data_points} data points)"
            else:
                trend = "Bearish"
                interpretation = f"Price below 20-day SMA - short-term downtrend (limited to {data_points} data points)"
        else:
            trend = "Unknown"
            interpretation = f"Insufficient data for moving averages (only {data_points} points available)"
        
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
        
        try:
            current_price = df["close"].iloc[-1]
            upper_band = bbands[f"BBU_{period}_2.0"].iloc[-1]
            middle_band = bbands[f"BBM_{period}_2.0"].iloc[-1]
            lower_band = bbands[f"BBL_{period}_2.0"].iloc[-1]
        except (KeyError, IndexError, AttributeError):
            return {"signal": "Unknown", "interpretation": "Unable to calculate Bollinger Bands"}
        
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
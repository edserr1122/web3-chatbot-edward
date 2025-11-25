"""
Comparative Analysis Module
Compares multiple cryptocurrency tokens across various dimensions.
"""

from typing import Dict, Any, List, Optional
import logging
import time
from src.analyzers.fundamental_analyzer import FundamentalAnalyzer
from src.analyzers.price_analyzer import PriceAnalyzer
from src.analyzers.technical_analyzer import TechnicalAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.coinmarketcap_client import CoinMarketCapClient
from src.utils.formatters import OutputFormatter

logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """Performs comparative analysis between multiple cryptocurrency tokens."""
    
    def __init__(
        self,
        fundamental_analyzer: Optional[FundamentalAnalyzer] = None,
        price_analyzer: Optional[PriceAnalyzer] = None,
        technical_analyzer: Optional[TechnicalAnalyzer] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        coingecko_client: Optional[CoinGeckoClient] = None,
        coinmarketcap_client: Optional[CoinMarketCapClient] = None,
        coincap_client: Optional[Any] = None,  # CoinCapClient - fallback for historical data
        binance_clients: Optional[List[Any]] = None  # Ordered Binance clients (global first)
    ):
        """
        Initialize Comparative Analyzer.
        
        Args:
            fundamental_analyzer: Fundamental analyzer instance
            price_analyzer: Price analyzer instance
            technical_analyzer: Technical analyzer instance
            sentiment_analyzer: Sentiment analyzer instance
            coingecko_client: CoinGecko client for optimized batch fetching
            coinmarketcap_client: CoinMarketCap client for optimized batch fetching
            coincap_client: CoinCap client for historical data fallback (free, no API key)
            binance_client: Binance client for historical data fallback (free, no API key)
        """
        self.fundamental = fundamental_analyzer
        self.price = price_analyzer
        self.technical = technical_analyzer
        self.sentiment = sentiment_analyzer
        self.coingecko = coingecko_client
        self.coinmarketcap = coinmarketcap_client
        self.coincap = coincap_client
        self.binance_clients = binance_clients or []
    
    def compare(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Perform comparative analysis of multiple tokens.
        
        ⚡ OPTIMIZED: Uses batch API calls when possible to reduce API usage.
        
        Args:
            symbols: List of token symbols (e.g., ["BTC", "ETH"])
            
        Returns:
            dict: Comparative analysis results
        """
        if len(symbols) < 2:
            raise ValueError("At least 2 tokens required for comparison")
        
        try:
            logger.info(f"🔍 [ComparativeAnalyzer] Starting comparative analysis for {', '.join(symbols)}")
            
            # OPTIMIZATION: Try to use batch API calls
            batch_data_cg = None
            batch_data_cmc = None
            
            # Try CoinGecko batch fetch
            if self.coingecko:
                try:
                    logger.info(f"⚡ [ComparativeAnalyzer] Using optimized batch fetch from CoinGecko")
                    batch_data_cg = self.coingecko.get_coins_markets(symbols)
                    logger.info(f"✅ [ComparativeAnalyzer] Fetched {len(batch_data_cg)} tokens from CoinGecko in ONE API call")
                except Exception as e:
                    logger.warning(f"CoinGecko batch fetch failed: {e}")
            
            # Try CoinMarketCap batch fetch
            if self.coinmarketcap:
                try:
                    logger.info(f"⚡ [ComparativeAnalyzer] Using optimized batch fetch from CoinMarketCap")
                    batch_data_cmc = self.coinmarketcap.get_quotes_latest(symbols)
                    logger.info(f"✅ [ComparativeAnalyzer] Fetched {len(batch_data_cmc)} tokens from CoinMarketCap in ONE API call")
                except Exception as e:
                    logger.warning(f"CoinMarketCap batch fetch failed: {e}")
            
            # Gather data for all tokens
            tokens_data = {}
            for symbol in symbols:
                tokens_data[symbol] = self._gather_token_data(symbol, batch_data_cg, batch_data_cmc)
            
            logger.info(f"📊 [ComparativeAnalyzer] Analyzing {len(symbols)} tokens across all dimensions")
            
            # Perform comparisons
            comparison = {
                "tokens": symbols,
                "fundamental_comparison": self._compare_fundamentals(tokens_data),
                "price_comparison": self._compare_prices(tokens_data),
                "technical_comparison": self._compare_technicals(tokens_data),
                "sentiment_comparison": self._compare_sentiments(tokens_data),
                "summary": self._generate_summary(symbols, tokens_data),
            }
            
            logger.info(f"✅ [ComparativeAnalyzer] Comparative analysis completed for {', '.join(symbols)}")
            return comparison
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise
    
    def _gather_token_data(
        self, 
        symbol: str, 
        batch_data_cg: Optional[List[Dict]] = None,
        batch_data_cmc: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gather all analysis data for a single token.
        
        Args:
            symbol: Token symbol
            batch_data_cg: Pre-fetched batch data from CoinGecko (optimization)
            batch_data_cmc: Pre-fetched batch data from CoinMarketCap (optimization)
        
        Returns:
            dict: Complete token analysis data
        """
        data = {"symbol": symbol}
        
        # Extract pre-fetched CoinGecko data for this symbol if available
        prefetched_cg = None
        if batch_data_cg:
            for token in batch_data_cg:
                if token.get("symbol", "").upper() == symbol.upper():
                    prefetched_cg = token
                    logger.info(f"   ♻️  Using pre-fetched CoinGecko data for {symbol}")
                    break
        
        # Extract pre-fetched CoinMarketCap data for this symbol if available
        prefetched_cmc = None
        if batch_data_cmc:
            if symbol.upper() in batch_data_cmc:
                prefetched_cmc = batch_data_cmc[symbol.upper()]
                logger.info(f"   ♻️  Using pre-fetched CoinMarketCap data for {symbol}")
        
        # Store pre-fetched data for analyzers to use (future optimization)
        data["_prefetched_cg"] = prefetched_cg
        data["_prefetched_cmc"] = prefetched_cmc
        
        # Fundamental data
        if self.fundamental:
            try:
                data["fundamental"] = self.fundamental.analyze(symbol)
            except Exception as e:
                logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
                data["fundamental"] = None
        
        # Price data - can use prefetched data to avoid duplicate calls
        if self.price:
            try:
                data["price"] = self.price.analyze(symbol)
            except Exception as e:
                logger.warning(f"Price analysis failed for {symbol}: {e}")
                data["price"] = None
        
        # Technical data
        if self.technical:
            try:
                data["technical"] = self.technical.analyze(symbol)
            except Exception as e:
                logger.warning(f"Technical analysis failed for {symbol}: {e}")
                data["technical"] = None
        
        # Sentiment data
        if self.sentiment:
            try:
                data["sentiment"] = self.sentiment.analyze(symbol)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
                data["sentiment"] = None
        
        return data
    
    def _compare_fundamentals(self, tokens_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare fundamental metrics across tokens."""
        comparison = {}
        
        for symbol, data in tokens_data.items():
            fundamental = data.get("fundamental")
            if not fundamental:
                comparison[symbol] = {"available": False}
                continue
            
            market_metrics = fundamental.get("market_metrics", {})
            supply_metrics = fundamental.get("supply_metrics", {})
            liquidity_metrics = fundamental.get("liquidity_metrics", {})
            
            comparison[symbol] = {
                "available": True,
                "market_cap": market_metrics.get("market_cap"),
                "market_cap_rank": market_metrics.get("market_cap_rank"),
                "volume_24h": market_metrics.get("volume_24h"),
                "liquidity_rating": liquidity_metrics.get("liquidity_rating"),
                "supply_model": supply_metrics.get("supply_model"),
            }
        
        # Add rankings
        comparison["rankings"] = self._rank_by_metric(comparison, "market_cap")
        
        return comparison
    
    def _compare_prices(self, tokens_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare price metrics across tokens."""
        comparison = {}
        
        for symbol, data in tokens_data.items():
            price_data = data.get("price")
            if not price_data:
                comparison[symbol] = {"available": False}
                continue
            
            price_changes = price_data.get("price_changes", {})
            trends = price_data.get("trends", {})
            volatility = price_data.get("volatility", {})
            
            comparison[symbol] = {
                "available": True,
                "current_price": price_data.get("current_price"),
                "change_24h": price_changes.get("24h"),
                "change_7d": price_changes.get("7d"),
                "change_30d": price_changes.get("30d"),
                "trend_24h": trends.get("24h", {}).get("direction"),
                "trend_7d": trends.get("7d", {}).get("direction"),
                "volatility_rating": volatility.get("assessment"),
            }
        
        # Identify best/worst performers
        comparison["best_performer_24h"] = self._find_best_performer(comparison, "change_24h")
        comparison["worst_performer_24h"] = self._find_worst_performer(comparison, "change_24h")
        comparison["best_performer_7d"] = self._find_best_performer(comparison, "change_7d")
        
        return comparison
    
    def _compare_technicals(self, tokens_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare technical indicators across tokens."""
        comparison = {}
        
        for symbol, data in tokens_data.items():
            technical = data.get("technical")
            if not technical:
                comparison[symbol] = {"available": False}
                continue
            
            indicators = technical.get("indicators", {})
            signals = technical.get("signals", {})
            
            rsi = indicators.get("rsi", {})
            macd = indicators.get("macd", {})
            ma = indicators.get("moving_averages", {})
            
            comparison[symbol] = {
                "available": True,
                "rsi_value": rsi.get("value"),
                "rsi_signal": rsi.get("signal"),
                "macd_signal": macd.get("signal"),
                "ma_trend": ma.get("trend"),
                "overall_signal": signals.get("overall"),
                "signal_confidence": signals.get("confidence"),
            }
        
        return comparison
    
    def _compare_sentiments(self, tokens_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare sentiment metrics across tokens."""
        comparison = {}
        
        for symbol, data in tokens_data.items():
            sentiment = data.get("sentiment")
            if not sentiment:
                comparison[symbol] = {"available": False}
                continue
            
            overall = sentiment.get("overall_sentiment", {})
            social = sentiment.get("social_sentiment", {})
            news = sentiment.get("news_sentiment", {})
            
            comparison[symbol] = {
                "available": True,
                "overall_score": overall.get("score"),
                "overall_label": overall.get("label"),
                "social_score": social.get("galaxy_score") if social.get("available") else None,
                "news_sentiment": news.get("overall_sentiment") if news.get("available") else None,
                "news_positive_pct": news.get("positive_percentage") if news.get("available") else None,
            }
        
        # Identify most/least positive sentiment
        comparison["most_positive"] = self._find_best_performer(comparison, "overall_score")
        comparison["least_positive"] = self._find_worst_performer(comparison, "overall_score")
        
        return comparison
    
    def _rank_by_metric(
        self, 
        comparison: Dict[str, Dict], 
        metric: str,
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Rank tokens by a specific metric."""
        rankings = []
        
        for symbol, data in comparison.items():
            if symbol == "rankings" or not data.get("available"):
                continue
            
            value = data.get(metric)
            if value is not None:
                rankings.append({"symbol": symbol, "value": value})
        
        # Sort
        rankings.sort(key=lambda x: x["value"], reverse=reverse)
        
        # Add rank numbers
        for i, item in enumerate(rankings):
            item["rank"] = i + 1
        
        return rankings
    
    def _find_best_performer(
        self, 
        comparison: Dict[str, Dict], 
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """Find the best performing token for a metric."""
        best = None
        best_value = float('-inf')
        
        for symbol, data in comparison.items():
            if isinstance(data, dict) and data.get("available"):
                value = data.get(metric)
                if value is not None and value > best_value:
                    best_value = value
                    best = {"symbol": symbol, "value": value}
        
        return best
    
    def _find_worst_performer(
        self, 
        comparison: Dict[str, Dict], 
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """Find the worst performing token for a metric."""
        worst = None
        worst_value = float('inf')
        
        for symbol, data in comparison.items():
            if isinstance(data, dict) and data.get("available"):
                value = data.get(metric)
                if value is not None and value < worst_value:
                    worst_value = value
                    worst = {"symbol": symbol, "value": value}
        
        return worst
    
    def _generate_summary(
        self, 
        symbols: List[str], 
        tokens_data: Dict[str, Dict]
    ) -> str:
        """Generate comparative analysis summary."""
        summary_parts = []
        
        # Intro
        symbols_str = " vs ".join(symbols)
        summary_parts.append(f"Comparing {symbols_str}")
        
        # Market cap comparison
        market_caps = {}
        for symbol, data in tokens_data.items():
            fundamental = data.get("fundamental")
            if fundamental:
                mc = fundamental.get("market_metrics", {}).get("market_cap")
                if mc:
                    market_caps[symbol] = mc
        
        if market_caps:
            largest = max(market_caps, key=market_caps.get)
            summary_parts.append(
                f"{largest} has the largest market cap at "
                f"{OutputFormatter.format_large_number(market_caps[largest])}"
            )
        
        # Price performance
        price_changes = {}
        for symbol, data in tokens_data.items():
            price_data = data.get("price")
            if price_data:
                change = price_data.get("price_changes", {}).get("24h")
                if change:
                    price_changes[symbol] = change
        
        if price_changes:
            best = max(price_changes, key=price_changes.get)
            worst = min(price_changes, key=price_changes.get)
            summary_parts.append(
                f"In the last 24h, {best} performed best "
                f"({OutputFormatter.format_percentage(price_changes[best])}) "
                f"while {worst} performed worst "
                f"({OutputFormatter.format_percentage(price_changes[worst])})"
            )
        
        # Sentiment comparison
        sentiments = {}
        for symbol, data in tokens_data.items():
            sentiment_data = data.get("sentiment")
            if sentiment_data:
                score = sentiment_data.get("overall_sentiment", {}).get("score")
                if score:
                    sentiments[symbol] = score
        
        if sentiments:
            most_positive = max(sentiments, key=sentiments.get)
            summary_parts.append(
                f"{most_positive} has the most positive sentiment "
                f"(score: {sentiments[most_positive]:.1f}/100)"
            )
        
        return ". ".join(summary_parts) + "."
    
    def compare_temporal(self, symbols: List[str], days_ago: int) -> Dict[str, Any]:
        """
        Compare current token data against historical data from N days ago.
        
        Args:
            symbols: List of token symbols to compare
            days_ago: Number of days in the past to compare against
            
        Returns:
            dict: Temporal comparison results with changes over time
        """
        if days_ago <= 0:
            raise ValueError("days_ago must be greater than 0 for temporal comparison")
        
        try:
            logger.info(f"🔍 [ComparativeAnalyzer] Temporal comparison: {', '.join(symbols)} (current vs {days_ago} days ago)")
            
            temporal_changes = {}
            
            for idx, symbol in enumerate(symbols):
                # Add small delay between tokens to avoid rate limits (except for first token)
                if idx > 0:
                    time.sleep(2)  # 2-second delay between tokens
                    logger.info(f"   ⏱️  Waiting 2s between tokens to avoid rate limits...")
                
                logger.info(f"   📊 Analyzing temporal changes for {symbol}")
                
                try:
                    # Get current data
                    current_price_data = None
                    if self.coinmarketcap:
                        try:
                            current_price_data = self.coinmarketcap.get_token_data(symbol)
                        except Exception as e:
                            logger.warning(f"CMC current data fetch failed for {symbol}: {e}")
                    
                    if not current_price_data and self.coingecko:
                        try:
                            current_price_data = self.coingecko.get_token_data(symbol)
                        except Exception as e:
                            logger.warning(f"CoinGecko current data fetch failed for {symbol}: {e}")
                    
                    # Get historical data with multi-level fallback (CoinGecko → CoinCap → Binance)
                    historical_data = None
                    data_source = None
                    
                    # Try CoinGecko first (best option)
                    if self.coingecko:
                        try:
                            historical_data = self.coingecko.get_market_chart(symbol, days=days_ago + 1)
                            data_source = "CoinGecko"
                            logger.info(f"   ✅ [CoinGecko] Successfully fetched historical data for {symbol}")
                        except Exception as e:
                            logger.warning(f"   ⚠️  [CoinGecko] Historical data fetch failed for {symbol}: {e}")
                    
                    # Fallback 1: CoinCap
                    if not historical_data and self.coincap:
                        try:
                            logger.info(f"   🔄 [CoinCap] Using fallback for historical data ({symbol})")
                            historical_data = self.coincap.get_market_chart(symbol, days=days_ago + 1)
                            data_source = "CoinCap"
                            logger.info(f"   ✅ [CoinCap] Successfully fetched historical data for {symbol}")
                        except Exception as e:
                            logger.warning(f"   ⚠️  [CoinCap] Historical data fetch failed for {symbol}: {e}")
                    
                    # Fallback 2: Binance clients (global first, then US)
                    if not historical_data and self.binance_clients:
                        for binance_client in self.binance_clients:
                            binance_name = binance_client.__class__.__name__.replace("Client", "")
                            try:
                                logger.info(f"   🔄 [{binance_name}] Using fallback for historical data ({symbol})")
                                historical_data = binance_client.get_market_chart(symbol, days=days_ago + 1)
                                data_source = binance_name
                                logger.info(f"   ✅ [{binance_name}] Successfully fetched historical data for {symbol}")
                                break
                            except Exception as e:
                                logger.warning(f"   ⚠️  [{binance_name}] Historical data fetch failed for {symbol}: {e}")
                    
                    # Log final data source used
                    if historical_data and data_source:
                        logger.info(f"   📍 Using {data_source} data for {symbol}")
                    
                    if not current_price_data:
                        temporal_changes[symbol] = {
                            "available": False,
                            "error": "Unable to fetch current price data"
                        }
                        continue
                    
                    if not historical_data or not historical_data.get("prices"):
                        temporal_changes[symbol] = {
                            "available": False,
                            "error": "Unable to fetch historical data"
                        }
                        continue
                    
                    # Extract data points
                    current_price = current_price_data.get("price_usd", 0)
                    current_market_cap = current_price_data.get("market_cap", 0)  # ✅ Correct key
                    current_volume_24h = current_price_data.get("total_volume", 0)  # ✅ Correct key
                    
                    # Get historical price from N days ago (first data point)
                    historical_prices = historical_data.get("prices", [])
                    historical_market_caps = historical_data.get("market_caps", [])
                    historical_volumes = historical_data.get("total_volumes", [])
                    
                    if not historical_prices:
                        temporal_changes[symbol] = {
                            "available": False,
                            "error": "Insufficient historical data"
                        }
                        continue
                    
                    # Use first data point as "N days ago"
                    old_price = historical_prices[0][1] if historical_prices else 0
                    old_market_cap = historical_market_caps[0][1] if historical_market_caps else 0
                    old_volume = historical_volumes[0][1] if historical_volumes else 0
                    
                    # Debug logging
                    logger.info(f"   📊 {symbol} Current: Price=${current_price:,.2f}, MC=${current_market_cap:,.0f}, Vol=${current_volume_24h:,.0f}")
                    logger.info(f"   📊 {symbol} {days_ago}d ago: Price=${old_price:,.2f}, MC=${old_market_cap:,.0f}, Vol=${old_volume:,.0f}")
                    
                    # Validate data
                    if current_price == 0 or old_price == 0:
                        logger.warning(f"   ⚠️  {symbol}: Missing price data (current: ${current_price}, old: ${old_price})")
                    if current_market_cap == 0 or old_market_cap == 0:
                        logger.warning(f"   ⚠️  {symbol}: Missing market cap data (current: ${current_market_cap:,.0f}, old: ${old_market_cap:,.0f})")
                    if current_volume_24h == 0 or old_volume == 0:
                        logger.warning(f"   ⚠️  {symbol}: Missing volume data (current: ${current_volume_24h:,.0f}, old: ${old_volume:,.0f})")
                    
                    # Calculate changes
                    price_change_pct = ((current_price - old_price) / old_price * 100) if old_price > 0 else 0
                    market_cap_change_pct = ((current_market_cap - old_market_cap) / old_market_cap * 100) if old_market_cap > 0 else 0
                    volume_change_pct = ((current_volume_24h - old_volume) / old_volume * 100) if old_volume > 0 else 0
                    
                    # Build result with data availability notes
                    result = {
                        "available": True,
                        "current_price": OutputFormatter.format_price(current_price),
                        "old_price": OutputFormatter.format_price(old_price),
                        "price_change": OutputFormatter.format_percentage(price_change_pct) if old_price > 0 else "N/A",
                        "price_change_raw": price_change_pct,
                        "current_market_cap": OutputFormatter.format_large_number(current_market_cap),
                        "old_market_cap": OutputFormatter.format_large_number(old_market_cap),
                        "market_cap_change": OutputFormatter.format_percentage(market_cap_change_pct) if old_market_cap > 0 else "N/A",
                        "market_cap_change_raw": market_cap_change_pct,
                        "current_volume_24h": OutputFormatter.format_large_number(current_volume_24h),
                        "old_volume_24h": OutputFormatter.format_large_number(old_volume),
                        "volume_change": OutputFormatter.format_percentage(volume_change_pct) if old_volume > 0 else "N/A",
                        "volume_change_raw": volume_change_pct,
                        "days_compared": days_ago
                    }
                    
                    # Add data quality notes
                    notes = []
                    if old_market_cap == 0:
                        notes.append("Historical market cap data unavailable")
                    if old_volume == 0:
                        notes.append("Historical volume data unavailable")
                    if notes:
                        result["data_notes"] = "; ".join(notes)
                    
                    temporal_changes[symbol] = result
                    
                    logger.info(f"   ✅ {symbol}: Price {price_change_pct:+.2f}%, Market Cap {market_cap_change_pct:+.2f}%, Volume {volume_change_pct:+.2f}%")
                
                except Exception as e:
                    logger.error(f"   ❌ Temporal comparison failed for {symbol}: {e}")
                    temporal_changes[symbol] = {
                        "available": False,
                        "error": str(e)
                    }
            
            # Generate summary
            summary = self._generate_temporal_summary(temporal_changes, days_ago)
            
            # Check if all comparisons failed
            successful_count = sum(1 for data in temporal_changes.values() if data.get("available"))
            if successful_count == 0:
                # All failed - likely due to rate limiting
                error_messages = [data.get("error", "") for data in temporal_changes.values() if data.get("error")]
                if any("rate limiting" in msg.lower() or "temporarily unavailable" in msg.lower() for msg in error_messages):
                    summary = (
                        f"⚠️  Temporal comparison temporarily unavailable: CoinGecko API is rate-limited. "
                        f"Historical price data requires CoinGecko's market chart API, which is currently in cooldown. "
                        f"Please wait 1-2 minutes and try again, or try a current-state comparison instead."
                    )
                else:
                    summary = f"Unable to retrieve historical data for the requested {days_ago}-day period. " + summary
            
            return {
                "type": "temporal_comparison",
                "symbols": symbols,
                "days_ago": days_ago,
                "temporal_changes": temporal_changes,
                "summary": summary,
                "successful_count": successful_count,
                "total_count": len(symbols)
            }
        
        except Exception as e:
            logger.error(f"Temporal comparison failed: {e}")
            raise
    
    def _generate_temporal_summary(self, temporal_changes: Dict[str, Dict], days_ago: int) -> str:
        """Generate summary for temporal comparison."""
        summary_parts = []
        
        # Collect all successful comparisons
        successful = {sym: data for sym, data in temporal_changes.items() if data.get("available")}
        
        if not successful:
            return f"Unable to generate temporal comparison for the requested {days_ago}-day period due to data availability issues."
        
        # Find biggest gainers/losers
        price_changes = {sym: data.get("price_change_raw", 0) for sym, data in successful.items()}
        
        if price_changes:
            best = max(price_changes, key=price_changes.get)
            worst = min(price_changes, key=price_changes.get)
            
            if len(successful) == 1:
                symbol = list(successful.keys())[0]
                change = price_changes[symbol]
                direction = "gained" if change > 0 else "lost"
                summary_parts.append(
                    f"Over the past {days_ago} days, {symbol} has {direction} {abs(change):.2f}% in value"
                )
            else:
                summary_parts.append(
                    f"Over the past {days_ago} days, {best} gained the most "
                    f"({price_changes[best]:+.2f}%) while {worst} performed worst "
                    f"({price_changes[worst]:+.2f}%)"
                )
        
        # Market cap changes
        mc_changes = {sym: data.get("market_cap_change_raw", 0) for sym, data in successful.items()}
        if mc_changes:
            avg_mc_change = sum(mc_changes.values()) / len(mc_changes)
            trend = "grown" if avg_mc_change > 0 else "declined"
            summary_parts.append(f"Market capitalization has {trend} by an average of {avg_mc_change:+.2f}%")
        
        return ". ".join(summary_parts) + "."
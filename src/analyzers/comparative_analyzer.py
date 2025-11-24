"""
Comparative Analysis Module
Compares multiple cryptocurrency tokens across various dimensions.
"""

from typing import Dict, Any, List, Optional
import logging
from src.analyzers.fundamental_analyzer import FundamentalAnalyzer
from src.analyzers.price_analyzer import PriceAnalyzer
from src.analyzers.technical_analyzer import TechnicalAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.utils.formatters import OutputFormatter

logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """Performs comparative analysis between multiple cryptocurrency tokens."""
    
    def __init__(
        self,
        fundamental_analyzer: Optional[FundamentalAnalyzer] = None,
        price_analyzer: Optional[PriceAnalyzer] = None,
        technical_analyzer: Optional[TechnicalAnalyzer] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        """
        Initialize Comparative Analyzer.
        
        Args:
            fundamental_analyzer: Fundamental analyzer instance
            price_analyzer: Price analyzer instance
            technical_analyzer: Technical analyzer instance
            sentiment_analyzer: Sentiment analyzer instance
        """
        self.fundamental = fundamental_analyzer
        self.price = price_analyzer
        self.technical = technical_analyzer
        self.sentiment = sentiment_analyzer
    
    def compare(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Perform comparative analysis of multiple tokens.
        
        Args:
            symbols: List of token symbols (e.g., ["BTC", "ETH"])
            
        Returns:
            dict: Comparative analysis results
        """
        if len(symbols) < 2:
            raise ValueError("At least 2 tokens required for comparison")
        
        try:
            # Gather data for all tokens
            tokens_data = {}
            for symbol in symbols:
                tokens_data[symbol] = self._gather_token_data(symbol)
            
            # Perform comparisons
            comparison = {
                "tokens": symbols,
                "fundamental_comparison": self._compare_fundamentals(tokens_data),
                "price_comparison": self._compare_prices(tokens_data),
                "technical_comparison": self._compare_technicals(tokens_data),
                "sentiment_comparison": self._compare_sentiments(tokens_data),
                "summary": self._generate_summary(symbols, tokens_data),
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise
    
    def _gather_token_data(self, symbol: str) -> Dict[str, Any]:
        """Gather all analysis data for a single token."""
        data = {"symbol": symbol}
        
        # Fundamental data
        if self.fundamental:
            try:
                data["fundamental"] = self.fundamental.analyze(symbol)
            except Exception as e:
                logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
                data["fundamental"] = None
        
        # Price data
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
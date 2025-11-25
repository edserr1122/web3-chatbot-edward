"""
Sentiment Analysis Module
Analyzes news sentiment, community engagement, and Fear & Greed Index.
"""

from typing import Dict, Any, Optional
import logging
from src.data_sources.cryptopanic_client import CryptoPanicClient
from src.data_sources.fear_greed_client import FearGreedClient
from src.data_sources.coingecko_client import CoinGeckoClient

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Performs sentiment analysis on cryptocurrency tokens."""
    
    def __init__(
        self,
        cryptopanic_client: Optional[CryptoPanicClient] = None,
        fear_greed_client: Optional[FearGreedClient] = None,
        coingecko_client: Optional[CoinGeckoClient] = None
    ):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            cryptopanic_client: CryptoPanic API client (for news sentiment)
            fear_greed_client: Fear & Greed Index client (for market mood + global context)
            coingecko_client: CoinGecko API client (for community data)
        """
        self.cryptopanic = cryptopanic_client
        self.fear_greed = fear_greed_client
        self.coingecko = coingecko_client
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            logger.info(f"🔍 [SentimentAnalyzer] Starting analysis for {symbol}")
            
            # Gather sentiment data from available sources
            news_sentiment = self._get_news_sentiment(symbol)
            fear_greed = self._get_fear_greed_index()
            community_data = self._get_community_data(symbol)
            market_context = self._get_market_context()  # NEW: Global market context
            
            # Log data source results
            sources_used = []
            if news_sentiment.get("available"): sources_used.append("CryptoPanic")
            if fear_greed.get("available"): sources_used.append("Fear & Greed Index")
            if community_data.get("available"): sources_used.append("CoinGecko Community")
            if market_context.get("available"): sources_used.append("Alternative.me Global")
            logger.info(f"📊 [SentimentAnalyzer] Data sources used: {', '.join(sources_used) if sources_used else 'None'}")
            
            # Aggregate sentiment
            overall_sentiment = self._aggregate_sentiment(
                news_sentiment,
                fear_greed
            )
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "news_sentiment": news_sentiment,
                "fear_greed_index": fear_greed,
                "community_data": community_data,
                "market_context": market_context,  # NEW: Add market context
                "overall_sentiment": overall_sentiment,
                "summary": self._generate_summary(
                    symbol,
                    news_sentiment,
                    fear_greed,
                    overall_sentiment
                ),
            }
            
            logger.info(f"✅ [SentimentAnalyzer] Analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ [SentimentAnalyzer] Analysis failed for {symbol}: {e}")
            raise
    
    def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get news sentiment from CryptoPanic.
        
        ⚠️ Note: Messari news feed is not available in free tier.
        """
        cryptopanic_data = None
        
        # Try CryptoPanic
        if self.cryptopanic:
            try:
                cryptopanic_data = self.cryptopanic.get_token_data(symbol)
            except Exception as e:
                logger.warning(f"CryptoPanic sentiment fetch failed: {e}")
        
        # If no source available
        if not cryptopanic_data:
            return {"available": False, "note": "News data not available"}
        
        # Process CryptoPanic data
        result = {"available": True}
        
        if cryptopanic_data:
            sentiment_dist = cryptopanic_data.get("sentiment_distribution", {})
            total = cryptopanic_data.get("news_count", 0)
            
            positive_pct = sentiment_dist.get("positive_pct", 0)
            negative_pct = sentiment_dist.get("negative_pct", 0)
            neutral_pct = sentiment_dist.get("neutral_pct", 0)
            
            # Get panic score (v2 API feature)
            avg_panic_score = cryptopanic_data.get("average_panic_score")
            
            # Get detailed votes (v2 API feature)
            total_votes = cryptopanic_data.get("total_votes", {})
            
            # Determine overall news sentiment
            if positive_pct > 60:
                overall = "Positive"
            elif positive_pct > 40:
                overall = "Mixed/Neutral"
            else:
                overall = "Negative"
            
            # Interpret panic score
            panic_interpretation = self._interpret_panic_score(avg_panic_score)
            
            result.update({
                "total_articles": total,
                "positive_count": sentiment_dist.get("positive", 0),
                "negative_count": sentiment_dist.get("negative", 0),
                "neutral_count": sentiment_dist.get("neutral", 0),
                "positive_percentage": positive_pct,
                "negative_percentage": negative_pct,
                "neutral_percentage": neutral_pct,
                "overall_sentiment": overall,
                
                # v2 API: Panic Score (market impact)
                "panic_score": avg_panic_score,
                "panic_interpretation": panic_interpretation,
                
                # v2 API: Detailed community votes
                "community_votes": {
                    "positive": total_votes.get("positive", 0),
                    "negative": total_votes.get("negative", 0),
                    "important": total_votes.get("important", 0),
                    "liked": total_votes.get("liked", 0),
                    "disliked": total_votes.get("disliked", 0),
                    "lol": total_votes.get("lol", 0),
                    "toxic": total_votes.get("toxic", 0),
                    "saved": total_votes.get("saved", 0),
                    "comments": total_votes.get("comments", 0),
                },
                
                "recent_headlines": [
                    {
                        "title": article.get("title"),
                        "published_at": article.get("published_at"),
                        "source": "CryptoPanic",
                        "url": article.get("url"),
                        "panic_score": article.get("panic_score"),
                        "votes": article.get("votes", {}),
                    }
                    for article in cryptopanic_data.get("recent_news", [])[:3]
                ],
                "interpretation": self._interpret_news_sentiment(positive_pct, negative_pct, avg_panic_score),
            })
        
        return result
    
    def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get market-wide Fear & Greed Index."""
        if not self.fear_greed:
            return {"available": False, "note": "Fear & Greed Index not available"}
        
        try:
            data = self.fear_greed.get_current()
            
            return {
                "available": True,
                "value": data.get("value"),
                "classification": data.get("classification"),
                "interpretation": data.get("interpretation"),
                "timestamp": data.get("timestamp"),
            }
            
        except Exception as e:
            logger.warning(f"Fear & Greed Index fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _get_community_data(self, symbol: str) -> Dict[str, Any]:
        """Get community engagement data from CoinGecko."""
        if not self.coingecko:
            return {"available": False, "note": "Community data not available"}
        
        try:
            data = self.coingecko.get_token_data(symbol)
            community = data.get("community_data", {})
            developer = data.get("developer_data", {})
            
            return {
                "available": True,
                "twitter_followers": community.get("twitter_followers"),
                "reddit_subscribers": community.get("reddit_subscribers"),
                "reddit_active_accounts": community.get("reddit_accounts_active_48h"),
                "github_stars": developer.get("stars"),
                "github_forks": developer.get("forks"),
                "interpretation": self._interpret_community_data(community),
            }
            
        except Exception as e:
            logger.warning(f"Community data fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _get_market_context(self) -> Dict[str, Any]:
        """
        Get global market context from Alternative.me.
        
        ⚡ Provides useful context like BTC dominance and total market cap.
        Helps understand if sentiment is token-specific or market-wide.
        """
        if not self.fear_greed:
            return {"available": False, "note": "Global market context not available"}
        
        try:
            global_data = self.fear_greed.get_global_data()
            
            return {
                "available": True,
                "total_market_cap": global_data.get("total_market_cap"),
                "total_volume_24h": global_data.get("total_volume_24h"),
                "bitcoin_dominance": global_data.get("bitcoin_dominance"),
                "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
                "active_markets": global_data.get("active_markets"),
                "interpretation": self._interpret_market_context(global_data.get("bitcoin_dominance")),
            }
            
        except Exception as e:
            logger.warning(f"Global market context fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _interpret_market_context(self, btc_dominance: Optional[float]) -> str:
        """Interpret market context based on BTC dominance."""
        if not btc_dominance:
            return "Unable to assess market context"
        
        if btc_dominance > 60:
            return "Very high BTC dominance - Market is risk-off, favoring Bitcoin"
        elif btc_dominance > 50:
            return "High BTC dominance - Conservative market sentiment"
        elif btc_dominance > 40:
            return "Moderate BTC dominance - Balanced market"
        else:
            return "Low BTC dominance - Altcoin season, risk-on sentiment"
    
    def _interpret_news_sentiment(
        self, 
        positive_pct: float, 
        negative_pct: float,
        panic_score: Optional[float] = None
    ) -> str:
        """Interpret news sentiment with optional panic score."""
        base_interpretation = ""
        
        if positive_pct > 70:
            base_interpretation = "Overwhelmingly positive news coverage"
        elif positive_pct > 55:
            base_interpretation = "Mostly positive news coverage"
        elif positive_pct > 45:
            base_interpretation = "Balanced news coverage"
        elif positive_pct > 30:
            base_interpretation = "Mostly negative news coverage"
        else:
            base_interpretation = "Predominantly negative news coverage"
        
        # Add panic score context if available
        if panic_score is not None:
            if panic_score > 70:
                base_interpretation += ". Very high market impact - major news event"
            elif panic_score > 50:
                base_interpretation += ". Significant market impact"
            elif panic_score > 30:
                base_interpretation += ". Moderate market attention"
        
        return base_interpretation
    
    def _interpret_panic_score(self, panic_score: Optional[float]) -> str:
        """
        Interpret CryptoPanic's panic score (0-100).
        Higher scores indicate greater market impact and importance.
        """
        if panic_score is None:
            return "No panic score data available"
        
        if panic_score >= 80:
            return "Extremely high market impact - Breaking news with massive attention"
        elif panic_score >= 60:
            return "High market impact - Significant news driving discussion"
        elif panic_score >= 40:
            return "Moderate market impact - Notable news with community interest"
        elif panic_score >= 20:
            return "Low market impact - Minor news or limited attention"
        else:
            return "Very low market impact - Minimal community engagement"
    
    def _interpret_community_data(self, community: Dict[str, Any]) -> str:
        """Interpret community engagement."""
        twitter = community.get("twitter_followers", 0)
        reddit = community.get("reddit_subscribers", 0)
        
        total_community = twitter + reddit
        
        if total_community > 1_000_000:
            return "Very large and active community"
        elif total_community > 100_000:
            return "Large community with strong engagement"
        elif total_community > 10_000:
            return "Moderate community size"
        else:
            return "Small community or limited social presence"
    
    def _aggregate_sentiment(
        self,
        news: Dict[str, Any],
        fear_greed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate sentiment from all sources."""
        sentiment_scores = []
        
        # News sentiment
        if news.get("available"):
            positive_pct = news.get("positive_percentage", 50)
            sentiment_scores.append(positive_pct)
        
        # Fear & Greed (market-wide, but relevant)
        if fear_greed.get("available"):
            fg_value = fear_greed.get("value", 50)
            sentiment_scores.append(fg_value)
        
        # Calculate average
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
        else:
            avg_score = 50  # Neutral default
        
        # Determine overall sentiment
        if avg_score > 65:
            overall = "Positive"
            confidence = "High"
        elif avg_score > 55:
            overall = "Slightly Positive"
            confidence = "Medium"
        elif avg_score > 45:
            overall = "Neutral"
            confidence = "Medium"
        elif avg_score > 35:
            overall = "Slightly Negative"
            confidence = "Medium"
        else:
            overall = "Negative"
            confidence = "High"
        
        return {
            "score": round(avg_score, 2),
            "label": overall,
            "confidence": confidence,
            "sources_count": len(sentiment_scores),
            "interpretation": self._interpret_overall_sentiment(avg_score),
        }
    
    def _interpret_overall_sentiment(self, score: float) -> str:
        """Interpret overall sentiment score."""
        if score > 70:
            return "Market sentiment is very positive - strong bullish indicators"
        elif score > 60:
            return "Market sentiment is positive - generally bullish outlook"
        elif score > 50:
            return "Market sentiment is slightly positive - cautiously optimistic"
        elif score > 40:
            return "Market sentiment is slightly negative - some concerns present"
        elif score > 30:
            return "Market sentiment is negative - bearish indicators"
        else:
            return "Market sentiment is very negative - strong bearish indicators"
    
    def _generate_summary(
        self,
        symbol: str,
        news: Dict[str, Any],
        fear_greed: Dict[str, Any],
        overall: Dict[str, Any]
    ) -> str:
        """Generate sentiment analysis summary."""
        summary_parts = []
        
        # Overall sentiment
        overall_label = overall.get("label", "Unknown")
        overall_score = overall.get("score", 50)
        summary_parts.append(
            f"Overall market sentiment for {symbol.upper()} is {overall_label} "
            f"(score: {overall_score}/100)"
        )
        
        # News sentiment
        if news.get("available"):
            news_sentiment = news.get("overall_sentiment", "Unknown")
            positive_pct = news.get("positive_percentage", 0)
            summary_parts.append(
                f"News coverage is {news_sentiment.lower()} with {positive_pct:.0f}% positive articles"
            )
        
        # Fear & Greed
        if fear_greed.get("available"):
            fg_value = fear_greed.get("value")
            fg_class = fear_greed.get("classification", "")
            summary_parts.append(
                f"Market-wide Fear & Greed Index is at {fg_value} ({fg_class})"
            )
        
        return ". ".join(summary_parts) + "."
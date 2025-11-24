"""
Sentiment Analysis Module
Analyzes social media sentiment, news sentiment, community engagement, and Fear & Greed Index.
"""

from typing import Dict, Any, Optional
import logging
from src.data_sources.lunarcrush_client import LunarCrushClient
from src.data_sources.cryptopanic_client import CryptoPanicClient
from src.data_sources.messari_client import MessariClient
from src.data_sources.fear_greed_client import FearGreedClient
from src.data_sources.coingecko_client import CoinGeckoClient

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Performs sentiment analysis on cryptocurrency tokens."""
    
    def __init__(
        self,
        lunarcrush_client: Optional[LunarCrushClient] = None,
        cryptopanic_client: Optional[CryptoPanicClient] = None,
        fear_greed_client: Optional[FearGreedClient] = None,
        coingecko_client: Optional[CoinGeckoClient] = None,
        messari_client: Optional[MessariClient] = None
    ):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            lunarcrush_client: LunarCrush API client
            cryptopanic_client: CryptoPanic API client
            fear_greed_client: Fear & Greed Index client
            coingecko_client: CoinGecko API client
            messari_client: Messari API client
        """
        self.lunarcrush = lunarcrush_client
        self.cryptopanic = cryptopanic_client
        self.fear_greed = fear_greed_client
        self.coingecko = coingecko_client
        self.messari = messari_client
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Gather sentiment data from available sources
            social_sentiment = self._get_social_sentiment(symbol)
            news_sentiment = self._get_news_sentiment(symbol)
            fear_greed = self._get_fear_greed_index()
            community_data = self._get_community_data(symbol)
            
            # Aggregate sentiment
            overall_sentiment = self._aggregate_sentiment(
                social_sentiment,
                news_sentiment,
                fear_greed
            )
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "social_sentiment": social_sentiment,
                "news_sentiment": news_sentiment,
                "fear_greed_index": fear_greed,
                "community_data": community_data,
                "overall_sentiment": overall_sentiment,
                "summary": self._generate_summary(
                    symbol,
                    social_sentiment,
                    news_sentiment,
                    fear_greed,
                    overall_sentiment
                ),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            raise
    
    def _get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment from LunarCrush."""
        if not self.lunarcrush:
            return {"available": False, "note": "LunarCrush data not available"}
        
        try:
            data = self.lunarcrush.get_token_data(symbol)
            
            galaxy_score = data.get("galaxy_score")
            alt_rank = data.get("alt_rank")
            sentiment = data.get("sentiment")
            social_volume = data.get("social_volume")
            social_dominance = data.get("social_dominance")
            
            # Interpret sentiment
            sentiment_label = "Neutral"
            if sentiment and sentiment > 3.5:
                sentiment_label = "Positive"
            elif sentiment and sentiment < 2.5:
                sentiment_label = "Negative"
            
            return {
                "available": True,
                "galaxy_score": galaxy_score,  # 0-100 scale
                "alt_rank": alt_rank,
                "sentiment_score": sentiment,  # 1-5 scale
                "sentiment_label": sentiment_label,
                "social_volume": social_volume,
                "social_dominance": social_dominance,
                "interpretation": self._interpret_social_sentiment(galaxy_score, sentiment),
            }
            
        except Exception as e:
            logger.warning(f"LunarCrush sentiment fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment from CryptoPanic."""
        if not self.cryptopanic:
            return {"available": False, "note": "CryptoPanic data not available"}
        
        try:
            data = self.cryptopanic.get_token_data(symbol)
            
            sentiment_dist = data.get("sentiment_distribution", {})
            total = data.get("news_count", 0)
            
            positive_pct = sentiment_dist.get("positive_pct", 0)
            negative_pct = sentiment_dist.get("negative_pct", 0)
            
            # Determine overall news sentiment
            if positive_pct > 60:
                overall = "Positive"
            elif positive_pct > 40:
                overall = "Mixed/Neutral"
            else:
                overall = "Negative"
            
            return {
                "available": True,
                "total_articles": total,
                "positive_count": sentiment_dist.get("positive", 0),
                "negative_count": sentiment_dist.get("negative", 0),
                "neutral_count": sentiment_dist.get("neutral", 0),
                "positive_percentage": positive_pct,
                "negative_percentage": negative_pct,
                "overall_sentiment": overall,
                "recent_headlines": [
                    {
                        "title": article.get("title"),
                        "published_at": article.get("published_at"),
                        "votes": article.get("votes", {}),
                    }
                    for article in data.get("recent_news", [])[:3]
                ],
                "interpretation": self._interpret_news_sentiment(positive_pct, negative_pct),
            }
            
        except Exception as e:
            logger.warning(f"CryptoPanic sentiment fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
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
    
    def _interpret_social_sentiment(
        self, 
        galaxy_score: Optional[float], 
        sentiment: Optional[float]
    ) -> str:
        """Interpret LunarCrush social sentiment."""
        if not galaxy_score or not sentiment:
            return "Unable to assess social sentiment"
        
        if galaxy_score > 75 and sentiment > 3.5:
            return "Very strong social engagement with positive sentiment"
        elif galaxy_score > 60 and sentiment > 3:
            return "Strong social presence with generally positive sentiment"
        elif galaxy_score > 40:
            return "Moderate social engagement with neutral sentiment"
        else:
            return "Low social engagement or negative sentiment"
    
    def _interpret_news_sentiment(
        self, 
        positive_pct: float, 
        negative_pct: float
    ) -> str:
        """Interpret news sentiment."""
        if positive_pct > 70:
            return "Overwhelmingly positive news coverage"
        elif positive_pct > 55:
            return "Mostly positive news coverage"
        elif positive_pct > 45:
            return "Balanced news coverage"
        elif positive_pct > 30:
            return "Mostly negative news coverage"
        else:
            return "Predominantly negative news coverage"
    
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
        social: Dict[str, Any],
        news: Dict[str, Any],
        fear_greed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate sentiment from all sources."""
        sentiment_scores = []
        
        # Social sentiment (convert to 0-100 scale)
        if social.get("available"):
            sentiment_score = social.get("sentiment_score", 3)
            # Convert 1-5 scale to 0-100
            normalized = ((sentiment_score - 1) / 4) * 100
            sentiment_scores.append(normalized)
        
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
        social: Dict[str, Any],
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
        
        # Social sentiment
        if social.get("available"):
            galaxy_score = social.get("galaxy_score")
            if galaxy_score:
                summary_parts.append(f"Social metrics show a Galaxy Score of {galaxy_score}/100")
        
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
"""
CryptoPanic API client for crypto news and sentiment.
Documentation: https://cryptopanic.com/developers/api/
API v2 Reference: https://cryptopanic.com/developers/api/
"""

from typing import Dict, Any, List, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class CryptoPanicClient(BaseAPIClient):
    """Client for CryptoPanic API v2 (Developer Plan)."""
    
    def __init__(self, api_key: str):
        """
        Initialize CryptoPanic client.
        
        Args:
            api_key: CryptoPanic API key (auth_token)
        """
        # Using v2 API with developer plan path
        super().__init__(api_key=api_key, base_url="https://cryptopanic.com/api/developer/v2/")
        self.min_request_interval = 0.5  # Developer plan: 2 req/sec
        
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get news and sentiment for a token.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: News and sentiment data with v2 API structure
        """
        # Get general news
        news = self.get_news(currencies=symbol)
        
        # Calculate sentiment distribution based on votes
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_votes = {
            "positive": 0, 
            "negative": 0, 
            "important": 0,
            "liked": 0,
            "disliked": 0,
            "lol": 0,
            "toxic": 0,
            "saved": 0,
            "comments": 0,
        }
        panic_scores = []
        
        for article in news:
            votes = article.get("votes", {})
            
            # Aggregate all vote types
            total_votes["positive"] += votes.get("positive", 0)
            total_votes["negative"] += votes.get("negative", 0)
            total_votes["important"] += votes.get("important", 0)
            total_votes["liked"] += votes.get("liked", 0)
            total_votes["disliked"] += votes.get("disliked", 0)
            total_votes["lol"] += votes.get("lol", 0)
            total_votes["toxic"] += votes.get("toxic", 0)
            total_votes["saved"] += votes.get("saved", 0)
            total_votes["comments"] += votes.get("comments", 0)
            
            # Categorize sentiment based on votes
            pos = votes.get("positive", 0) + votes.get("liked", 0)
            neg = votes.get("negative", 0) + votes.get("disliked", 0) + votes.get("toxic", 0)
            
            if pos > neg:
                sentiment_counts["positive"] += 1
            elif neg > pos:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
            
            # Collect panic scores
            panic_score = article.get("panic_score")
            if panic_score is not None:
                panic_scores.append(panic_score)
        
        total_articles = len(news)
        avg_panic_score = sum(panic_scores) / len(panic_scores) if panic_scores else None
        
        return {
            "symbol": symbol.upper(),
            "news_count": total_articles,
            "sentiment_distribution": {
                "positive": sentiment_counts["positive"],
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"],
                "positive_pct": (sentiment_counts["positive"] / total_articles * 100) if total_articles > 0 else 0,
                "negative_pct": (sentiment_counts["negative"] / total_articles * 100) if total_articles > 0 else 0,
                "neutral_pct": (sentiment_counts["neutral"] / total_articles * 100) if total_articles > 0 else 0,
            },
            "total_votes": total_votes,
            "average_panic_score": avg_panic_score,
            "recent_news": news[:5],  # Top 5 recent articles
        }
    
    def get_news(
        self, 
        currencies: Optional[str] = None, 
        kind: str = "news",
        filter_type: Optional[str] = None,
        public: bool = True,
        regions: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Get crypto news articles.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            kind: Type of posts ("news", "media", or "all")
            filter_type: Optional filter ("rising", "hot", "bullish", "bearish", "important", "saved", "lol")
            public: Use public mode (recommended for apps)
            regions: Language code (e.g., "en", "fr", "es")
            
        Returns:
            list: List of news articles with metadata
        """
        endpoint = "posts/"
        params = {
            "auth_token": self.api_key,
            "kind": kind,
            "public": "true" if public else "false",
            "regions": regions,
        }
        
        if currencies:
            params["currencies"] = currencies.upper()
        
        if filter_type:
            params["filter"] = filter_type
        
        response = self._make_request(endpoint, params)
        
        # v2 API returns paginated results
        results = response.get("results", [])
        
        # Parse and enrich results
        enriched_results = []
        for item in results:
            enriched_results.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "description": item.get("description"),
                "published_at": item.get("published_at"),
                "created_at": item.get("created_at"),
                "kind": item.get("kind"),
                "url": item.get("url"),
                "original_url": item.get("original_url"),
                "image": item.get("image"),
                "source": item.get("source", {}),
                "votes": item.get("votes", {}),
                "panic_score": item.get("panic_score"),
                "panic_score_1h": item.get("panic_score_1h"),
                "author": item.get("author"),
                "instruments": item.get("instruments", []),  # Mentioned currencies
            })
        
        return enriched_results
    
    def get_bullish_news(self, currencies: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get bullish news for specified currencies.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            
        Returns:
            list: Bullish news articles
        """
        return self.get_news(currencies=currencies, filter_type="bullish")
    
    def get_bearish_news(self, currencies: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get bearish news for specified currencies.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            
        Returns:
            list: Bearish news articles
        """
        return self.get_news(currencies=currencies, filter_type="bearish")
    
    def get_important_news(self, currencies: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get important news for specified currencies.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            
        Returns:
            list: Important news articles
        """
        return self.get_news(currencies=currencies, filter_type="important")
    
    def get_rising_news(self, currencies: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trending/rising news for specified currencies.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            
        Returns:
            list: Rising/trending news articles
        """
        return self.get_news(currencies=currencies, filter_type="rising")
    
    def test_connection(self) -> bool:
        """
        Test connection to CryptoPanic API v2.
        
        Returns:
            bool: True if connection successful
        """
        try:
            endpoint = "posts/"
            params = {
                "auth_token": self.api_key,
                "kind": "news",
                "public": "true",
                "regions": "en",
                "per_page": 1,
            }
            response = self._make_request(
                endpoint,
                params,
                force_refresh=True,
                use_cache=False,
            )
            return bool(response.get("results"))
        except Exception as e:
            logger.error(f"CryptoPanic connection test failed: {e}")
            return False
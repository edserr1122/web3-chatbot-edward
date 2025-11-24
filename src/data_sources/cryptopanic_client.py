"""
CryptoPanic API client for crypto news and sentiment.
Documentation: https://cryptopanic.com/developers/api/
"""

from typing import Dict, Any, List, Optional
from .base_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class CryptoPanicClient(BaseAPIClient):
    """Client for CryptoPanic API."""
    
    def __init__(self, api_key: str):
        """
        Initialize CryptoPanic client.
        
        Args:
            api_key: CryptoPanic API key (required)
        """
        super().__init__(api_key=api_key, base_url="https://cryptopanic.com/api/v1/")
        self.min_request_interval = 2.0  # Free tier: 500 requests per day
        
    def get_token_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get news and sentiment for a token.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: News and sentiment data
        """
        news = self.get_news(currencies=symbol, limit=20)
        
        # Calculate sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_votes = {"positive": 0, "negative": 0, "important": 0}
        
        for article in news:
            votes = article.get("votes", {})
            total_votes["positive"] += votes.get("positive", 0)
            total_votes["negative"] += votes.get("negative", 0)
            total_votes["important"] += votes.get("important", 0)
            
            # Categorize based on votes
            pos = votes.get("positive", 0)
            neg = votes.get("negative", 0)
            
            if pos > neg:
                sentiment_counts["positive"] += 1
            elif neg > pos:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        
        total_articles = len(news)
        
        return {
            "symbol": symbol.upper(),
            "news_count": total_articles,
            "sentiment_distribution": {
                "positive": sentiment_counts["positive"],
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"],
                "positive_pct": (sentiment_counts["positive"] / total_articles * 100) if total_articles > 0 else 0,
                "negative_pct": (sentiment_counts["negative"] / total_articles * 100) if total_articles > 0 else 0,
            },
            "total_votes": total_votes,
            "recent_news": news[:5],  # Top 5 recent articles
        }
    
    def get_news(
        self, 
        currencies: Optional[str] = None, 
        kind: str = "news",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get crypto news articles.
        
        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            kind: Type of posts ("news", "media", or "all")
            limit: Number of results (max 50 for free tier)
            
        Returns:
            list: List of news articles
        """
        endpoint = "posts/"
        params = {
            "auth_token": self.api_key,
            "kind": kind,
        }
        
        if currencies:
            params["currencies"] = currencies.upper()
        
        if limit:
            params["limit"] = min(limit, 50)  # Free tier max
        
        response = self._make_request(endpoint, params)
        return response.get("results", [])
    
    def test_connection(self) -> bool:
        """
        Test connection to CryptoPanic API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.get_news(limit=1)
            return True
        except Exception as e:
            logger.error(f"CryptoPanic connection test failed: {e}")
            return False
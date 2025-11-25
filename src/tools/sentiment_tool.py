"""
Sentiment analysis tool for LangGraph agent.
"""

from typing import Dict, Any
import logging
from langchain_core.tools import tool
from src.analyzers import SentimentAnalyzer
from src.memory import cache_manager

logger = logging.getLogger(__name__)


class SentimentTool:
    """Sentiment analysis tool."""
    
    def __init__(self, analyzer: SentimentAnalyzer):
        """
        Initialize sentiment tool.
        
        Args:
            analyzer: SentimentAnalyzer instance
        """
        self.analyzer = analyzer
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis with caching.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Analysis results
        """
        cache_key = cache_manager.make_key("sentiment", symbol.upper())
        
        def fetch():
            logger.info(f"Performing sentiment analysis for {symbol}")
            return self.analyzer.analyze(symbol)
        
        return cache_manager.get_or_set(cache_key, fetch)


def create_sentiment_tool(analyzer: SentimentAnalyzer):
    """
    Create sentiment analysis tool for LangGraph.
    
    Args:
        analyzer: SentimentAnalyzer instance
        
    Returns:
        LangChain tool
    """
    tool_instance = SentimentTool(analyzer)
    
    @tool
    def analyze_sentiment(symbol: str) -> str:
        """
        Analyze market sentiment from social media, news, and community engagement.
        Use this for questions about sentiment, social metrics, Fear & Greed, or market mood.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        """
        try:
            result = tool_instance.analyze(symbol)
            summary = result.get("summary", "")
            overall_sentiment = result.get("overall_sentiment", {})
            social_sentiment = result.get("social_sentiment", {})
            news_sentiment = result.get("news_sentiment", {})
            fear_greed = result.get("fear_greed_index", {})
            
            response = f"**Sentiment Analysis for {symbol.upper()}:**\n\n"
            response += f"{summary}\n\n"
            
            # Overall sentiment
            response += f"**Overall Sentiment:** {overall_sentiment.get('label', 'Unknown')} "
            response += f"(Score: {overall_sentiment.get('score', 0):.1f}/100)\n"
            response += f"  {overall_sentiment.get('interpretation', '')}\n\n"
            
            # Social sentiment
            if social_sentiment.get("available"):
                response += "**Social Metrics (LunarCrush):**\n"
                response += f"- Galaxy Score: {social_sentiment.get('galaxy_score', 'N/A')}/100\n"
                response += f"- Sentiment: {social_sentiment.get('sentiment_label', 'N/A')}\n"
                response += f"- Social Volume: {social_sentiment.get('social_volume', 'N/A')}\n\n"
            
            # News sentiment
            if news_sentiment.get("available"):
                response += "**News Sentiment (CryptoPanic):**\n"
                response += f"- Overall: {news_sentiment.get('overall_sentiment', 'Unknown')}\n"
                response += f"- Positive: {news_sentiment.get('positive_percentage', 0):.1f}%\n"
                response += f"- Negative: {news_sentiment.get('negative_percentage', 0):.1f}%\n\n"
            
            # Fear & Greed
            if fear_greed.get("available"):
                response += "**Fear & Greed Index (Market-wide):**\n"
                response += f"- Value: {fear_greed.get('value', 'N/A')}/100\n"
                response += f"- Classification: {fear_greed.get('classification', 'Unknown')}\n"
                response += f"  {fear_greed.get('interpretation', '')}\n"
            
            return response
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return f"Error analyzing sentiment for {symbol}: {str(e)}"
    
    return analyze_sentiment


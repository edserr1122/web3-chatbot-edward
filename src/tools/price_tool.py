"""
Price analysis tool for LangGraph agent.
"""

from typing import Dict, Any
import logging
from langchain_core.tools import tool
from src.analyzers.price_analyzer import PriceAnalyzer
from src.memory.cache_manager import cache_manager

logger = logging.getLogger(__name__)


class PriceTool:
    """Price analysis tool."""
    
    def __init__(self, analyzer: PriceAnalyzer):
        """
        Initialize price tool.
        
        Args:
            analyzer: PriceAnalyzer instance
        """
        self.analyzer = analyzer
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform price analysis with caching.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Analysis results
        """
        cache_key = cache_manager.make_key("price", symbol.upper())
        
        def fetch():
            logger.info(f"Performing price analysis for {symbol}")
            return self.analyzer.analyze(symbol)
        
        return cache_manager.get_or_set(cache_key, fetch)


def create_price_tool(analyzer: PriceAnalyzer):
    """
    Create price analysis tool for LangGraph.
    
    Args:
        analyzer: PriceAnalyzer instance
        
    Returns:
        LangChain tool
    """
    tool_instance = PriceTool(analyzer)
    
    @tool
    def analyze_price(symbol: str) -> str:
        """
        Analyze price trends, volatility, and historical performance of a cryptocurrency.
        Use this for questions about price movements, trends, volatility, or support/resistance.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        """
        try:
            result = tool_instance.analyze(symbol)
            summary = result.get("summary", "")
            price_changes = result.get("price_changes", {})
            trends = result.get("trends", {})
            volatility = result.get("volatility", {})
            support_resistance = result.get("support_resistance", {})
            
            response = f"**Price Analysis for {symbol.upper()}:**\n\n"
            response += f"{summary}\n\n"
            
            response += "**Price Changes:**\n"
            if price_changes.get("24h"):
                response += f"- 24h: {price_changes['24h']:+.2f}%\n"
            if price_changes.get("7d"):
                response += f"- 7d: {price_changes['7d']:+.2f}%\n"
            if price_changes.get("30d"):
                response += f"- 30d: {price_changes['30d']:+.2f}%\n"
            
            response += f"\n**Trend:** {trends.get('overall', 'Unknown')}\n"
            response += f"**Volatility:** {volatility.get('assessment', 'Unknown')}\n\n"
            
            if support_resistance.get("support") or support_resistance.get("resistance"):
                response += "**Support/Resistance:**\n"
                response += f"- Support: {support_resistance.get('support_formatted', 'N/A')}\n"
                response += f"- Resistance: {support_resistance.get('resistance_formatted', 'N/A')}\n"
                response += f"- Position: {support_resistance.get('current_position', 'N/A')}\n"
            
            return response
        except Exception as e:
            logger.error(f"Price analysis error for {symbol}: {e}")
            return f"Error analyzing price for {symbol}: {str(e)}"
    
    return analyze_price


"""
Fundamental analysis tool for LangGraph agent.
"""

from typing import Dict, Any
import logging
from langchain_core.tools import tool
from src.analyzers import FundamentalAnalyzer
from src.memory import cache_manager

logger = logging.getLogger(__name__)


class FundamentalTool:
    """Fundamental analysis tool."""
    
    def __init__(self, analyzer: FundamentalAnalyzer):
        """
        Initialize fundamental tool.
        
        Args:
            analyzer: FundamentalAnalyzer instance
        """
        self.analyzer = analyzer
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform fundamental analysis with caching.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Analysis results
        """
        cache_key = cache_manager.make_key("fundamental", symbol.upper())
        
        def fetch():
            logger.info(f"Performing fundamental analysis for {symbol}")
            return self.analyzer.analyze(symbol)
        
        return cache_manager.get_or_set(cache_key, fetch)


def create_fundamental_tool(analyzer: FundamentalAnalyzer):
    """
    Create fundamental analysis tool for LangGraph.
    
    Args:
        analyzer: FundamentalAnalyzer instance
        
    Returns:
        LangChain tool
    """
    tool_instance = FundamentalTool(analyzer)
    
    @tool
    def analyze_fundamentals(symbol: str) -> str:
        """
        Analyze fundamental metrics of a cryptocurrency token including market cap, 
        supply, volume, and tokenomics. Use this for questions about market size, 
        circulating supply, or fundamental value.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        """
        try:
            result = tool_instance.analyze(symbol)
            summary = result.get("summary", "")
            market_metrics = result.get("market_metrics", {})
            supply_metrics = result.get("supply_metrics", {})
            liquidity_metrics = result.get("liquidity_metrics", {})
            
            response = f"**Fundamental Analysis for {symbol.upper()}:**\n\n"
            response += f"{summary}\n\n"
            
            response += "**Market Metrics:**\n"
            response += f"- Market Cap: {market_metrics.get('market_cap_formatted', 'N/A')}\n"
            response += f"- Rank: #{market_metrics.get('market_cap_rank', 'N/A')}\n"
            response += f"- 24h Volume: {market_metrics.get('volume_24h_formatted', 'N/A')}\n"
            response += f"- Liquidity: {liquidity_metrics.get('liquidity_rating', 'N/A')}\n\n"
            
            response += "**Supply Metrics:**\n"
            response += f"- Supply Model: {supply_metrics.get('supply_model', 'N/A')}\n"
            if supply_metrics.get('circulating_percentage'):
                response += f"- Circulating: {supply_metrics.get('circulating_percentage', 0):.1f}% of max supply\n"
            
            return response
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return f"Error analyzing fundamentals for {symbol}: {str(e)}"
    
    return analyze_fundamentals


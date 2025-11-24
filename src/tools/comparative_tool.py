"""
Comparative analysis tool for LangGraph agent.
"""

from typing import Dict, Any, List
import logging
from langchain_core.tools import tool
from src.analyzers.comparative_analyzer import ComparativeAnalyzer
from src.memory.cache_manager import cache_manager

logger = logging.getLogger(__name__)


class ComparativeTool:
    """Comparative analysis tool."""
    
    def __init__(self, analyzer: ComparativeAnalyzer):
        """
        Initialize comparative tool.
        
        Args:
            analyzer: ComparativeAnalyzer instance
        """
        self.analyzer = analyzer
    
    def analyze(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Perform comparative analysis with caching.
        
        Args:
            symbols: List of token symbols
            
        Returns:
            Analysis results
        """
        # Create cache key from sorted symbols
        symbols_key = "-".join(sorted([s.upper() for s in symbols]))
        cache_key = cache_manager.make_key("comparative", symbols_key)
        
        def fetch():
            logger.info(f"Performing comparative analysis for {', '.join(symbols)}")
            return self.analyzer.compare(symbols)
        
        return cache_manager.get_or_set(cache_key, fetch)


def create_comparative_tool(analyzer: ComparativeAnalyzer):
    """
    Create comparative analysis tool for LangGraph.
    
    Args:
        analyzer: ComparativeAnalyzer instance
        
    Returns:
        LangChain tool
    """
    tool_instance = ComparativeTool(analyzer)
    
    @tool
    def compare_tokens(symbols: str) -> str:
        """
        Compare multiple cryptocurrency tokens across all analysis dimensions.
        Use this when user wants to compare two or more tokens (e.g., "compare Bitcoin and Ethereum").
        
        Args:
            symbols: Comma-separated token symbols (e.g., "BTC,ETH,SOL")
        """
        try:
            # Parse symbols
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            
            if len(symbol_list) < 2:
                return "Please provide at least 2 tokens to compare (comma-separated)"
            
            # Perform analysis
            result = tool_instance.analyze(symbol_list)
            summary = result.get("summary", "")
            
            fundamental_comparison = result.get("fundamental_comparison", {})
            price_comparison = result.get("price_comparison", {})
            technical_comparison = result.get("technical_comparison", {})
            sentiment_comparison = result.get("sentiment_comparison", {})
            
            response = f"**Comparative Analysis: {' vs '.join(symbol_list)}**\n\n"
            response += f"{summary}\n\n"
            
            # Fundamental comparison
            response += "**Fundamental Comparison:**\n"
            for symbol in symbol_list:
                data = fundamental_comparison.get(symbol, {})
                if data.get("available"):
                    response += f"\n{symbol}:\n"
                    response += f"- Market Cap: {data.get('market_cap', 'N/A')}\n"
                    response += f"- Rank: #{data.get('market_cap_rank', 'N/A')}\n"
                    response += f"- Liquidity: {data.get('liquidity_rating', 'N/A')}\n"
            
            # Price comparison
            response += "\n**Price Performance:**\n"
            best_24h = price_comparison.get("best_performer_24h", {})
            worst_24h = price_comparison.get("worst_performer_24h", {})
            
            if best_24h:
                response += f"- Best 24h: {best_24h.get('symbol')} ({best_24h.get('value', 0):+.2f}%)\n"
            if worst_24h:
                response += f"- Worst 24h: {worst_24h.get('symbol')} ({worst_24h.get('value', 0):+.2f}%)\n"
            
            # Technical comparison
            response += "\n**Technical Signals:**\n"
            for symbol in symbol_list:
                data = technical_comparison.get(symbol, {})
                if data.get("available"):
                    response += f"- {symbol}: {data.get('overall_signal', 'N/A')} "
                    response += f"({data.get('signal_confidence', '0%')})\n"
            
            # Sentiment comparison
            response += "\n**Sentiment:**\n"
            most_positive = sentiment_comparison.get("most_positive", {})
            least_positive = sentiment_comparison.get("least_positive", {})
            
            if most_positive:
                response += f"- Most Positive: {most_positive.get('symbol')} "
                response += f"(Score: {most_positive.get('value', 0):.1f}/100)\n"
            if least_positive:
                response += f"- Least Positive: {least_positive.get('symbol')} "
                response += f"(Score: {least_positive.get('value', 0):.1f}/100)\n"
            
            return response
        except Exception as e:
            logger.error(f"Comparative analysis error: {e}")
            return f"Error comparing tokens: {str(e)}"
    
    return compare_tokens


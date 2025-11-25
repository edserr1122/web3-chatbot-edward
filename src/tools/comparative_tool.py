"""
Comparative analysis tool for LangGraph agent.
"""

from typing import Dict, Any, List
import logging
from langchain_core.tools import tool
from src.analyzers import ComparativeAnalyzer
from src.memory import cache_manager

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
    
    def analyze(self, symbols: List[str], days_ago: int = 0) -> Dict[str, Any]:
        """
        Perform comparative analysis with caching.
        
        Args:
            symbols: List of token symbols
            days_ago: Number of days ago to compare against (0 = compare tokens now, >0 = compare current vs historical)
            
        Returns:
            Analysis results
        """
        # Create cache key from sorted symbols and time parameter
        symbols_key = "-".join(sorted([s.upper() for s in symbols]))
        time_suffix = f"-vs-{days_ago}d" if days_ago > 0 else ""
        cache_key = cache_manager.make_key("comparative", f"{symbols_key}{time_suffix}")
        
        def fetch():
            if days_ago > 0:
                logger.info(f"Performing temporal comparative analysis for {', '.join(symbols)} (current vs {days_ago} days ago)")
                return self.analyzer.compare_temporal(symbols, days_ago)
            else:
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
    def compare_tokens(symbols: str, days_ago: int = 0) -> str:
        """
        Compare multiple cryptocurrency tokens across all analysis dimensions.
        
        Use this when user wants to:
        - Compare two or more tokens (e.g., "compare Bitcoin and Ethereum")
        - Compare current data against historical data (e.g., "compare BTC vs 3 days ago", "how has ETH changed since 7 days ago")
        
        For temporal comparisons (current vs past):
        - "Compare BTC against 3 days ago" → symbols="BTC", days_ago=3
        - "Compare BTC, ETH vs 7 days ago" → symbols="BTC,ETH", days_ago=7
        
        For token-to-token comparisons (no time):
        - "Compare BTC vs ETH" → symbols="BTC,ETH", days_ago=0
        
        Args:
            symbols: Comma-separated token symbols (e.g., "BTC,ETH,SOL")
            days_ago: Number of days in the past to compare against (default: 0 = compare tokens only)
        """
        try:
            # Parse symbols
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            
            if len(symbol_list) < 1:
                return "Please provide at least 1 token symbol"
            
            if days_ago > 0 and len(symbol_list) < 1:
                return "Please provide at least 1 token for temporal comparison"
            
            # Perform analysis
            result = tool_instance.analyze(symbol_list, days_ago=days_ago)
            summary = result.get("summary", "")
            
            # Check if this is a temporal comparison
            is_temporal = days_ago > 0
            
            if is_temporal:
                response = f"**Temporal Comparison: {' & '.join(symbol_list)} (Current vs {days_ago} days ago)**\n\n"
                response += f"{summary}\n\n"
                
                # Check if any data was successfully retrieved
                successful_count = result.get("successful_count", 0)
                total_count = result.get("total_count", len(symbol_list))
                
                if successful_count == 0:
                    # All failed - provide helpful error message
                    response += "\n**💡 Tip:** Temporal comparisons require historical data from CoinGecko. "
                    response += "If you're seeing rate limit errors, please wait 1-2 minutes before trying again.\n"
                    return response
                
                # Format temporal comparison results
                temporal_changes = result.get("temporal_changes", {})
                for symbol in symbol_list:
                    changes = temporal_changes.get(symbol, {})
                    if changes.get("available"):
                        response += f"\n**{symbol} - Changes over {days_ago} days:**\n"
                        response += f"- Current Price: {changes.get('current_price', 'N/A')} (was {changes.get('old_price', 'N/A')})\n"
                        response += f"- Price Change: {changes.get('price_change', 'N/A')}\n"
                        response += f"- Market Cap Change: {changes.get('market_cap_change', 'N/A')}\n"
                        response += f"- Volume Change: {changes.get('volume_change', 'N/A')}\n"
                        
                        # Add data quality notes if present
                        if changes.get("data_notes"):
                            response += f"- Note: {changes.get('data_notes')}\n"
                    else:
                        # Show error for this specific token
                        error = changes.get("error", "Unknown error")
                        response += f"\n**{symbol}:** ❌ {error}\n"
                
                if successful_count < total_count:
                    response += f"\n\n⚠️  Note: Only {successful_count}/{total_count} tokens had data available. "
                    response += "Some tokens may be rate-limited or have insufficient historical data.\n"
                
                return response
            
            # Standard token-to-token comparison
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


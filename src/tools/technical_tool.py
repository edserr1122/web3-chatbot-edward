"""
Technical analysis tool for LangGraph agent.
"""

from typing import Dict, Any
import logging
from langchain_core.tools import tool
from src.analyzers.technical_analyzer import TechnicalAnalyzer
from src.memory.cache_manager import cache_manager

logger = logging.getLogger(__name__)


class TechnicalTool:
    """Technical analysis tool."""
    
    def __init__(self, analyzer: TechnicalAnalyzer):
        """
        Initialize technical tool.
        
        Args:
            analyzer: TechnicalAnalyzer instance
        """
        self.analyzer = analyzer
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform technical analysis with caching.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Analysis results
        """
        cache_key = cache_manager.make_key("technical", symbol.upper())
        
        def fetch():
            logger.info(f"Performing technical analysis for {symbol}")
            return self.analyzer.analyze(symbol)
        
        return cache_manager.get_or_set(cache_key, fetch)


def create_technical_tool(analyzer: TechnicalAnalyzer):
    """
    Create technical analysis tool for LangGraph.
    
    Args:
        analyzer: TechnicalAnalyzer instance
        
    Returns:
        LangChain tool
    """
    tool_instance = TechnicalTool(analyzer)
    
    @tool
    def analyze_technical(symbol: str) -> str:
        """
        Analyze technical indicators like RSI, MACD, and moving averages.
        Use this for questions about technical analysis, trading signals, or indicators.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        """
        try:
            result = tool_instance.analyze(symbol)
            summary = result.get("summary", "")
            indicators = result.get("indicators", {})
            signals = result.get("signals", {})
            
            response = f"**Technical Analysis for {symbol.upper()}:**\n\n"
            response += f"{summary}\n\n"
            
            # RSI
            rsi = indicators.get("rsi", {})
            if rsi.get("value"):
                response += f"**RSI (14):** {rsi['value']} - {rsi.get('signal', 'Unknown')}\n"
                response += f"  {rsi.get('interpretation', '')}\n\n"
            
            # MACD
            macd = indicators.get("macd", {})
            if macd.get("signal"):
                response += f"**MACD:** {macd.get('signal', 'Unknown')}\n"
                response += f"  {macd.get('interpretation', '')}\n\n"
            
            # Moving Averages
            ma = indicators.get("moving_averages", {})
            if ma.get("trend"):
                response += f"**Moving Averages:** {ma.get('trend', 'Unknown')} trend\n"
                response += f"  SMA(20): {ma.get('sma_20', 'N/A')}\n"
                response += f"  SMA(50): {ma.get('sma_50', 'N/A')}\n\n"
            
            # Overall signal
            response += f"**Overall Signal:** {signals.get('overall', 'Neutral')} "
            response += f"({signals.get('confidence', '0%')} indicator agreement)\n"
            
            return response
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return f"Error analyzing technical indicators for {symbol}: {str(e)}"
    
    return analyze_technical


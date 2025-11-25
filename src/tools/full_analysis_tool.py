"""
Full analysis tool for LangGraph agent.
Performs comprehensive analysis across all dimensions.
"""

from typing import Dict, Any
import logging
from langchain_core.tools import tool
from src.tools import (
    FundamentalTool,
    PriceTool,
    TechnicalTool,
    SentimentTool,
)

logger = logging.getLogger(__name__)


class FullAnalysisTool:
    """Full analysis tool combining all analyses."""
    
    def __init__(
        self,
        fundamental_tool: FundamentalTool,
        price_tool: PriceTool,
        technical_tool: TechnicalTool,
        sentiment_tool: SentimentTool
    ):
        """
        Initialize full analysis tool.
        
        Args:
            fundamental_tool: FundamentalTool instance
            price_tool: PriceTool instance
            technical_tool: TechnicalTool instance
            sentiment_tool: SentimentTool instance
        """
        self.fundamental = fundamental_tool
        self.price = price_tool
        self.technical = technical_tool
        self.sentiment = sentiment_tool
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform full analysis across all dimensions.
        
        Args:
            symbol: Token symbol
            
        Returns:
            Combined analysis results
        """
        return {
            "fundamental": self.fundamental.analyze(symbol),
            "price": self.price.analyze(symbol),
            "technical": self.technical.analyze(symbol),
            "sentiment": self.sentiment.analyze(symbol),
        }


def create_full_analysis_tool(
    fundamental_tool: FundamentalTool,
    price_tool: PriceTool,
    technical_tool: TechnicalTool,
    sentiment_tool: SentimentTool
):
    """
    Create full analysis tool for LangGraph.
    
    Args:
        fundamental_tool: FundamentalTool instance
        price_tool: PriceTool instance
        technical_tool: TechnicalTool instance
        sentiment_tool: SentimentTool instance
        
    Returns:
        LangChain tool
    """
    tool_instance = FullAnalysisTool(
        fundamental_tool,
        price_tool,
        technical_tool,
        sentiment_tool
    )
    
    @tool
    def full_analysis(symbol: str) -> str:
        """
        Perform comprehensive analysis across all dimensions (fundamental, price, 
        technical, and sentiment). Use this when user asks for complete/full/comprehensive 
        analysis or general "tell me about" questions.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        """
        try:
            results = tool_instance.analyze(symbol)
            
            response = f"**Comprehensive Analysis for {symbol.upper()}**\n\n"
            
            # Fundamental
            if "fundamental" in results:
                response += "═══════════════════════════════════════\n"
                response += "FUNDAMENTAL ANALYSIS\n"
                response += "═══════════════════════════════════════\n"
                response += f"{results['fundamental'].get('summary', '')}\n\n"
            
            # Price
            if "price" in results:
                response += "═══════════════════════════════════════\n"
                response += "PRICE ANALYSIS\n"
                response += "═══════════════════════════════════════\n"
                response += f"{results['price'].get('summary', '')}\n\n"
            
            # Technical
            if "technical" in results:
                response += "═══════════════════════════════════════\n"
                response += "TECHNICAL ANALYSIS\n"
                response += "═══════════════════════════════════════\n"
                response += f"{results['technical'].get('summary', '')}\n\n"
            
            # Sentiment
            if "sentiment" in results:
                response += "═══════════════════════════════════════\n"
                response += "SENTIMENT ANALYSIS\n"
                response += "═══════════════════════════════════════\n"
                response += f"{results['sentiment'].get('summary', '')}\n\n"
            
            return response
        except Exception as e:
            logger.error(f"Full analysis error for {symbol}: {e}")
            return f"Error performing full analysis for {symbol}: {str(e)}"
    
    return full_analysis


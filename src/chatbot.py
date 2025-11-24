"""
Main Chatbot Class - Interface-Agnostic Core
Orchestrates all components and provides a clean API for any interface (CLI, Web, API).
"""

import logging
from typing import Optional, Dict, Any
from src.agents.crypto_agent import CryptoAgent
from src.analyzers.fundamental_analyzer import FundamentalAnalyzer
from src.analyzers.price_analyzer import PriceAnalyzer
from src.analyzers.technical_analyzer import TechnicalAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.comparative_analyzer import ComparativeAnalyzer
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.coinmarketcap_client import CoinMarketCapClient
from src.data_sources.lunarcrush_client import LunarCrushClient
from src.data_sources.cryptopanic_client import CryptoPanicClient
from src.data_sources.messari_client import MessariClient
from src.data_sources.fear_greed_client import FearGreedClient
from src.utils.config import config

logger = logging.getLogger(__name__)


class CryptoChatbot:
    """
    Main chatbot class - interface-agnostic core.
    Handles initialization and provides clean API for any interface.
    """
    
    def __init__(self):
        """Initialize the chatbot with all components."""
        logger.info("🚀 Initializing Crypto Analysis Chatbot...")
        
        # Validate configuration
        if not config.validate():
            raise RuntimeError("Configuration validation failed. Check your .env file.")
        
        # Initialize data sources
        self._init_data_sources()
        
        # Initialize analyzers
        self._init_analyzers()
        
        # Initialize agent
        self._init_agent()
        
        logger.info("✅ Crypto Analysis Chatbot initialized successfully")
    
    def _init_data_sources(self):
        """Initialize all data source clients."""
        logger.info("Initializing data sources...")
        
        # CoinGecko (works with or without API key)
        self.coingecko = CoinGeckoClient(config.COINGECKO_API_KEY)
        logger.info("  ✓ CoinGecko client initialized")
        
        # CoinMarketCap (optional)
        self.coinmarketcap = None
        if config.COINMARKETCAP_API_KEY:
            try:
                self.coinmarketcap = CoinMarketCapClient(config.COINMARKETCAP_API_KEY)
                logger.info("  ✓ CoinMarketCap client initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ CoinMarketCap initialization failed: {e}")
        
        # LunarCrush (optional)
        self.lunarcrush = None
        if config.LUNARCRUSH_API_KEY:
            try:
                self.lunarcrush = LunarCrushClient(config.LUNARCRUSH_API_KEY)
                logger.info("  ✓ LunarCrush client initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ LunarCrush initialization failed: {e}")
        
        # CryptoPanic (optional)
        self.cryptopanic = None
        if config.CRYPTOPANIC_API_KEY:
            try:
                self.cryptopanic = CryptoPanicClient(config.CRYPTOPANIC_API_KEY)
                logger.info("  ✓ CryptoPanic client initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ CryptoPanic initialization failed: {e}")
        
        # Messari (optional)
        self.messari = None
        if config.MESSARI_API_KEY:
            try:
                self.messari = MessariClient(config.MESSARI_API_KEY)
                logger.info("  ✓ Messari client initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ Messari initialization failed: {e}")
        
        # Fear & Greed Index (always available, no key needed)
        self.fear_greed = FearGreedClient()
        logger.info("  ✓ Fear & Greed Index client initialized")
    
    def _init_analyzers(self):
        """Initialize all analyzer modules."""
        logger.info("Initializing analyzers...")
        
        # Fundamental Analyzer
        self.fundamental_analyzer = FundamentalAnalyzer(
            coingecko_client=self.coingecko,
            coinmarketcap_client=self.coinmarketcap,
            messari_client=self.messari
        )
        logger.info("  ✓ Fundamental analyzer initialized")
        
        # Price Analyzer
        self.price_analyzer = PriceAnalyzer(
            coingecko_client=self.coingecko,
            coinmarketcap_client=self.coinmarketcap
        )
        logger.info("  ✓ Price analyzer initialized")
        
        # Technical Analyzer
        self.technical_analyzer = TechnicalAnalyzer(
            coingecko_client=self.coingecko
        )
        logger.info("  ✓ Technical analyzer initialized")
        
        # Sentiment Analyzer
        self.sentiment_analyzer = SentimentAnalyzer(
            lunarcrush_client=self.lunarcrush,
            cryptopanic_client=self.cryptopanic,
            fear_greed_client=self.fear_greed,
            coingecko_client=self.coingecko,
            messari_client=self.messari
        )
        logger.info("  ✓ Sentiment analyzer initialized")
        
        # Comparative Analyzer
        self.comparative_analyzer = ComparativeAnalyzer(
            fundamental_analyzer=self.fundamental_analyzer,
            price_analyzer=self.price_analyzer,
            technical_analyzer=self.technical_analyzer,
            sentiment_analyzer=self.sentiment_analyzer
        )
        logger.info("  ✓ Comparative analyzer initialized")
    
    def _init_agent(self):
        """Initialize the LangGraph agent."""
        logger.info("Initializing LangGraph agent...")
        
        self.agent = CryptoAgent(
            fundamental_analyzer=self.fundamental_analyzer,
            price_analyzer=self.price_analyzer,
            technical_analyzer=self.technical_analyzer,
            sentiment_analyzer=self.sentiment_analyzer,
            comparative_analyzer=self.comparative_analyzer
        )
        
        logger.info("  ✓ LangGraph agent initialized")
    
    def chat(self, message: str) -> str:
        """
        Process a user message and return the response.
        
        Args:
            message: User's input message
            
        Returns:
            Agent's response
        """
        return self.agent.chat(message)
    
    def get_conversation_history(self) -> list:
        """
        Get the conversation history.
        
        Returns:
            List of messages
        """
        return self.agent.get_conversation_history()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.agent.clear_memory()
        logger.info("Conversation history cleared")
    
    def set_session(self, session_id: str):
        """
        Set the session ID for conversation isolation.
        
        Args:
            session_id: Session identifier
        """
        self.agent.set_session(session_id)
    
    def get_available_data_sources(self) -> Dict[str, bool]:
        """
        Get information about available data sources.
        
        Returns:
            Dictionary of data source availability
        """
        return config.get_available_data_sources()
    
    def print_status(self):
        """Print chatbot status and configuration."""
        print("\n" + "=" * 70)
        print("🤖 CRYPTO ANALYSIS CHATBOT - STATUS")
        print("=" * 70)
        
        # Configuration
        config.print_status()
        
        # Capabilities
        print("\n" + "=" * 70)
        print("📊 ANALYSIS CAPABILITIES")
        print("=" * 70)
        print("  ✓ Fundamental Analysis (Market cap, supply, volume)")
        print("  ✓ Price Analysis (Trends, volatility, support/resistance)")
        print("  ✓ Technical Analysis (RSI, MACD, Moving Averages)")
        print("  ✓ Sentiment Analysis (Social, news, Fear & Greed)")
        print("  ✓ Comparative Analysis (Multi-token comparison)")
        print("=" * 70 + "\n")


def create_chatbot() -> CryptoChatbot:
    """
    Factory function to create and initialize a chatbot instance.
    
    Returns:
        Initialized CryptoChatbot instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        return CryptoChatbot()
    except Exception as e:
        logger.error(f"Failed to create chatbot: {e}", exc_info=True)
        raise RuntimeError(f"Chatbot initialization failed: {e}")


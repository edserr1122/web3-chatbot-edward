"""
Main Chatbot Class - Interface-Agnostic Core
Orchestrates all components and provides a clean API for any interface (CLI, Web, API).
"""

import logging
from typing import Optional, Dict, Any
from src.agents import CryptoAgent
from src.analyzers import (
    FundamentalAnalyzer,
    PriceAnalyzer,
    TechnicalAnalyzer,
    SentimentAnalyzer,
    ComparativeAnalyzer,
)
from src.data_sources import (
    CoinGeckoClient,
    CoinMarketCapClient,
    CryptoPanicClient,
    MessariClient,
    FearGreedClient,
)
from src.utils import config, setup_logging

logger = logging.getLogger(__name__)


class CryptoChatbot:
    """
    Main chatbot class - interface-agnostic core.
    Handles initialization and provides clean API for any interface.
    """
    
    def __init__(self):
        """Initialize the chatbot with all components."""
        # Ensure logging is configured even in non-CLI contexts (tests, notebooks, etc.)
        setup_logging(enable_console=False)
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
        
        # CoinCap (requires API key - fallback for historical data)
        from src.data_sources import CoinCapClient
        self.coincap = None
        if config.COINCAP_API_KEY:
            try:
                self.coincap = CoinCapClient(config.COINCAP_API_KEY)
                # Test connection
                if self.coincap.test_connection():
                    logger.info("  ✓ CoinCap client initialized (fallback for historical data)")
                else:
                    self.coincap = None
                    logger.warning("  ⚠️ CoinCap API connection failed - check your API key")
            except Exception as e:
                logger.warning(f"  ⚠️ CoinCap initialization failed: {e}")
                self.coincap = None
        else:
            logger.info("  ℹ️  CoinCap API key not configured - skipping (fallback unavailable)")
        
        # Binance (free, no API key required - preferred source for OHLC data)
        # Try Binance global first (richer data), then Binance.US for geo-restricted regions
        from src.data_sources import BinanceClient, BinanceUSClient
        self.binance_clients = []
        self.binance_global = None
        self.binance_us = None
        
        # Binance (global) first
        try:
            binance_global = BinanceClient()
            if binance_global.test_connection():
                self.binance_global = binance_global
                self.binance_clients.append(binance_global)
                logger.info("  ✓ Binance (global) client initialized (primary OHLC source)")
            else:
                logger.warning("  ⚠️ Binance (global) connection failed")
        except Exception as e:
            logger.warning(f"  ⚠️ Binance (global) not available: {e}")
        
        # Binance.US second
        try:
            binance_us = BinanceUSClient()
            if binance_us.test_connection():
                self.binance_us = binance_us
                self.binance_clients.append(binance_us)
                logger.info("  ✓ Binance.US client initialized (secondary OHLC source)")
            else:
                logger.warning("  ⚠️ Binance.US connection failed")
        except Exception as e:
            logger.warning(f"  ⚠️ Binance.US not available: {e}")
        
        if not self.binance_clients:
            logger.warning("  ⚠️ No Binance client available (global and US failed) - will rely on CoinGecko/CoinCap")
        
        # CoinMarketCap (optional)
        self.coinmarketcap = None
        if config.COINMARKETCAP_API_KEY:
            try:
                self.coinmarketcap = CoinMarketCapClient(config.COINMARKETCAP_API_KEY)
                logger.info("  ✓ CoinMarketCap client initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ CoinMarketCap initialization failed: {e}")
        
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
        
        # Fundamental Analyzer (with global market context)
        self.fundamental_analyzer = FundamentalAnalyzer(
            coingecko_client=self.coingecko,
            coinmarketcap_client=self.coinmarketcap,
            messari_client=self.messari,
            cryptopanic_client=self.cryptopanic,
            alternative_client=self.fear_greed  # Global market context
        )
        logger.info("  ✓ Fundamental analyzer initialized")
        
        # Price Analyzer (with fallbacks for historical data)
        self.price_analyzer = PriceAnalyzer(
            coingecko_client=self.coingecko,
            coinmarketcap_client=self.coinmarketcap,
            coincap_client=self.coincap,  # Fallback for historical data
            binance_clients=self.binance_clients  # Fallback for historical data
        )
        logger.info("  ✓ Price analyzer initialized")
        
        # Technical Analyzer (CoinGecko + Binance fallback for OHLC)
        self.technical_analyzer = TechnicalAnalyzer(
            coingecko_client=self.coingecko,
            binance_clients=self.binance_clients,  # Binance (global → US)
            coincap_client=self.coincap  # Final fallback via TA endpoints
        )
        logger.info("  ✓ Technical analyzer initialized")
        
        # Sentiment Analyzer (news + market mood + community)
        self.sentiment_analyzer = SentimentAnalyzer(
            cryptopanic_client=self.cryptopanic,
            fear_greed_client=self.fear_greed,
            coingecko_client=self.coingecko
        )
        logger.info("  ✓ Sentiment analyzer initialized")
        
        # Comparative Analyzer (with CoinGecko & CoinMarketCap clients for optimized batch fetching)
        self.comparative_analyzer = ComparativeAnalyzer(
            fundamental_analyzer=self.fundamental_analyzer,
            price_analyzer=self.price_analyzer,
            technical_analyzer=self.technical_analyzer,
            sentiment_analyzer=self.sentiment_analyzer,
            coingecko_client=self.coingecko,  # Enable optimized batch API calls
            coinmarketcap_client=self.coinmarketcap,  # Enable CoinMarketCap batch calls
            coincap_client=self.coincap,  # Fallback 1 for historical data
            binance_clients=self.binance_clients  # Fallback 2 for historical data
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
        print("  ✓ Sentiment Analysis (News, Fear & Greed, Community)")
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


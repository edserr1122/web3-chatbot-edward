"""
Configuration management for the crypto chatbot.
Loads environment variables and provides configuration access.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_INTENT_CLASSIFIER_MODEL: str = os.getenv("GROQ_INTENT_CLASSIFIER_MODEL", "gemma-7b-it")  # Smaller, faster model for intent classification
    
    # Crypto Data API Keys
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    COINMARKETCAP_API_KEY: Optional[str] = os.getenv("COINMARKETCAP_API_KEY")
    COINCAP_API_KEY: Optional[str] = os.getenv("COINCAP_API_KEY")
    
    # Sentiment & News API Keys
    CRYPTOPANIC_API_KEY: Optional[str] = os.getenv("CRYPTOPANIC_API_KEY")
    MESSARI_API_KEY: Optional[str] = os.getenv("MESSARI_API_KEY")
    
    # Free APIs
    FEAR_GREED_API_URL: str = os.getenv(
        "FEAR_GREED_API_URL", 
        "https://api.alternative.me/fng/"
    )
    
    # Redis Cloud Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_USERNAME: Optional[str] = os.getenv("REDIS_USERNAME", "default")
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    
    # Cache & History Settings
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes
    HISTORY_DB_PATH: str = os.getenv(
        "HISTORY_DB_PATH",
        os.path.join(os.getcwd(), "data", "chat_history.db")
    )
    HISTORY_CONTEXT_LIMIT: int = int(os.getenv("HISTORY_CONTEXT_LIMIT", "20"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_redis_url(cls) -> str:
        """
        Build Redis connection URL for Redis Cloud.
        
        Returns:
            str: Redis connection URL
        """
        if not cls.REDIS_ENABLED:
            return ""
        
        protocol = "rediss" if cls.REDIS_SSL else "redis"
        auth = ""
        
        if cls.REDIS_PASSWORD:
            if cls.REDIS_USERNAME and cls.REDIS_USERNAME != "default":
                auth = f"{cls.REDIS_USERNAME}:{cls.REDIS_PASSWORD}@"
            else:
                auth = f"default:{cls.REDIS_PASSWORD}@"
        
        return f"{protocol}://{auth}{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def get_redis_connection_kwargs(cls) -> dict:
        """
        Get Redis connection keyword arguments for redis-py client.
        
        Returns:
            dict: Connection parameters for redis.Redis()
        """
        kwargs = {
            "host": cls.REDIS_HOST,
            "port": cls.REDIS_PORT,
            "db": cls.REDIS_DB,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
        }
        
        if cls.REDIS_PASSWORD:
            kwargs["password"] = cls.REDIS_PASSWORD
        
        if cls.REDIS_USERNAME and cls.REDIS_USERNAME != "default":
            kwargs["username"] = cls.REDIS_USERNAME
        
        if cls.REDIS_SSL:
            kwargs["ssl"] = True
            kwargs["ssl_cert_reqs"] = None  # For self-signed certificates
        
        return kwargs
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_keys = [
            ("GROQ_API_KEY", cls.GROQ_API_KEY),
        ]
        
        missing_keys = []
        for key_name, key_value in required_keys:
            if not key_value:
                missing_keys.append(key_name)
        
        if missing_keys:
            print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
            print("💡 Please create a .env file based on .env.example and add your API keys.")
            return False
        
        # Validate Redis Cloud configuration if enabled
        if cls.REDIS_ENABLED:
            if not cls.REDIS_HOST or cls.REDIS_HOST == "localhost":
                print("⚠️  Warning: REDIS_ENABLED is true but REDIS_HOST is not configured for Redis Cloud")
                print("💡 Please set REDIS_HOST to your Redis Cloud endpoint")
            
            if not cls.REDIS_PASSWORD:
                print("⚠️  Warning: Redis Cloud typically requires REDIS_PASSWORD")
                print("💡 Please set REDIS_PASSWORD in your .env file")
        
        return True
    
    @classmethod
    def get_available_data_sources(cls) -> dict:
        """
        Get a dictionary of available data sources based on configured API keys.
        
        Returns:
            dict: Dictionary with data source names and their availability status
        """
        return {
            "coingecko": bool(cls.COINGECKO_API_KEY) or True,  # Has free tier without key
            "coinmarketcap": bool(cls.COINMARKETCAP_API_KEY),
            "coincap": bool(cls.COINCAP_API_KEY),
            "cryptopanic": bool(cls.CRYPTOPANIC_API_KEY),
            "messari": bool(cls.MESSARI_API_KEY),
            "fear_greed": True,  # Always available (free API)
            "redis": cls.REDIS_ENABLED,
        }
    
    @classmethod
    def print_status(cls):
        """Print configuration status for debugging."""
        print("=" * 60)
        print("🔧 Configuration Status")
        print("=" * 60)
        print(f"GROQ Model: {cls.GROQ_MODEL}")
        print(f"GROQ API Key: {'✅ Set' if cls.GROQ_API_KEY else '❌ Missing'}")
        print(f"\nData Sources:")
        
        sources = cls.get_available_data_sources()
        for source, available in sources.items():
            status = "✅" if available else "❌"
            print(f"  {status} {source.replace('_', ' ').title()}")
        
        if cls.REDIS_ENABLED:
            print(f"\nRedis Configuration:")
            print(f"  Host: {cls.REDIS_HOST}")
            print(f"  Port: {cls.REDIS_PORT}")
            print(f"  SSL: {'✅ Enabled' if cls.REDIS_SSL else '❌ Disabled'}")
            print(f"  Password: {'✅ Set' if cls.REDIS_PASSWORD else '❌ Not set'}")
            print(f"  Database: {cls.REDIS_DB}")
        
        print(f"\nCache TTL: {cls.CACHE_TTL_SECONDS} seconds")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 60)


# Singleton instance
config = Config()
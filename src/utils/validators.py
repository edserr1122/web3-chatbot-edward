"""
Input validation utilities for the crypto chatbot.
Validates user inputs, token symbols, and API responses.
"""

import re
from typing import Optional, List, Dict, Any


class InputValidator:
    """Validates user inputs and data."""
    
    # Common cryptocurrency symbols (extendable)
    KNOWN_SYMBOLS = {
        "btc", "bitcoin",
        "eth", "ethereum",
        "bnb", "binance",
        "sol", "solana",
        "ada", "cardano",
        "xrp", "ripple",
        "dot", "polkadot",
        "avax", "avalanche",
        "matic", "polygon",
        "link", "chainlink",
        "atom", "cosmos",
        "uni", "uniswap",
        "ltc", "litecoin",
        "etc", "ethereum classic",
        "xlm", "stellar",
        "algo", "algorand",
        "vet", "vechain",
        "icp", "internet computer",
        "fil", "filecoin",
        "trx", "tron",
        "aave", "aave",
        "mkr", "maker",
        "snx", "synthetix",
    }
    
    @staticmethod
    def is_valid_token_input(user_input: str) -> bool:
        """
        Check if user input contains a valid token reference.
        
        Args:
            user_input: User's input string
            
        Returns:
            bool: True if input appears to be asking about a crypto token
        """
        if not user_input or len(user_input.strip()) == 0:
            return False
        
        user_input_lower = user_input.lower()
        
        # Check for common token-related keywords
        token_keywords = [
            "tell me about", "analyze", "analysis", "what is", "what's",
            "price", "market", "sentiment", "technical", "fundamental",
            "compare", "versus", "vs", "trend", "performance"
        ]
        
        has_keyword = any(keyword in user_input_lower for keyword in token_keywords)
        has_known_symbol = any(symbol in user_input_lower for symbol in InputValidator.KNOWN_SYMBOLS)
        
        return has_keyword or has_known_symbol
    
    @staticmethod
    def extract_token_symbols(user_input: str) -> List[str]:
        """
        Extract potential token symbols from user input.
        
        Args:
            user_input: User's input string
            
        Returns:
            List[str]: List of potential token symbols found
        """
        user_input_lower = user_input.lower()
        found_tokens = []
        
        for symbol in InputValidator.KNOWN_SYMBOLS:
            if symbol in user_input_lower:
                # Avoid duplicates (e.g., "btc" and "bitcoin" both found)
                if symbol.upper() not in [t.upper() for t in found_tokens]:
                    found_tokens.append(symbol)
        
        return found_tokens
    
    @staticmethod
    def is_crypto_related(user_input: str) -> bool:
        """
        Check if user input is related to cryptocurrency domain.
        
        Args:
            user_input: User's input string
            
        Returns:
            bool: True if input is crypto-related
        """
        user_input_lower = user_input.lower()
        
        crypto_keywords = [
            "crypto", "cryptocurrency", "token", "coin", "blockchain",
            "defi", "web3", "nft", "mining", "staking", "wallet",
            "exchange", "trading", "market cap", "volume", "liquidity",
            "bull", "bear", "hodl", "pump", "dump", "moon"
        ]
        
        has_crypto_keyword = any(keyword in user_input_lower for keyword in crypto_keywords)
        has_known_symbol = any(symbol in user_input_lower for symbol in InputValidator.KNOWN_SYMBOLS)
        
        return has_crypto_keyword or has_known_symbol
    
    @staticmethod
    def normalize_token_symbol(symbol: str) -> str:
        """
        Normalize token symbol to uppercase standard format.
        
        Args:
            symbol: Token symbol or name
            
        Returns:
            str: Normalized symbol
        """
        symbol_lower = symbol.lower().strip()
        
        # Map common names to symbols
        name_to_symbol = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "binance": "BNB",
            "solana": "SOL",
            "cardano": "ADA",
            "ripple": "XRP",
            "polkadot": "DOT",
            "avalanche": "AVAX",
            "polygon": "MATIC",
            "chainlink": "LINK",
            "cosmos": "ATOM",
            "uniswap": "UNI",
            "litecoin": "LTC",
            "stellar": "XLM",
            "algorand": "ALGO",
            "vechain": "VET",
            "tron": "TRX",
        }
        
        if symbol_lower in name_to_symbol:
            return name_to_symbol[symbol_lower]
        
        return symbol.upper()
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Validate that API response contains required fields.
        
        Args:
            response: API response dictionary
            required_fields: List of required field names
            
        Returns:
            bool: True if all required fields are present
        """
        if not isinstance(response, dict):
            return False
        
        for field in required_fields:
            if field not in response:
                return False
        
        return True
    
    @staticmethod
    def sanitize_input(user_input: str, max_length: int = 500) -> str:
        """
        Sanitize user input by removing excessive whitespace and limiting length.
        
        Args:
            user_input: Raw user input
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized input
        """
        # Remove excessive whitespace
        sanitized = " ".join(user_input.split())
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def is_exit_command(user_input: str) -> bool:
        """
        Check if user input is an exit command.
        
        Args:
            user_input: User's input string
            
        Returns:
            bool: True if input is an exit command
        """
        exit_commands = ["exit", "quit", "bye", "goodbye", "stop", "end"]
        return user_input.lower().strip() in exit_commands
"""
Output formatting utilities for the crypto chatbot.
Formats data for display in CLI and other interfaces.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class OutputFormatter:
    """Formats output data for display."""
    
    @staticmethod
    def format_price(price: float, currency: str = "USD") -> str:
        """
        Format price with appropriate currency symbol and decimals.
        
        Args:
            price: Price value
            currency: Currency code (default: USD)
            
        Returns:
            str: Formatted price string
        """
        if currency.upper() == "USD":
            if price >= 1:
                return f"${price:,.2f}"
            elif price >= 0.01:
                return f"${price:.4f}"
            else:
                return f"${price:.8f}"
        else:
            return f"{price:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format percentage with + or - sign and color indicator.
        
        Args:
            value: Percentage value
            decimals: Number of decimal places
            
        Returns:
            str: Formatted percentage string
        """
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.{decimals}f}%"
    
    @staticmethod
    def format_large_number(number: float, decimals: int = 2) -> str:
        """
        Format large numbers with K, M, B, T suffixes.
        
        Args:
            number: Number to format
            decimals: Number of decimal places
            
        Returns:
            str: Formatted number string (e.g., "1.5B", "250.3M")
        """
        if number >= 1_000_000_000_000:  # Trillion
            return f"${number / 1_000_000_000_000:.{decimals}f}T"
        elif number >= 1_000_000_000:  # Billion
            return f"${number / 1_000_000_000:.{decimals}f}B"
        elif number >= 1_000_000:  # Million
            return f"${number / 1_000_000:.{decimals}f}M"
        elif number >= 1_000:  # Thousand
            return f"${number / 1_000:.{decimals}f}K"
        else:
            return f"${number:.{decimals}f}"
    
    @staticmethod
    def format_market_data(data: Dict[str, Any]) -> str:
        """
        Format market data dictionary into readable string.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            str: Formatted market data string
        """
        lines = []
        
        if "symbol" in data:
            lines.append(f"Symbol: {data['symbol'].upper()}")
        
        if "price" in data:
            lines.append(f"Price: {OutputFormatter.format_price(data['price'])}")
        
        if "market_cap" in data:
            lines.append(f"Market Cap: {OutputFormatter.format_large_number(data['market_cap'])}")
        
        if "volume_24h" in data:
            lines.append(f"24h Volume: {OutputFormatter.format_large_number(data['volume_24h'])}")
        
        if "price_change_24h" in data:
            lines.append(f"24h Change: {OutputFormatter.format_percentage(data['price_change_24h'])}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_timestamp(timestamp: Optional[datetime] = None) -> str:
        """
        Format timestamp for display.
        
        Args:
            timestamp: Datetime object (default: current time)
            
        Returns:
            str: Formatted timestamp string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def format_analysis_section(title: str, content: str) -> str:
        """
        Format an analysis section with title and content.
        
        Args:
            title: Section title
            content: Section content
            
        Returns:
            str: Formatted section
        """
        separator = "=" * 60
        return f"\n{separator}\n{title.upper()}\n{separator}\n{content}\n"
    
    @staticmethod
    def format_error_message(error: Exception, context: str = "") -> str:
        """
        Format error message for user display.
        
        Args:
            error: Exception object
            context: Additional context about the error
            
        Returns:
            str: Formatted error message
        """
        error_msg = f"❌ Error: {str(error)}"
        if context:
            error_msg += f"\nContext: {context}"
        return error_msg
    
    @staticmethod
    def format_list(items: List[Any], bullet: str = "•") -> str:
        """
        Format a list of items with bullets.
        
        Args:
            items: List of items to format
            bullet: Bullet character
            
        Returns:
            str: Formatted list
        """
        return "\n".join([f"{bullet} {item}" for item in items])
    
    @staticmethod
    def format_table(data: List[Dict[str, Any]], headers: List[str]) -> str:
        """
        Format data as a simple text table.
        
        Args:
            data: List of dictionaries containing row data
            headers: List of column headers
            
        Returns:
            str: Formatted table string
        """
        if not data:
            return "No data available"
        
        # Calculate column widths
        col_widths = {header: len(header) for header in headers}
        for row in data:
            for header in headers:
                value_len = len(str(row.get(header, "")))
                col_widths[header] = max(col_widths[header], value_len)
        
        # Build header row
        header_row = " | ".join([h.ljust(col_widths[h]) for h in headers])
        separator = "-" * len(header_row)
        
        # Build data rows
        data_rows = []
        for row in data:
            row_str = " | ".join([
                str(row.get(h, "")).ljust(col_widths[h]) for h in headers
            ])
            data_rows.append(row_str)
        
        return f"{header_row}\n{separator}\n" + "\n".join(data_rows)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
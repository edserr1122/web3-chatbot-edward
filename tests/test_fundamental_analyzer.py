"""
Tests for fundamental analysis functionality.
"""

import pytest
from unittest.mock import Mock, patch
from src.analyzers.fundamental_analyzer import FundamentalAnalyzer


class TestFundamentalAnalyzer:
    """Test fundamental analysis."""
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock API clients."""
        coingecko = Mock()
        coinmarketcap = Mock()
        messari = Mock()
        cryptopanic = Mock()
        alternative = Mock()
        
        return {
            'coingecko': coingecko,
            'coinmarketcap': coinmarketcap,
            'messari': messari,
            'cryptopanic': cryptopanic,
            'alternative': alternative
        }
    
    def test_analyze_returns_structure(self, mock_clients):
        """Test that analyze returns expected structure."""
        analyzer = FundamentalAnalyzer(
            coingecko_client=mock_clients['coingecko'],
            coinmarketcap_client=mock_clients['coinmarketcap'],
            messari_client=mock_clients['messari']
        )
        
        # Mock successful responses
        mock_clients['coingecko'].get_token_data.return_value = {
            "id": "bitcoin",
            "symbol": "BTC",
            "name": "Bitcoin",
            "price_usd": 50000,
            "market_cap": 1000000000,
            "market_cap_rank": 1,
            "total_volume": 50000000,
            "circulating_supply": 19000000
        }
        
        mock_clients['coinmarketcap'].get_token_data.return_value = {
            "id": 1,
            "symbol": "BTC",
            "name": "Bitcoin",
            "price_usd": 50000,
            "market_cap": 1000000000,
            "market_cap_rank": 1,
            "total_volume": 50000000,
            "circulating_supply": 19000000
        }
        
        mock_clients['messari'].get_token_data.return_value = None
        
        result = analyzer.analyze("BTC")

        assert "market_metrics" in result
        assert "market_cap" in result["market_metrics"]
        assert "supply_metrics" in result
        assert "liquidity_metrics" in result
        assert result["symbol"].upper() == "BTC"
    
    def test_analyze_handles_missing_data(self, mock_clients):
        """Test that analyzer handles missing data gracefully."""
        analyzer = FundamentalAnalyzer(
            coingecko_client=mock_clients['coingecko'],
            coinmarketcap_client=mock_clients['coinmarketcap']
        )
        
        # Mock failure responses
        mock_clients['coingecko'].get_token_data.return_value = None
        mock_clients['coinmarketcap'].get_token_data.return_value = None
        
        result = analyzer.analyze("INVALID")
        
        # Should still return a structure, even if empty
        assert isinstance(result, dict)
        assert "symbol" in result
    
    def test_analyze_uses_fallback_sources(self, mock_clients):
        """Test that analyzer uses fallback sources when primary fails."""
        analyzer = FundamentalAnalyzer(
            coingecko_client=mock_clients['coingecko'],
            coinmarketcap_client=mock_clients['coinmarketcap'],
            messari_client=mock_clients['messari']
        )
        
        # Primary source fails
        mock_clients['coingecko'].get_token_data.return_value = None
        
        # Fallback source succeeds
        mock_clients['coinmarketcap'].get_token_data.return_value = {
            "id": 1,
            "symbol": "BTC",
            "name": "Bitcoin",
            "price_usd": 50000,
            "market_cap": 1000000000,
            "market_cap_rank": 1,
            "total_volume": 50000000,
            "circulating_supply": 19000000
        }
        
        mock_clients['messari'].get_token_data.return_value = None
        
        result = analyzer.analyze("BTC")

        # Should have used fallback
        assert "market_metrics" in result
        assert result["market_metrics"]["market_cap"] is not None or result["market_metrics"].get("volume_24h") is not None
        # Should have tried primary first
        mock_clients['coingecko'].get_token_data.assert_called()


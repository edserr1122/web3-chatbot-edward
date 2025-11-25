"""
Tests for base API client functionality (circuit breaker, caching).
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.data_sources.base_client import BaseAPIClient


class MockAPIClient(BaseAPIClient):
    """Mock API client for testing base functionality."""
    
    def get_token_data(self, symbol: str):
        """Required abstract method implementation."""
        pass
    
    def test_connection(self):
        """Mock connection test."""
        return self._make_request("/test", method="GET", use_cache=False)
    
    def _get_headers(self):
        """Mock headers."""
        return {}


class TestBaseAPIClient:
    """Test base API client functionality."""
    
    def test_circuit_breaker_429(self):
        """Test circuit breaker activation on 429 errors."""
        client = MockAPIClient()
        class_name = client.__class__.__name__
        
        # Reset circuit breaker
        BaseAPIClient._circuit_breaker_until = {}
        
        # Mock 429 response
        with patch('requests.Session.get') as mock_get:
            import requests
            mock_response = Mock()
            mock_response.status_code = 429
            http_error = requests.exceptions.HTTPError("429 Rate Limit")
            http_error.response = mock_response
            mock_response.raise_for_status.side_effect = http_error
            mock_get.return_value = mock_response
            
            # First call should trigger circuit breaker
            try:
                client._make_request("/test", method="GET", use_cache=False)
            except:
                pass
            
            # Check circuit breaker is set
            assert class_name in BaseAPIClient._circuit_breaker_until
            circuit_breaker_time = BaseAPIClient._circuit_breaker_until[class_name]
            assert circuit_breaker_time > time.time()
    
    def test_circuit_breaker_prevents_requests(self):
        """Test that circuit breaker prevents requests during cooldown."""
        client = MockAPIClient()
        class_name = client.__class__.__name__
        
        # Reset circuit breaker first
        BaseAPIClient._circuit_breaker_until = {}
        
        # Set circuit breaker to future time
        BaseAPIClient._circuit_breaker_until[class_name] = time.time() + 120
        
        with patch('requests.Session.get') as mock_get:
            try:
                client._make_request("/test", method="GET", use_cache=False)
            except Exception as e:
                # Should raise exception about circuit breaker
                assert "circuit breaker" in str(e).lower() or "rate limit" in str(e).lower()
            
            # Should not have made actual request
            mock_get.assert_not_called()
    
    def test_caching_get_requests(self):
        """Test that GET requests are cached."""
        client = MockAPIClient()
        
        # Reset circuit breaker
        BaseAPIClient._circuit_breaker_until = {}
        
        with patch('src.data_sources.base_client.cache_manager') as mock_cache:
            mock_cache.enabled = True
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = None
            mock_cache.make_key.return_value = "test_cache_key"
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status.return_value = None
            
            with patch('requests.Session.get', return_value=mock_response):
                result = client._make_request("/test", method="GET", use_cache=True)
                
                # Should check cache
                mock_cache.get.assert_called()
                # Should set cache
                mock_cache.set.assert_called()
    
    def test_cache_bypass_for_connection_tests(self):
        """Test that connection tests bypass cache."""
        client = MockAPIClient()
        
        # Reset circuit breaker
        BaseAPIClient._circuit_breaker_until = {}
        
        with patch('src.data_sources.base_client.cache_manager') as mock_cache:
            mock_cache.enabled = True
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_response.raise_for_status.return_value = None
            
            with patch('requests.Session.get', return_value=mock_response):
                client._make_request("/test", method="GET", use_cache=False, force_refresh=True)
                
                # Should not use cache for connection tests
                mock_cache.get.assert_not_called()
    
    def test_rate_limiting(self):
        """Test minimum request interval."""
        client = MockAPIClient()
        client.min_request_interval = 0.1
        
        # Reset circuit breaker
        BaseAPIClient._circuit_breaker_until = {}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.Session.get', return_value=mock_response):
            start = time.time()
            client._make_request("/test1", method="GET", use_cache=False)
            client._make_request("/test2", method="GET", use_cache=False)
            elapsed = time.time() - start
            
            # Should have waited at least min_request_interval between requests
            assert elapsed >= client.min_request_interval


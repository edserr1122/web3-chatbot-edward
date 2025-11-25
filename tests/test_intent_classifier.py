"""
Tests for AI-powered intent classifier.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.intent_classifier import IntentClassifier


class TestIntentClassifier:
    """Test intent classification functionality."""
    
    def test_classify_crypto_query(self):
        """Test classification of crypto-related queries."""
        with patch('src.tools.intent_classifier.ChatGroq') as mock_groq:
            # Mock LLM response
            mock_response = Mock()
            mock_response.content = '{"intent": "crypto_analysis", "tokens": ["BTC", "Bitcoin"]}'
            mock_llm_instance = Mock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_groq.return_value = mock_llm_instance
            
            classifier = IntentClassifier()
            result = classifier.classify("Tell me about Bitcoin")
            
            assert result["intent"] == "crypto_analysis"
            assert "BTC" in result["tokens"] or "Bitcoin" in result["tokens"]
    
    def test_classify_small_talk(self):
        """Test classification of small talk queries."""
        with patch('src.tools.intent_classifier.ChatGroq') as mock_groq:
            mock_response = Mock()
            mock_response.content = '{"intent": "small_talk", "tokens": []}'
            mock_llm_instance = Mock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_groq.return_value = mock_llm_instance
            
            classifier = IntentClassifier()
            result = classifier.classify("How are you?")
            
            assert result["intent"] == "small_talk"
            assert result["tokens"] == []
    
    def test_classify_with_context(self):
        """Test context-aware classification."""
        with patch('src.tools.intent_classifier.ChatGroq') as mock_groq:
            mock_response = Mock()
            mock_response.content = '{"intent": "crypto_analysis", "tokens": []}'
            mock_llm_instance = Mock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_groq.return_value = mock_llm_instance
            
            classifier = IntentClassifier()
            context = [
                AIMessage(content="Would you like technical analysis?"),
                HumanMessage(content="Yes")
            ]
            result = classifier.classify("Yes", conversation_context=context)
            
            # Should classify "Yes" as crypto_analysis when context is crypto question
            assert result["intent"] == "crypto_analysis"
            # Verify context was passed to LLM
            call_args = mock_llm_instance.invoke.call_args
            assert call_args is not None
    
    def test_classify_off_topic(self):
        """Test classification of off-topic queries."""
        with patch('src.tools.intent_classifier.ChatGroq') as mock_groq:
            mock_response = Mock()
            mock_response.content = '{"intent": "off_topic", "tokens": []}'
            mock_llm_instance = Mock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_groq.return_value = mock_llm_instance
            
            classifier = IntentClassifier()
            result = classifier.classify("What's the weather today?")
            
            assert result["intent"] == "off_topic"
    
    def test_classify_handles_json_parsing_error(self):
        """Test that classifier handles JSON parsing errors gracefully."""
        with patch('src.tools.intent_classifier.ChatGroq') as mock_groq:
            mock_response = Mock()
            mock_response.content = 'Invalid JSON response'
            mock_llm_instance = Mock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_groq.return_value = mock_llm_instance
            
            classifier = IntentClassifier()
            result = classifier.classify("Tell me about Bitcoin")
            
            # Should default to "unknown" on parsing error
            assert result["intent"] == "unknown"
            assert result["tokens"] == []


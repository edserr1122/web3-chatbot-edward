"""
Intent classification tool using AI to detect user intent and extract crypto tokens.
Replaces keyword-based guardrails with intelligent AI-driven classification.
"""

from typing import Dict, Any, List, Optional
import logging
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import config

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    AI-powered intent classifier for crypto chatbot.
    Uses Groq LLM to intelligently classify user intent and extract crypto tokens.
    """
    
    CLASSIFICATION_PROMPT = """You are an intent classifier for a cryptocurrency analysis assistant.

Analyze the user's message and classify it into one of these categories:
1. **crypto_analysis**: Questions about cryptocurrency tokens, prices, markets, analysis, comparisons, etc.
2. **small_talk**: Greetings, polite conversation, "how are you", "thanks", etc.
3. **off_topic**: Questions completely unrelated to crypto (movies, weather, general knowledge, etc.)
4. **unknown**: If you're genuinely unsure

Additionally, extract any cryptocurrency token names or symbols mentioned (e.g., "BTC", "Bitcoin", "Ethereum", "SOL", "TAO", etc.).
Extract tokens even if they're mentioned casually or with typos.

Return ONLY valid JSON in this exact format:
{
  "intent": "crypto_analysis" | "small_talk" | "off_topic" | "unknown",
  "tokens": ["BTC", "ETH", ...]  // Array of token symbols/names detected (empty array if none)
}

Examples:
- "Analyze Bitcoin" → {"intent": "crypto_analysis", "tokens": ["BTC", "Bitcoin"]}
- "How are you?" → {"intent": "small_talk", "tokens": []}
- "What's the weather?" → {"intent": "off_topic", "tokens": []}
- "Compare BTC and ETH" → {"intent": "crypto_analysis", "tokens": ["BTC", "ETH"]}
- "Tell me about ApeCoin" → {"intent": "crypto_analysis", "tokens": ["APE", "ApeCoin"]}
- "What's your favorite coin?" → {"intent": "small_talk", "tokens": []}
- "Thanks for the help!" → {"intent": "small_talk", "tokens": []}
"""
    
    def __init__(self):
        """
        Initialize intent classifier with Groq LLM.
        Uses a smaller, faster model optimized for quick classification.
        """
        # Use dedicated intent classifier model (smaller/faster) or fallback to optimized main model
        model = getattr(config, 'GROQ_INTENT_CLASSIFIER_MODEL', None) or "llama-3.1-8b-instant"
        
        self.llm = ChatGroq(
            model=model,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=150,  # Limit tokens for faster response (classification is short JSON)
            api_key=config.GROQ_API_KEY
        )
        logger.info(f"✅ IntentClassifier initialized with Groq AI (model: {model}, optimized for speed)")
    
    def classify(self, user_input: str) -> Dict[str, Any]:
        """
        Classify user intent and extract crypto tokens.
        
        Args:
            user_input: User's message
            
        Returns:
            dict with keys:
                - intent: "crypto_analysis" | "small_talk" | "off_topic" | "unknown"
                - tokens: List of detected token symbols/names
        """
        try:
            messages = [
                SystemMessage(content=self.CLASSIFICATION_PROMPT),
                HumanMessage(content=f"User message: {user_input}")
            ]
            
            # Call LLM
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            # LLM might wrap JSON in markdown code blocks or add extra text
            json_text = self._extract_json(response_text)
            
            # Parse JSON
            result = json.loads(json_text)
            
            # Validate structure
            intent = result.get("intent", "unknown")
            tokens = result.get("tokens", [])
            
            # Normalize intent
            valid_intents = ["crypto_analysis", "small_talk", "off_topic", "unknown"]
            if intent not in valid_intents:
                logger.warning(f"Invalid intent '{intent}' from LLM, defaulting to 'unknown'")
                intent = "unknown"
            
            # Normalize tokens (uppercase, remove duplicates)
            tokens = list(set([t.upper().strip() for t in tokens if t and t.strip()]))
            
            logger.debug(f"Intent classification: intent={intent}, tokens={tokens}")
            
            return {
                "intent": intent,
                "tokens": tokens
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {response_text[:200]}... Error: {e}")
            return {
                "intent": "unknown",
                "tokens": []
            }
        except Exception as e:
            logger.error(f"Error in intent classification: {e}", exc_info=True)
            return {
                "intent": "unknown",
                "tokens": []
            }
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM response.
        Handles cases where LLM wraps JSON in markdown code blocks or adds extra text.
        
        Args:
            text: Raw LLM response
            
        Returns:
            JSON string
        """
        # Try to find JSON in markdown code blocks
        import re
        
        # Look for ```json ... ``` or ``` ... ```
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Look for JSON object directly
        json_match = re.search(r'\{.*?"intent".*?"tokens".*?\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return original text (will fail parsing, but we handle that)
        return text


# Singleton instance
_intent_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """
    Get or create singleton intent classifier instance.
    
    Returns:
        IntentClassifier instance
    """
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def classify_intent(user_input: str) -> Dict[str, Any]:
    """
    Convenience function to classify user intent.
    
    Args:
        user_input: User's message
        
    Returns:
        dict with "intent" and "tokens" keys
    """
    classifier = get_intent_classifier()
    return classifier.classify(user_input)


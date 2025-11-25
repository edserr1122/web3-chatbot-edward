"""
LangGraph-based crypto analysis agent.
Uses MemorySaver for conversation persistence and modular tool architecture.
"""

from typing import Dict, Any, List, TypedDict, Annotated, Optional
import logging
import re
import json
import uuid
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from src.analyzers import (
    FundamentalAnalyzer,
    PriceAnalyzer,
    TechnicalAnalyzer,
    SentimentAnalyzer,
    ComparativeAnalyzer,
)
from src.tools import (
    FundamentalTool,
    PriceTool,
    TechnicalTool,
    SentimentTool,
    ComparativeTool,
    create_fundamental_tool,
    create_price_tool,
    create_technical_tool,
    create_sentiment_tool,
    create_comparative_tool,
    create_full_analysis_tool,
    get_intent_classifier,
)
from src.utils import config, InputValidator
from src.memory import history_store

logger = logging.getLogger(__name__)


# Define agent state
class AgentState(TypedDict):
    """State for the crypto analysis agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str  # "crypto_analysis", "small_talk", "off_topic", "unknown"
    tokens: List[str]  # Extracted crypto token symbols/names
    revision_count: int  # Number of revision attempts
    evaluation_score: float  # Last evaluation score (0.0-1.0)
    evaluation_feedback: str  # Feedback from evaluator
    validator_attempts: int  # Number of validator fix attempts (to prevent infinite loops)


class CryptoAgent:
    """LangGraph-based agent for cryptocurrency analysis."""
    
    SYSTEM_PROMPT = """You are a professional cryptocurrency analyst assistant. Your role is to help users analyze cryptocurrency tokens and answer their questions about crypto markets.

**Your Capabilities:**
- Fundamental Analysis: Market cap, supply, volume, tokenomics
- Price Analysis: Historical trends, volatility, support/resistance
- Technical Analysis: RSI, MACD, moving averages, Bollinger Bands
- Sentiment Analysis: Social sentiment, news sentiment, Fear & Greed Index
- Comparative Analysis: Compare multiple tokens across all dimensions
- Full Analysis: Comprehensive multi-dimensional analysis

**Guidelines:**
1. Always be professional, accurate, and data-driven
2. Use the appropriate tool based on the user's question
3. Provide substantive, multi-paragraph responses with specific metrics
4. Reference previous analyses when relevant (you have conversation memory)
5. Stay focused on cryptocurrency - politely redirect off-topic requests
6. Ask for clarification when token symbols are unclear
7. Explain technical terms for non-expert users
8. Format numbers, percentages, and data clearly
9. For general "tell me about" questions, use full_analysis tool
10. For specific questions (price, sentiment, etc.), use the specific tool

**Tool Selection Strategy:**
- "Tell me about Bitcoin" → use full_analysis
- "What's Bitcoin's price?" → use analyze_price
- "Is Bitcoin bullish?" → use analyze_technical and analyze_sentiment
- "Compare Bitcoin and Ethereum" → use compare_tokens
- "Bitcoin's market cap?" → use analyze_fundamentals

Remember: Provide valuable, data-driven insights backed by real-time data."""
    
    def __init__(
        self,
        fundamental_analyzer: FundamentalAnalyzer,
        price_analyzer: PriceAnalyzer,
        technical_analyzer: TechnicalAnalyzer,
        sentiment_analyzer: SentimentAnalyzer,
        comparative_analyzer: ComparativeAnalyzer,
        user_id: str = "cli_user",
        session_id: Optional[str] = None,
    ):
        """
        Initialize the crypto agent.
        
        Args:
            fundamental_analyzer: FundamentalAnalyzer instance
            price_analyzer: PriceAnalyzer instance
            technical_analyzer: TechnicalAnalyzer instance
            sentiment_analyzer: SentimentAnalyzer instance
            comparative_analyzer: ComparativeAnalyzer instance
        """
        self.validator = InputValidator()
        self.user_id = user_id
        self.history_context_limit = config.HISTORY_CONTEXT_LIMIT
        
        # Initialize AI-powered intent classifier
        self.intent_classifier = get_intent_classifier()
        
        # Initialize LLM with higher max_tokens for comprehensive responses
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=0.7,
            max_tokens=4096,  # Higher limit for comprehensive analysis responses
            api_key=config.GROQ_API_KEY
        )
        
        # Initialize evaluator LLM (smaller, faster model for evaluation)
        self.evaluator_llm = ChatGroq(
            model=config.GROQ_INTENT_CLASSIFIER_MODEL,
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=200,  # Short evaluation responses
            api_key=config.GROQ_API_KEY
        )
        
        # Evaluation settings
        self.evaluation_threshold = config.EVALUATION_THRESHOLD
        self.max_revisions = config.MAX_REVISIONS
        self.max_validator_attempts = config.MAX_VALIDATOR_ATTEMPTS
        
        # Create tool instances
        self.fundamental_tool = FundamentalTool(fundamental_analyzer)
        self.price_tool = PriceTool(price_analyzer)
        self.technical_tool = TechnicalTool(technical_analyzer)
        self.sentiment_tool = SentimentTool(sentiment_analyzer)
        self.comparative_tool = ComparativeTool(comparative_analyzer)
        
        # Create LangChain tools
        self.tools = [
            create_fundamental_tool(fundamental_analyzer),
            create_price_tool(price_analyzer),
            create_technical_tool(technical_analyzer),
            create_sentiment_tool(sentiment_analyzer),
            create_comparative_tool(comparative_analyzer),
            create_full_analysis_tool(
                self.fundamental_tool,
                self.price_tool,
                self.technical_tool,
                self.sentiment_tool
            ),
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize memory (LangGraph's MemorySaver)
        self.memory = MemorySaver()
        
        # Build and compile graph
        self.graph = self._build_graph()
        
        # Thread config for conversation persistence
        self.thread_id = session_id or self._generate_session_id()
        self._ensure_history_session()
        
        logger.info("✅ CryptoAgent initialized with LangGraph MemorySaver")
    
    def _build_graph(self):
        """Build the LangGraph workflow with memory and intent-based routing."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("small_talk", self._small_talk_node)
        workflow.add_node("off_topic", self._off_topic_node)
        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("response_validator", self._response_validator_node)
        workflow.add_node("evaluator", self._evaluator_node)
        
        # Add edges
        workflow.set_entry_point("classify_intent")
        
        # Route based on intent classification
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "small_talk": "small_talk",
                "off_topic": "off_topic",
                "crypto_analysis": "agent",
                "unknown": "agent"  # Fallback to agent for unknown (will use guardrails)
            }
        )
        
        # Small talk and off-topic nodes also go through response validator
        workflow.add_edge("small_talk", "response_validator")
        workflow.add_edge("off_topic", "response_validator")
        
        # Agent routing (tools or response_validator)
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "response_validator"  # Route to validator before evaluator
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Response validator routing (valid → evaluator, invalid → agent to fix)
        workflow.add_conditional_edges(
            "response_validator",
            self._should_fix_response,
            {
                "valid": "evaluator",  # Response is complete, proceed to evaluation
                "invalid": "agent"  # Response has issues, fix it
            }
        )
        
        # Evaluator routing (end if score passes, or back to agent for revision)
        workflow.add_conditional_edges(
            "evaluator",
            self._should_revise,
            {
                "pass": END,  # Score >= threshold, return to user
                "revise": "agent"  # Score < threshold, revise
            }
        )
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)
    
    def _call_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Call the agent (LLM with tools) using full conversation history.
        
        Args:
            state: Current agent state (includes full conversation history from memory)
            
        Returns:
            Updated state
        """
        messages = state["messages"]
        
        # Log conversation history length for debugging
        logger.debug(f"Agent node received {len(messages)} messages from conversation history")
        
        # Limit conversation history to prevent token limit issues
        # Keep system message, last 20 messages (to maintain context but avoid token overflow)
        MAX_HISTORY_MESSAGES = 20
        if len(messages) > MAX_HISTORY_MESSAGES:
            logger.info(f"Conversation history too long ({len(messages)} messages), limiting to last {MAX_HISTORY_MESSAGES} messages")
            # Keep system message if present, then last N messages
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
            messages = system_msgs + non_system_msgs[-MAX_HISTORY_MESSAGES:]
        
        # Add system message if this is the first call (no system message in history)
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + messages
        
        # Call LLM with full conversation history - this includes all previous messages
        response = self.llm_with_tools.invoke(messages)

        print("response", response)
        
        # Check if response was truncated by examining response_metadata
        is_truncated = False
        finish_reason = None
        if hasattr(response, 'response_metadata'):
            finish_reason = response.response_metadata.get('finish_reason', '')
            # finish_reason can be: 'stop', 'length', 'tool_calls', etc.
            # 'length' means the response was truncated due to token limit
            if finish_reason == 'length':
                is_truncated = True
                logger.warning(f"Response was truncated due to token limit (finish_reason: {finish_reason})")
        
        # Log response length and check for truncation indicators
        if hasattr(response, 'content'):
            response_length = len(response.content) if response.content else 0
            logger.debug(f"LLM response length: {response_length} characters, finish_reason: {finish_reason}")
            
            # Only handle explicit truncation (finish_reason='length') - let AI-based validator handle semantic incompleteness
            if is_truncated:
                logger.warning(f"Response was truncated due to token limit (finish_reason: {finish_reason}, length: {response_length})")
                logger.info("Attempting to continue truncated response...")
                continued_response = self._continue_truncated_response(messages, response)
                if continued_response:
                    logger.info("Successfully continued response")
                    return {"messages": [continued_response]}
                else:
                    logger.warning("Failed to continue response, will be caught by response validator")
        
        return {"messages": [response]}
    
    def _classify_intent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Classify user intent using AI-powered intent classifier.
        This is a LangGraph node.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with intent and tokens
        """
        messages = state["messages"]
        if not messages:
            return {"intent": "unknown", "tokens": []}
        
        # Get the last user message
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            user_input = last_message.content
        else:
            return {"intent": "unknown", "tokens": []}
        
        # Extract conversation context (last 3 messages for context-aware classification)
        # This helps classify follow-up responses like "Yes" correctly
        conversation_context = messages[-3:] if len(messages) >= 3 else messages
        
        try:
            classification = self.intent_classifier.classify(user_input, conversation_context=conversation_context)
            intent = classification.get("intent", "unknown")
            tokens = classification.get("tokens", [])
            
            logger.info(f"🤖 AI Intent Classification: intent={intent}, tokens={tokens}")
            
            return {
                "intent": intent,
                "tokens": tokens
            }
        except Exception as e:
            logger.error(f"Error in AI intent classification: {e}", exc_info=True)
            return {"intent": "unknown", "tokens": []}
    
    def _small_talk_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle small talk using LLM with full conversation history.
        This is a LangGraph node.
        
        Args:
            state: Current agent state (includes full conversation history)
            
        Returns:
            Updated state with AI-generated small talk response
        """
        messages = state["messages"]
        if not messages:
            return {"messages": [AIMessage(content="I'm here to help!")]}
        
        try:
            # Use full conversation history for context-aware responses
            response_text = self._handle_small_talk_with_history(messages)
            return {"messages": [AIMessage(content=response_text)]}
        except Exception as e:
            logger.error(f"Error in small talk node: {e}", exc_info=True)
            return {"messages": [AIMessage(content="I'm here to help! Feel free to ask me about cryptocurrency analysis.")]}
    
    def _off_topic_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle off-topic requests using LLM with full conversation history.
        This is a LangGraph node.
        
        Args:
            state: Current agent state (includes full conversation history)
            
        Returns:
            Updated state with AI-generated off-topic response
        """
        messages = state.get("messages", [])
        try:
            # Use full conversation history for context-aware responses
            response_text = self._handle_off_topic_with_history(messages)
            return {"messages": [AIMessage(content=response_text)]}
        except Exception as e:
            logger.error(f"Error in off-topic node: {e}", exc_info=True)
            return {"messages": [AIMessage(content="I'm a cryptocurrency analysis assistant. Could you please ask me something related to cryptocurrency?")]}
    
    def _route_by_intent(self, state: AgentState) -> str:
        """
        Route to appropriate node based on classified intent.
        
        Args:
            state: Current agent state
            
        Returns:
            Node name to route to
        """
        intent = state.get("intent", "unknown")
        
        # For unknown intent, check fallback guardrails
        if intent == "unknown":
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, HumanMessage):
                    user_input = last_message.content
                    # Fallback to keyword-based detection
                    if self._is_small_talk(user_input):
                        return "small_talk"
                    if not self._is_relevant_query(user_input):
                        return "off_topic"
        
        return intent if intent in ["small_talk", "off_topic", "crypto_analysis"] else "crypto_analysis"
    
    def _continue_truncated_response(self, conversation_messages: List[BaseMessage], truncated_response) -> Any:
        """
        Continue a truncated LLM response by asking it to complete the thought.
        
        Args:
            conversation_messages: Full conversation history
            truncated_response: The truncated response message
            
        Returns:
            Continued response message or None if continuation fails
        """
        try:
            # Get the original truncated content
            original_content = truncated_response.content if hasattr(truncated_response, 'content') else ""
            if not original_content:
                logger.warning("Truncated response has no content to continue")
                return None
            
            # Limit conversation history for continuation to avoid token limits
            MAX_HISTORY_FOR_CONTINUATION = 15
            continuation_messages = conversation_messages.copy()
            if len(continuation_messages) > MAX_HISTORY_FOR_CONTINUATION:
                system_msgs = [m for m in continuation_messages if isinstance(m, SystemMessage)]
                non_system_msgs = [m for m in continuation_messages if not isinstance(m, SystemMessage)]
                continuation_messages = system_msgs + non_system_msgs[-MAX_HISTORY_FOR_CONTINUATION:]
            
            # Add the truncated response and continuation instruction
            continuation_messages.append(truncated_response)
            continuation_messages.append(
                HumanMessage(content="Your previous response was cut off. Please continue from where you left off and complete your thought. Do not repeat what you already said.")
            )
            
            # Call LLM to continue (use base LLM without tools for continuation)
            continued = self.llm.invoke(continuation_messages)
            
            # Combine the responses
            if hasattr(continued, 'content') and continued.content:
                # Merge the truncated and continued content
                continued_content = continued.content.strip()
                # Remove any repetition at the start of continuation
                if continued_content.lower().startswith(original_content[-50:].lower()):
                    # Continuation seems to repeat, just append
                    combined_content = original_content + " " + continued_content
                else:
                    # Normal continuation
                    combined_content = original_content + "\n\n" + continued_content
                
                # Create a new AIMessage with combined content
                from langchain_core.messages import AIMessage
                combined_response = AIMessage(content=combined_content)
                
                # Copy metadata if available
                if hasattr(truncated_response, 'response_metadata'):
                    combined_response.response_metadata = truncated_response.response_metadata.copy()
                    # Update finish_reason to indicate it was completed
                    if 'finish_reason' in combined_response.response_metadata:
                        combined_response.response_metadata['finish_reason'] = 'stop'
                
                logger.info(f"Successfully continued response. Original: {len(original_content)}, Continued: {len(continued_content)}, Combined: {len(combined_content)}")
                return combined_response
            
            logger.warning("Continuation response has no content")
            return None
            
        except Exception as e:
            logger.error(f"Error continuing truncated response: {e}", exc_info=True)
            return None
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if we should continue to tools or end.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "end"
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def _response_validator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate that the response is complete and properly formatted.
        Checks for raw tool call syntax, incomplete responses, and missing tool results.
        If issues are found, adds feedback message for agent to fix.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state (with feedback message if issues found)
        """
        messages = state["messages"]
        
        # Get the last AI message
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            logger.warning("No AI message found in response validator")
            return {}
        
        response_content = ""
        if hasattr(last_ai_message, 'content'):
            response_content = last_ai_message.content or ""
        
        # Check for critical issues
        has_raw_tool_syntax = "<function=" in response_content or "<function_calls>" in response_content
        
        # Check for unprocessed tool calls
        has_unprocessed_tool = False
        if "function=" in response_content and "{" in response_content:
            import re
            tool_call_pattern = r'<function[^>]*>\s*\{[^}]*"symbol"[^}]*\}'
            if re.search(tool_call_pattern, response_content):
                has_unprocessed_tool = True
        
        # Check for incomplete ending using AI-based semantic check
        has_incomplete_ending = False
        validator_attempts = state.get("validator_attempts", 0)
        
        # Only check for incomplete ending if we haven't exceeded max attempts (prevent infinite loops)
        if response_content and len(response_content.strip()) > 100 and validator_attempts < self.max_validator_attempts:
            # Use AI to check if response is semantically complete
            has_incomplete_ending = self._check_incomplete_with_ai(response_content)
        
        # If issues found, add feedback message
        if has_raw_tool_syntax or has_unprocessed_tool or has_incomplete_ending:
            issues = []
            if has_raw_tool_syntax:
                issues.append("raw_tool_syntax")
            if has_unprocessed_tool:
                issues.append("unprocessed_tool")
            if has_incomplete_ending:
                issues.append("incomplete_ending")
            
            logger.warning(f"Response validation failed: {', '.join(issues)}")
            
            feedback_parts = ["**RESPONSE VALIDATION FEEDBACK**:\n"]
            if has_raw_tool_syntax or has_unprocessed_tool:
                feedback_parts.append("Your previous response contained raw tool call syntax or incomplete formatting. ")
                feedback_parts.append("If you called tools, make sure to incorporate the tool results into your response ")
                feedback_parts.append("in a natural, readable format. Do not include raw function calls or JSON in your response.\n\n")
            if has_incomplete_ending:
                feedback_parts.append("Your previous response appears to be incomplete or cut off mid-sentence. ")
                feedback_parts.append("Please complete your response with a proper ending. Make sure to finish all thoughts and sentences.\n\n")
            
            feedback_parts.append("Please regenerate a complete, natural language response.")
            
            feedback_message = SystemMessage(content="".join(feedback_parts))
            return {
                "messages": [feedback_message],
                "validator_attempts": validator_attempts + 1
            }
        
        # Response is valid - reset validator attempts
        logger.debug("Response validation passed")
        return {"validator_attempts": 0}
    
    def _check_incomplete_with_ai(self, response_content: str) -> bool:
        """
        Use AI to check if response is semantically complete.
        This is more sophisticated than simple punctuation checks.
        
        Args:
            response_content: The response text to check
            
        Returns:
            True if response is incomplete, False if complete
        """
        try:
            # Get last 100 chars for context
            last_portion = response_content.strip()[-100:] if len(response_content.strip()) > 100 else response_content.strip()
            
            completeness_prompt = f"""You are a response completeness checker. Determine if the following response is semantically complete or cut off mid-sentence.

**Response ending (last 100 chars)**:
{last_portion}

**CRITICAL RULES**:
1. **Emojis ALWAYS indicate completeness**: If a response ends with ANY emoji (😊, 🍞, ✨, 🚀, 📈, 💰, ✅, ❌, etc.), it is ALWAYS COMPLETE. Emojis are valid sentence endings.
2. **Punctuation is not required**: Responses can end with emojis, punctuation, or natural sentence endings.
3. **Only flag if clearly cut off**: Only mark as INCOMPLETE if the response clearly cuts off mid-word, mid-sentence, or mid-thought.

**A response is COMPLETE if it**:
- Ends with ANY emoji (🚀, 😊, 🍞, ✨, 📈, etc.) - THIS IS ALWAYS COMPLETE
- Finishes all thoughts and sentences
- Has proper closure (period, exclamation, question mark, or natural ending)
- Ends with complete words (even without punctuation)
- Doesn't cut off mid-word or mid-thought

**A response is INCOMPLETE ONLY if it**:
- Cuts off mid-sentence (e.g., "7 **Finish the" or "knead the dough on a")
- Ends with incomplete words (e.g., "The process of mak" - word cut off)
- Clearly stops before finishing a thought (e.g., "The process of making bread involves..." with no ending)
- Appears truncated mid-word

**Examples**:
- "Feel free to clarify! 😊" → COMPLETE (emoji ending)
- "Making bread yourself is rewarding. 🍞✨" → COMPLETE (emoji ending)
- "Great analysis! 🚀" → COMPLETE (emoji ending)
- "Bitcoin is bullish 📈" → COMPLETE (emoji ending, no punctuation needed)
- "7 **Finish the" → INCOMPLETE (cuts off mid-sentence)
- "knead the dough on a" → INCOMPLETE (cuts off mid-sentence)
- "The process of mak" → INCOMPLETE (word cut off)

**IMPORTANT**: If the response ends with an emoji, it is ALWAYS COMPLETE. Do not flag it as incomplete.

Return ONLY a JSON object:
{{
  "is_complete": true or false,
  "reason": "brief explanation"
}}"""

            response = self.evaluator_llm.invoke([SystemMessage(content=completeness_prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_complete = result.get("is_complete", True)  # Default to complete if unsure
                reason = result.get("reason", "")
                
                # Double-check: if response ends with emoji, force it to be complete
                # This is a safety check in case the AI misinterprets
                import unicodedata
                last_char = response_content.strip()[-1] if response_content.strip() else ""
                # Check if last character is an emoji (Unicode emoji range)
                is_emoji = (
                    ord(last_char) >= 0x1F300 and ord(last_char) <= 0x1F9FF  # Emoticons, symbols, etc.
                    or ord(last_char) >= 0x1F600 and ord(last_char) <= 0x1F64F  # Emoticons
                    or ord(last_char) >= 0x2600 and ord(last_char) <= 0x26FF  # Miscellaneous symbols
                    or ord(last_char) >= 0x2700 and ord(last_char) <= 0x27BF  # Dingbats
                    or unicodedata.category(last_char) == 'So'  # Symbol, other (includes many emojis)
                )
                
                if is_emoji and not is_complete:
                    logger.warning(f"AI flagged emoji-ending response as incomplete, but emojis are always complete. Overriding. Reason: {reason}")
                    is_complete = True
                
                if not is_complete:
                    logger.warning(f"AI detected incomplete response: {reason}")
                else:
                    logger.debug(f"AI confirmed response is complete: {reason}")
                
                return not is_complete  # Return True if incomplete
            else:
                # If JSON parsing fails, fall back to simple check (but be lenient)
                logger.warning("Failed to parse AI completeness check, using fallback")
                # Fallback: check if ends with emoji - if so, always complete
                import unicodedata
                last_char = response_content.strip()[-1] if response_content.strip() else ""
                is_emoji = (
                    ord(last_char) >= 0x1F300 and ord(last_char) <= 0x1F9FF
                    or ord(last_char) >= 0x1F600 and ord(last_char) <= 0x1F64F
                    or ord(last_char) >= 0x2600 and ord(last_char) <= 0x26FF
                    or ord(last_char) >= 0x2700 and ord(last_char) <= 0x27BF
                    or unicodedata.category(last_char) == 'So'
                )
                
                if is_emoji:
                    logger.debug("Fallback: Response ends with emoji, marking as complete")
                    return False  # Not incomplete
                
                # Only flag if clearly cut off (ends with incomplete word pattern)
                last_word = response_content.strip().split()[-1] if response_content.strip().split() else ""
                # Only flag if last word is very short (< 3 chars) and doesn't end with punctuation
                if len(last_word) < 3 and not any(c in last_word for c in '.!?。！？'):
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error in AI completeness check: {e}", exc_info=True)
            # Fallback: be lenient, don't flag unless clearly cut off
            return False
    
    def _should_fix_response(self, state: AgentState) -> str:
        """
        Determine if response needs fixing or is valid.
        
        Args:
            state: Current agent state
            
        Returns:
            "valid" if response is good, "invalid" if needs fixing
        """
        messages = state["messages"]
        
        # Get the last AI message
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return "valid"  # No message to validate, proceed
        
        response_content = ""
        if hasattr(last_ai_message, 'content'):
            response_content = last_ai_message.content or ""
        
        # Check for critical issues that require fixing
        has_raw_tool = "<function=" in response_content or "<function_calls>" in response_content
        
        # Check for unprocessed tool calls
        has_unprocessed_tool = False
        if "function=" in response_content and "{" in response_content:
            import re
            tool_call_pattern = r'<function[^>]*>\s*\{[^}]*"symbol"[^}]*\}'
            if re.search(tool_call_pattern, response_content):
                has_unprocessed_tool = True
        
        # Check for incomplete ending using AI (only if we haven't exceeded max attempts)
        validator_attempts = state.get("validator_attempts", 0)
        has_incomplete_ending = False
        
        if response_content and len(response_content.strip()) > 100 and validator_attempts < self.max_validator_attempts:
            has_incomplete_ending = self._check_incomplete_with_ai(response_content)
        
        # If any critical issues, need to fix (but respect max attempts)
        if (has_raw_tool or has_unprocessed_tool or has_incomplete_ending) and validator_attempts < self.max_validator_attempts:
            issues = []
            if has_raw_tool:
                issues.append("raw_tool_syntax")
            if has_unprocessed_tool:
                issues.append("unprocessed_tool")
            if has_incomplete_ending:
                issues.append("incomplete_ending")
            logger.info(f"🔄 Response has critical issues ({', '.join(issues)}), routing back to agent to fix (attempt {validator_attempts + 1}/{self.max_validator_attempts})")
            return "invalid"
        
        # If max attempts reached, accept the response (prevent infinite loop)
        if validator_attempts >= self.max_validator_attempts:
            logger.warning(f"⚠️  Max validator attempts reached ({validator_attempts}), accepting response to prevent infinite loop")
            return "valid"
        
        # Response is valid, proceed to evaluator
        logger.debug("✅ Response validation passed, proceeding to evaluator")
        return "valid"
    
    def _evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluate the agent's response for quality (completeness, freshness, relevance).
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with evaluation_score and evaluation_feedback
        """
        messages = state["messages"]
        
        # Get the last AI message (the response to evaluate)
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message or not hasattr(last_ai_message, 'content') or not last_ai_message.content:
            logger.warning("No AI message found to evaluate, defaulting to passing")
            return {
                "evaluation_score": 1.0,
                "evaluation_feedback": "No response to evaluate"
            }
        
        response_content = last_ai_message.content
        user_query = ""
        intent = state.get("intent", "unknown")
        tokens = state.get("tokens", [])
        
        # Get the original user query
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # Evaluate using LLM
        evaluation_prompt = f"""You are a quality evaluator for cryptocurrency analysis responses.

Evaluate the following response based on three criteria:
1. **Completeness**: Is the response complete? Does it end properly without truncation?
2. **Freshness**: Is the data recent? (Check if response mentions timestamps or data age)
3. **Relevance**: Does the response address the user's query and intent?

**User Query**: {user_query}
**Intent**: {intent}
**Tokens Mentioned**: {', '.join(tokens) if tokens else 'None'}

**Response to Evaluate**:
{response_content}

**Instructions**:
- Return a JSON object with:
  - "completeness_score": 0.0-1.0 (1.0 = complete, 0.0 = incomplete/truncated)
  - "freshness_score": 0.0-1.0 (1.0 = very recent data, 0.5 = moderately old, 0.0 = stale)
  - "relevance_score": 0.0-1.0 (1.0 = highly relevant, 0.0 = not relevant)
  - "overall_score": 0.0-1.0 (weighted average: completeness 40%, freshness 30%, relevance 30%)
  - "feedback": Brief explanation of scores and any issues found

**Example Response**:
{{
  "completeness_score": 0.9,
  "freshness_score": 0.8,
  "relevance_score": 1.0,
  "overall_score": 0.91,
  "feedback": "Response is complete and highly relevant. Data appears recent (within last hour)."
}}

Return ONLY the JSON object, no additional text."""

        try:
            evaluation_response = self.evaluator_llm.invoke([SystemMessage(content=evaluation_prompt)])
            evaluation_text = evaluation_response.content if hasattr(evaluation_response, 'content') else str(evaluation_response)
            
            # Extract JSON from response
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', evaluation_text, re.DOTALL)
            if json_match:
                evaluation_json = json.loads(json_match.group())
            else:
                # Fallback: try parsing the whole response
                evaluation_json = json.loads(evaluation_text)
            
            overall_score = evaluation_json.get("overall_score", 0.5)
            feedback = evaluation_json.get("feedback", "Evaluation completed")
            
            overall_score_float = float(overall_score)
            logger.info(f"Evaluator score: {overall_score_float:.2f} (threshold: {self.evaluation_threshold})")
            logger.debug(f"Evaluator feedback: {feedback}")
            
            # Prepare state updates
            state_updates = {
                "evaluation_score": overall_score_float,
                "evaluation_feedback": feedback
            }
            
            # If revision is needed, add feedback as a system message and increment revision count
            if overall_score_float < self.evaluation_threshold:
                revision_count = state.get("revision_count", 0)
                if revision_count < self.max_revisions:
                    feedback_message = SystemMessage(
                        content=f"**EVALUATION FEEDBACK (Revision {revision_count + 1}/{self.max_revisions})**:\n"
                               f"Score: {overall_score_float:.2f} (threshold: {self.evaluation_threshold})\n"
                               f"Issues: {feedback}\n\n"
                               f"Please revise your previous response to address these issues. "
                               f"Focus on improving completeness, data freshness, and relevance to the user's query."
                    )
                    state_updates["messages"] = [feedback_message]
                    state_updates["revision_count"] = revision_count + 1
            
            return state_updates
            
        except Exception as e:
            logger.error(f"Error in evaluator: {e}", exc_info=True)
            # Default to passing if evaluation fails
            return {
                "evaluation_score": 1.0,
                "evaluation_feedback": f"Evaluation error: {str(e)}"
            }
    
    def _should_revise(self, state: AgentState) -> str:
        """
        Determine if we should revise the response or return it to the user.
        
        Args:
            state: Current agent state
            
        Returns:
            "pass" if score >= threshold, "revise" if score < threshold and revisions remaining
        """
        score = state.get("evaluation_score", 1.0)
        revision_count = state.get("revision_count", 0)
        
        # If score passes threshold, return to user
        if score >= self.evaluation_threshold:
            logger.info(f"✅ Response passed evaluation (score: {score:.2f} >= {self.evaluation_threshold})")
            return "pass"
        
        # If we've exceeded max revisions, return anyway
        if revision_count >= self.max_revisions:
            logger.warning(f"⚠️  Max revisions reached ({revision_count}), returning response despite low score ({score:.2f})")
            return "pass"
        
        # Need revision
        logger.info(f"🔄 Response needs revision (score: {score:.2f} < {self.evaluation_threshold}, revision {revision_count + 1}/{self.max_revisions})")
        return "revise"
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and return response.
        All routing (intent classification, small talk, off-topic, crypto analysis) 
        is now handled within the LangGraph workflow.
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response
        """
        try:
            # Validate and sanitize input
            user_input = self.validator.sanitize_input(user_input)
            
            # Check if exit command
            if self.validator.is_exit_command(user_input):
                return "Goodbye! Thanks for using the Crypto Analysis Chatbot."
            
            # Ensure history session exists
            self._ensure_history_session()
            
            # Build optional historical context
            context_text, context_count = self._build_history_context(user_input)
            augmented_input = self._augment_user_input(user_input, context_text)
            if context_count:
                logger.info(f"Adding {context_count} historical messages to current prompt context")
            
            # Create human message with augmented content (if any)
            human_message = HumanMessage(content=augmented_input)
            
            # Record user message in history store
            self._record_history_message("user", user_input)
            
            # Prepare state update - LangGraph's MemorySaver will automatically load previous state
            # and merge with this update. The add_messages reducer will append new messages to existing ones.
            state_update = {
                "messages": [human_message],  # add_messages reducer will append to conversation history
                "intent": "unknown",  # Will be set by classify_intent node
                "tokens": [],  # Will be set by classify_intent node
                "revision_count": 0,  # Track revision attempts
                "evaluation_score": 0.0,  # Last evaluation score
                "evaluation_feedback": "",  # Feedback from evaluator
                "validator_attempts": 0  # Track validator fix attempts
            }
            
            # Run the graph with memory - conversation history is automatically loaded from checkpointer
            # and merged with the update. Previous messages are preserved.
            config_dict = {"configurable": {"thread_id": self.thread_id}}
            final_state = self.graph.invoke(state_update, config_dict)
            
            # Extract response - get the last AI message
            response_message = final_state["messages"][-1]
            
            # Handle different message types
            if hasattr(response_message, 'content'):
                response_content = response_message.content
                if not response_content:
                    # If content is empty, try to get from tool calls or other fields
                    logger.warning("Response message has empty content, checking for alternative content")
                    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                        logger.warning("Response contains tool calls but no content - this shouldn't happen at end of graph")
                    # Fallback: try to find the last non-empty AI message
                    for msg in reversed(final_state["messages"]):
                        if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                            response_content = msg.content
                            logger.info(f"Using earlier AI message with content (length: {len(response_content)})")
                            break
                
                # Log response length for debugging
                if response_content:
                    logger.debug(f"Final response length: {len(response_content)} characters")
                    # Note: Completeness is now checked by AI-based validator, not here
                    
                    # Extract evaluation metadata from final state
                    evaluation_metadata = {}
                    if "evaluation_score" in final_state:
                        evaluation_metadata["evaluation_score"] = final_state.get("evaluation_score", 0.0)
                    if "evaluation_feedback" in final_state:
                        evaluation_metadata["evaluation_feedback"] = final_state.get("evaluation_feedback", "")
                    if "revision_count" in final_state:
                        evaluation_metadata["revision_count"] = final_state.get("revision_count", 0)
                    
                    self._record_history_message("assistant", response_content, additional_metadata=evaluation_metadata if evaluation_metadata else None)
                    return response_content
                
                fallback_msg = "I apologize, but I couldn't generate a response. Please try again."
                self._record_history_message("assistant", fallback_msg)
                return fallback_msg
            else:
                logger.error(f"Unexpected response message type: {type(response_message)}")
                fallback_msg = "I apologize, but I encountered an error processing the response. Please try again."
                self._record_history_message("assistant", fallback_msg)
                return fallback_msg
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    
    def _is_relevant_query(self, user_input: str) -> bool:
        """
        Check if query is relevant to crypto analysis (guardrail).
        
        Args:
            user_input: User's input
            
        Returns:
            True if relevant, False otherwise
        """
        # Check if crypto-related
        if self.validator.is_crypto_related(user_input):
            return True
        
        # Allow general greetings and meta questions
        user_lower = user_input.lower()
        greetings = [
            "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
            "help", "what can you do", "how does this work", "thanks", "thank you"
        ]
        if any(phrase in user_lower for phrase in greetings):
            return True
        
        # Check if referencing previous conversation
        if any(word in user_lower for word in ["earlier", "before", "previous", "you said", "you mentioned"]):
            return True
        
        return False
    
    def _is_small_talk(self, user_input: str) -> bool:
        """Detect simple small-talk or courtesy messages."""
        user_lower = user_input.lower().strip()
        small_talk_phrases = [
            "how are you", "how's it going", "what's up", "are you there",
            "who are you", "what are you", "thanks", "thank you", "appreciate it",
            "good morning", "good evening", "good night", "nice to meet you"
        ]
        return any(phrase in user_lower for phrase in small_talk_phrases)
    
    def _handle_small_talk(self, user_input: str) -> str:
        """
        Provide a friendly, natural response to general small talk using LLM.
        This replaces the hardcoded responses with AI-driven conversation.
        (Legacy method - use _handle_small_talk_with_history for context-aware responses)
        """
        try:
            # Use the intent classifier's LLM (smaller, faster model) for small talk
            small_talk_llm = ChatGroq(
                model=getattr(config, 'GROQ_INTENT_CLASSIFIER_MODEL', None) or "llama-3.1-8b-instant",
                temperature=0.7,  # Slightly higher for more natural conversation
                max_tokens=200,  # Limit for quick responses
                api_key=config.GROQ_API_KEY
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly and professional cryptocurrency analysis assistant. 
The user is engaging in small talk or casual conversation. Respond naturally, warmly, and briefly. 
Keep your response concise (1-2 sentences), friendly, and gently guide the conversation toward cryptocurrency topics if appropriate.
Do not be overly formal or robotic. Be conversational and human-like."""),
                ("human", "{user_input}")
            ])
            
            chain = prompt | small_talk_llm
            response = chain.invoke({"user_input": user_input})
            
            # Extract content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating small talk response: {e}", exc_info=True)
            # Fallback to a simple friendly response if LLM fails
            return (
                "I'm here to help! Feel free to ask me about any cryptocurrency, market trends, "
                "or analysis you'd like to explore."
            )
    
    def _handle_small_talk_with_history(self, messages: List[BaseMessage]) -> str:
        """
        Provide a friendly, natural response to small talk using LLM with full conversation history.
        
        Args:
            messages: Full conversation history including previous messages
            
        Returns:
            AI-generated response
        """
        try:
            # Use the intent classifier's LLM (smaller, faster model) for small talk
            small_talk_llm = ChatGroq(
                model=getattr(config, 'GROQ_INTENT_CLASSIFIER_MODEL', None) or "llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=200,
                api_key=config.GROQ_API_KEY
            )
            
            # Build messages list with system prompt and conversation history
            conversation_messages = [
                SystemMessage(content="""You are a friendly and professional cryptocurrency analysis assistant. 
The user is engaging in small talk or casual conversation. Respond naturally, warmly, and briefly. 
Keep your response concise (1-2 sentences), friendly, and gently guide the conversation toward cryptocurrency topics if appropriate.
Do not be overly formal or robotic. Be conversational and human-like. Use the conversation history to provide context-aware responses.""")
            ]
            
            # Add conversation history (last 10 messages to avoid token limits)
            conversation_messages.extend(messages[-10:])
            
            # Call LLM with full conversation context
            response = small_talk_llm.invoke(conversation_messages)
            
            # Extract content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating small talk response with history: {e}", exc_info=True)
            return (
                "I'm here to help! Feel free to ask me about any cryptocurrency, market trends, "
                "or analysis you'd like to explore."
            )
    
    def _handle_off_topic(self) -> str:
        """
        Handle off-topic requests using LLM for natural, contextual responses.
        (Legacy method - use _handle_off_topic_with_history for context-aware responses)
        """
        try:
            # Use the intent classifier's LLM (smaller, faster model) for off-topic handling
            off_topic_llm = ChatGroq(
                model=getattr(config, 'GROQ_INTENT_CLASSIFIER_MODEL', None) or "llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=250,
                api_key=config.GROQ_API_KEY
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a cryptocurrency analysis assistant. The user has asked something that's not related to cryptocurrency.

Respond politely and naturally. Briefly explain that you specialize in cryptocurrency analysis, mention what you can help with (token analysis, price trends, technical indicators, sentiment, comparisons), and gently redirect them to crypto-related questions.

Keep it friendly, concise (2-3 sentences), and professional. Don't be dismissive or rude."""),
                ("human", "The user asked something off-topic. Please respond naturally.")
            ])
            
            chain = prompt | off_topic_llm
            response = chain.invoke({})
            
            # Extract content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating off-topic response: {e}", exc_info=True)
            # Fallback response
            return (
                "I'm a cryptocurrency analysis assistant focused on helping you analyze crypto tokens and markets. "
                "I can help with comprehensive token analysis, price trends, technical indicators, sentiment, and comparisons. "
                "Could you please ask me something related to cryptocurrency analysis?"
            )
    
    def _handle_off_topic_with_history(self, messages: List[BaseMessage]) -> str:
        """
        Handle off-topic requests using LLM with full conversation history.
        
        Args:
            messages: Full conversation history including previous messages
            
        Returns:
            AI-generated response
        """
        try:
            # Use the intent classifier's LLM (smaller, faster model) for off-topic handling
            off_topic_llm = ChatGroq(
                model=getattr(config, 'GROQ_INTENT_CLASSIFIER_MODEL', None) or "llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=250,
                api_key=config.GROQ_API_KEY
            )
            
            # Build messages list with system prompt and conversation history
            conversation_messages = [
                SystemMessage(content="""You are a cryptocurrency analysis assistant. The user has asked something that's not related to cryptocurrency.

Respond politely and naturally. Briefly explain that you specialize in cryptocurrency analysis, mention what you can help with (token analysis, price trends, technical indicators, sentiment, comparisons), and gently redirect them to crypto-related questions.

Keep it friendly, concise (2-3 sentences), and professional. Don't be dismissive or rude. Use the conversation history to provide context-aware responses.""")
            ]
            
            # Add conversation history (last 10 messages to avoid token limits)
            conversation_messages.extend(messages[-10:])
            
            # Call LLM with full conversation context
            response = off_topic_llm.invoke(conversation_messages)
            
            # Extract content from the response
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating off-topic response with history: {e}", exc_info=True)
            return (
                "I'm a cryptocurrency analysis assistant focused on helping you analyze crypto tokens and markets. "
                "I can help with comprehensive token analysis, price trends, technical indicators, sentiment, and comparisons. "
                "Could you please ask me something related to cryptocurrency analysis?"
            )
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """
        Get conversation history from memory.
        
        Returns:
            List of messages
        """
        try:
            config_dict = {"configurable": {"thread_id": self.thread_id}}
            state = self.graph.get_state(config_dict)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def _ensure_history_session(self):
        """Ensure the current session exists in the history store."""
        try:
            history_store.create_session(self.thread_id, self.user_id)
        except Exception as e:
            logger.warning(f"Unable to ensure history session: {e}")

    def _record_history_message(self, role: str, content: str, additional_metadata: Optional[Dict[str, Any]] = None):
        """Persist a message to SQLite history with metadata summary and optional additional metadata."""
        if not content:
            return
        try:
            metadata = {"summary": self._summarize_for_history(content)}
            # Add additional metadata (e.g., evaluation scores)
            if additional_metadata:
                metadata.update(additional_metadata)
            history_store.append_message(
                session_id=self.thread_id,
                user_id=self.user_id,
                role=role,
                content=content,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to record message history: {e}")

    @staticmethod
    def _summarize_for_history(content: str) -> str:
        """Create a short summary for metadata storage."""
        cleaned = " ".join(content.strip().split())
        if len(cleaned) <= 160:
            return cleaned
        return f"{cleaned[:157]}..."

    def _build_history_context(self, user_input: str) -> (str, int):
        """Return formatted historical context and count based on relative time query."""
        since_ts = self._parse_relative_time_query(user_input)
        if not since_ts:
            return "", 0
        records = history_store.get_messages_since(
            user_id=self.user_id,
            since_ts=since_ts,
            limit=self.history_context_limit,
        )
        if not records:
            return "", 0
        formatted = self._format_history_context(records)
        return formatted, len(records)

    def _augment_user_input(self, user_input: str, context_text: str) -> str:
        """Append historical context snippet to the user input."""
        if not context_text:
            return user_input
        return (
            f"{user_input}\n\n"
            "[Relevant earlier conversation]\n"
            f"{context_text}"
        )

    def _parse_relative_time_query(self, user_input: str) -> Optional[int]:
        """Parse phrases like 'yesterday' or '40 minutes ago' and return start timestamp."""
        text = user_input.lower()
        now = datetime.utcnow()

        if "yesterday" in text:
            start = now - timedelta(days=1)
            return int(start.timestamp())

        if "last week" in text:
            start = now - timedelta(days=7)
            return int(start.timestamp())

        time_match = re.search(r"(\d+)\s+(minute|hour|day|week)s?\s+ago", text)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            delta_map = {
                "minute": timedelta(minutes=value),
                "hour": timedelta(hours=value),
                "day": timedelta(days=value),
                "week": timedelta(weeks=value),
            }
            start = now - delta_map.get(unit, timedelta())
            return int(start.timestamp())

        return None

    @staticmethod
    def _format_history_context(records: List[Dict[str, Any]]) -> str:
        """Format history records into a readable snippet."""
        lines = []
        for record in records:
            timestamp = datetime.utcfromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
            role = record["role"].capitalize()
            summary = record["metadata"].get("summary") if isinstance(record["metadata"], dict) else None
            summary = summary or record["content"]
            lines.append(f"{timestamp} - {role}: {summary}")
        return "\n".join(lines)

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a new session identifier."""
        return f"session_{uuid.uuid4().hex[:8]}"
    
    def clear_memory(self):
        """Clear conversation memory for current thread."""
        try:
            # Create new thread ID to start fresh
            self.thread_id = self._generate_session_id()
            logger.info(f"Memory cleared - new thread: {self.thread_id}")
            self._ensure_history_session()
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
    
    def set_session(self, session_id: str):
        """
        Set the session/thread ID for conversation isolation.
        
        Args:
            session_id: Session identifier
        """
        self.thread_id = session_id
        logger.info(f"Session set to: {session_id}")
        self._ensure_history_session()

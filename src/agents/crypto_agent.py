"""
LangGraph-based crypto analysis agent.
Uses MemorySaver for conversation persistence and modular tool architecture.
"""

from typing import Dict, Any, List, TypedDict, Annotated
import logging
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
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
)
from src.utils import config, InputValidator

logger = logging.getLogger(__name__)


# Define agent state
class AgentState(TypedDict):
    """State for the crypto analysis agent."""
    messages: Annotated[List[BaseMessage], add_messages]


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
        comparative_analyzer: ComparativeAnalyzer
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
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=0.7,
            api_key=config.GROQ_API_KEY
        )
        
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
        self.thread_id = "default_session"
        
        logger.info("✅ CryptoAgent initialized with LangGraph MemorySaver")
    
    def _build_graph(self):
        """Build the LangGraph workflow with memory."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)
    
    def _call_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Call the agent (LLM with tools).
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        messages = state["messages"]
        
        # Add system message if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + messages
        
        # Call LLM
        response = self.llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
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
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and return response.
        
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
            
            # Check if crypto-related (guardrail)
            if not self._is_relevant_query(user_input):
                return self._handle_off_topic()
            
            # Create human message
            human_message = HumanMessage(content=user_input)
            
            # Initialize state
            initial_state = {"messages": [human_message]}
            
            # Run the graph with memory
            config_dict = {"configurable": {"thread_id": self.thread_id}}
            final_state = self.graph.invoke(initial_state, config_dict)
            
            # Extract response
            response_message = final_state["messages"][-1]
            response_content = response_message.content
            
            return response_content
            
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
        greetings = ["hello", "hi", "hey", "help", "what can you do", "how does this work"]
        user_lower = user_input.lower()
        
        if any(greeting in user_lower for greeting in greetings):
            return True
        
        # Check if referencing previous conversation
        if any(word in user_lower for word in ["earlier", "before", "previous", "you said", "you mentioned"]):
            return True
        
        return False
    
    def _handle_off_topic(self) -> str:
        """Handle off-topic requests."""
        return """I'm a cryptocurrency analysis assistant focused on helping you analyze crypto tokens and markets. 

I can help you with:
- Comprehensive token analysis (Bitcoin, Ethereum, Solana, etc.)
- Price trends and volatility analysis
- Technical indicators (RSI, MACD, moving averages)
- Market sentiment and social metrics
- Comparing different cryptocurrencies

Could you please ask me something related to cryptocurrency analysis?"""
    
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
    
    def clear_memory(self):
        """Clear conversation memory for current thread."""
        try:
            # Create new thread ID to start fresh
            import uuid
            self.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            logger.info(f"Memory cleared - new thread: {self.thread_id}")
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

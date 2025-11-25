# Architecture Documentation

## System Overview

The Crypto Token Analysis Chat Agent is built on **LangGraph** for stateful agent workflows, with a modular architecture that separates data sources, analyzers, and agent logic. The system uses AI-powered intent classification, quality evaluation, and persistent conversation memory.

## LangGraph Agent Architecture

### Workflow Graph

The agent uses a **stateful graph workflow** with the following nodes and routing:

```
User Input
    ↓
classify_intent (AI-powered intent classification)
    ↓
    ├─→ small_talk → response_validator → evaluator → END
    ├─→ off_topic → response_validator → evaluator → END
    └─→ crypto_analysis/unknown → agent
                                    ↓
                                [tool calls?]
                                    ↓
                            Yes → tools → agent (loop)
                            No  → response_validator
                                    ↓
                            [valid?]
                                    ↓
                            Yes → evaluator
                            No  → agent (fix) → response_validator (loop)
                                    ↓
                            [score >= threshold?]
                                    ↓
                            Yes → END (return to user)
                            No  → agent (revise) → evaluator (loop, max 2 attempts)
```

### Key Components

1. **Intent Classifier Node**: Uses Groq LLM to classify user intent (`crypto_analysis`, `small_talk`, `off_topic`, `unknown`) with conversation context awareness
2. **Agent Node**: Main LLM with tool access, handles crypto analysis queries
3. **Tool Node**: Executes LangChain tools (fundamental, price, technical, sentiment, comparative analysis)
4. **Response Validator Node**: AI-based completeness check (detects incomplete responses, raw tool syntax)
5. **Evaluator Node**: Quality scoring (completeness, freshness, relevance) with automatic revision

### State Management

**AgentState** (TypedDict) tracks:

- `messages`: Conversation history (managed by LangGraph's `add_messages` reducer)
- `intent`: Classified intent from AI
- `tokens`: Extracted crypto token symbols
- `revision_count`: Revision attempts (prevents infinite loops)
- `evaluation_score`: Quality score from evaluator
- `validator_attempts`: Response fix attempts

**LangGraph MemorySaver**: Persists state across turns within a session using thread IDs.

## Ambiguity Handling

### Token Symbol Resolution

1. **AI-Powered Extraction**: Intent classifier extracts token symbols from user queries using LLM (not hardcoded lists)
2. **Symbol Mapping**: Data sources maintain symbol-to-ID mappings (e.g., "BTC" → "bitcoin", "ETH" → "ethereum")
3. **LLM Clarification**: When ambiguous, the agent's system prompt instructs it to ask for clarification:
   - "Ask for clarification when token symbols are unclear"
   - Agent naturally handles ambiguous queries like "Tell me about ETH" vs "Tell me about ETC"

### Query Intent Ambiguity

- **Context-Aware Classification**: Intent classifier uses last 3 messages for context (e.g., "Yes" after crypto question → `crypto_analysis`)
- **Autonomous Tool Selection**: Agent intelligently selects analysis types based on query (e.g., "price?" → price tool, "sentiment?" → sentiment tool)
- **Fallback Handling**: Unknown intents route to agent with specialization paragraph to handle edge cases

## Guardrails Implementation

### AI-Powered Guardrails (Not Rule-Based)

1. **Intent Classification**:

   - Uses Groq LLM to classify intent dynamically
   - Context-aware (considers conversation history)
   - Extracts tokens using AI (handles any token, not just predefined list)

2. **Off-Topic Handling**:

   - `off_topic` node uses LLM to generate natural, context-aware responses
   - Dynamically generates specialization paragraph explaining crypto focus
   - Polite redirection without being dismissive

3. **Response Quality Guardrails**:
   - **Response Validator**: AI-based completeness check (not simple punctuation rules)
   - **Evaluator**: Quality scoring with automatic revision if score < threshold
   - **Loop Prevention**: Max attempts (2 revisions, 2 validator fixes) to prevent infinite loops

### Guardrail Flow

```
User Query → Intent Classifier (AI)
    ↓
[crypto_analysis] → Agent (with tools)
[off_topic] → Off-topic Handler (LLM-generated response)
[small_talk] → Small Talk Handler (LLM-generated response)
[unknown] → Agent (with specialization paragraph)
```

## Cross-Session Context Persistence

### Dual Memory System

1. **LangGraph MemorySaver** (In-Memory):

   - Persists state within a session using thread IDs
   - Stores conversation history for multi-turn context
   - Session-scoped (cleared when session ends)

2. **SQLite History Store** (Persistent):
   - **Sessions Table**: `id`, `user_id`, `started_at`, `last_active_at`
   - **Messages Table**: `session_id`, `user_id`, `role`, `content`, `metadata`, `timestamp`
   - **Metadata**: Stores evaluation scores, revision counts, message summaries

### Cross-Session Retrieval

- **Temporal Queries**: Users can reference past sessions with relative time ("What did you say about Bitcoin yesterday?")
- **Session Resumption**: `--session <session_id>` flag resumes previous conversations
- **History Context Building**: Agent retrieves relevant historical messages based on:
  - Relative time parsing ("yesterday", "40 minutes ago")
  - Session ID lookup
  - Message metadata filtering

## Data Source Architecture

### Multi-Source Fallback Chain

Each analyzer implements fallback chains for robustness:

- **OHLC Data**: Binance Global → Binance.US → CoinGecko → CoinCap
- **Price Data**: CoinGecko → CoinCap → Binance
- **Market Data**: CoinMarketCap → CoinGecko → Messari

### Circuit Breaker Pattern

- **Rate Limiting (429)**: 120s cooldown, prevents API abuse
- **Server Errors (5xx)**: 30s cooldown, handles temporary failures
- **Geo-Restrictions (451)**: Logged but doesn't trigger circuit breaker

### Caching Strategy

- **Redis Caching**: API responses cached with TTL (default 300s)
- **Cache Keys**: Structured as `api_cache:{Client}:{Method}:{Endpoint}:{params}`
- **Connection Tests**: Bypass cache (always fresh)
- **Cache Logging**: Hit/miss/set/delete logged at INFO level

## Quality Assurance

### Response Validation

1. **Syntax Check**: Detects raw tool call syntax, unprocessed tool calls
2. **Completeness Check**: AI-based semantic completeness (not simple punctuation rules)
3. **Emoji Handling**: Recognizes emoji endings as valid (🚀, 😊, etc.)

### Evaluation Framework

- **Scoring Criteria**:
  - Crypto queries: Completeness (40%), Freshness (30%), Relevance (30%)
  - Off-topic queries: Completeness (50%), Helpfulness (50%)
- **Automatic Revision**: If score < threshold (0.7), agent revises with feedback
- **Max Revisions**: 2 attempts to prevent infinite loops

## Key Design Decisions

1. **LangGraph over LangChain Agents**: Better state management and explicit workflow control
2. **AI-Powered Guardrails**: Dynamic, context-aware instead of hardcoded rules
3. **Dual Memory System**: LangGraph for session state, SQLite for cross-session persistence
4. **Multi-Source Fallbacks**: Robust data retrieval with automatic failover
5. **Quality Evaluation Loop**: Self-improving responses through evaluation and revision

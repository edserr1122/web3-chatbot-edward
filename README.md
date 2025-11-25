# Crypto Token Analysis Chat Agent

An AI-powered conversational agent for comprehensive cryptocurrency token analysis with multi-turn conversation support, real-time data integration, and intelligent analysis across multiple dimensions.

## Features

### Core Requirements ✅

- **Multi-turn Conversational Interface**: CLI-based chat interface with context-aware responses
- **Multiple Data Sources**: Integrated with CoinGecko, CoinMarketCap, Binance, CoinCap, CryptoPanic, Fear & Greed Index, and Messari
- **5 Analysis Types** (exceeds requirement of 3):
  - **Fundamental Analysis**: Market cap, supply, volume, tokenomics, liquidity metrics
  - **Price Analysis**: Historical trends, volatility, support/resistance levels
  - **Sentiment Analysis**: Social sentiment, news sentiment, Fear & Greed Index
  - **Technical Analysis**: RSI, MACD, moving averages, Bollinger Bands
  - **Comparative Analysis**: Side-by-side comparison of multiple tokens
- **Conversation Memory**: SQLite-backed history with session persistence and cross-session retrieval
- **Intelligent Guardrails**: AI-powered intent classification to keep conversations focused on cryptocurrency

### Bonus Features 🚀

- **Autonomous Mode**: Agent intelligently selects relevant analysis types based on user intent
- **Analysis Memory & Retrieval**: Reference previous analyses with temporal queries ("What did you say about Bitcoin yesterday?")
- **Evaluation Framework**: Quality scoring system with automatic response revision
- **Memory Persistence**: Conversation history saved across sessions with metadata

## Architecture

- **LangGraph**: Stateful agent workflow with intent-based routing
- **LangChain**: Tool integration and LLM orchestration
- **Multi-source Fallback**: Robust data retrieval with automatic fallback chains
- **Redis Caching**: API response caching to reduce redundant calls
- **SQLite Storage**: Persistent conversation history with session management

## Setup

### Prerequisites

- Python 3.9+
- Redis (optional, for caching)
- API keys for data sources (see `.env.example`)

### Installation

```bash
# Clone the repository
git clone https://github.com/edserr1122/web3-chatbot-edward.git
cd web3-chatbot-edward

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Add your API keys to .env
# Required: GROQ_API_KEY, COINMARKETCAP_API_KEY, COINCAP_API_KEY, etc.
# Optional: REDIS_URL, etc.
```

### Running

```bash
# Production mode (clean console output)
python main.py

# Verbose mode (detailed logs)
python main.py --verbose

# Resume a previous session
python main.py --session <session_id>
```

## Data Sources

- **CoinGecko**: Price data, market metrics, historical data
- **CoinMarketCap**: Market cap, rankings, quotes
- **Binance/Binance.US**: OHLC data, technical indicators
- **CoinCap**: Real-time pricing, historical data
- **CryptoPanic**: News sentiment, social metrics
- **Fear & Greed Index**: Market sentiment indicators
- **Messari**: Fundamental data, tokenomics

## Project Structure

```
src/
├── agents/          # LangGraph agent workflow
├── analyzers/       # Analysis type implementations
├── data_sources/    # API client integrations
├── memory/          # Conversation history & caching
├── tools/           # LangGraph tools for agent
├── utils/           # Configuration & logging
└── chatbot.py       # Main chatbot class that can be used in any type of interfaces

interfaces/
└── cli.py           # Command-line interface

main.py              # Entry point
```

## Configuration

Key environment variables (see `.env.example` for full list):

- `GROQ_API_KEY`: Required - LLM provider API key
- `COINMARKETCAP_API_KEY`: Required - For CoinMarketCap data
- `COINCAP_API_KEY`: Required - For CoinCap data
- `REDIS_URL`: Optional - For response caching
- `HISTORY_DB_PATH`: Optional - SQLite database path (default: `data/chat_history.db`)

## Testing

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_intent_classifier.py

# Run with verbose output
pytest tests/ -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=html
```

### Test Files

- **test_intent_classifier.py**: Tests AI-powered intent classification
- **test_history_store.py**: Tests SQLite conversation history persistence
- **test_base_client.py**: Tests API client base functionality (circuit breaker, caching)
- **test_fundamental_analyzer.py**: Tests fundamental analysis logic

**Note**: Tests use mocks to avoid actual API calls. History store tests use temporary databases.

## Notes

- All API keys should be stored in `.env` file (not committed to repository)
- Redis is optional - falls back to in-memory caching if unavailable
- Logs are written to `chatbot.log` (production mode shows only errors in console)
- Conversation history is stored in SQLite database for persistence

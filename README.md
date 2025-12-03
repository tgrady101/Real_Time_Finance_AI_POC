# Real-Time Financial Markets Intelligence Chatbot

S&P 500-focused AI chatbot with multi-agent architecture for comprehensive financial market analysis.

## Features

- **Multi-Agent Architecture**: Root Orchestrator with 5 specialized sub-agents (Market Data, Economic, Headlines, Portfolio, Data Store)
- **Dynamic Model Routing**: Automatic selection between gemini-2.5-flash (simple) and gemini-3-pro-preview (complex)
- **Real-Time Data**: Yahoo Finance MCP Server for stock prices and fundamentals
- **Economic Data**: FRED API integration for GDP, inflation, unemployment, interest rates
- **News Headlines**: Google Search integration for real-time company news
- **Portfolio Analysis**: Risk metrics, allocation breakdown, rebalancing recommendations
- **S&P 500 Scope**: 500 companies with ticker validation and fuzzy matching
- **Persistent Memory**: PostgreSQL-backed cross-session recall with `load_memory` tool
- **Earnings Call RAG**: Vertex AI Data Store for Q2/Q3 2025 S&P 500 earnings transcripts with citations
- **Observability**: Arize AX for LLM tracing + 25-evaluator test suite 

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Google Cloud Project with billing enabled
- Terraform installed (for Cloud SQL memory storage)
- Optional: Arize API key for observability

### 2. Setup Infrastructure

```bash
# Provision GCP resources with Terraform
cd terraform
terraform init
terraform plan
terraform apply
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings:
# - GOOGLE_CLOUD_PROJECT (required)
# - ARIZE_API_KEY, ARIZE_SPACE_ID (optional)
```

### 5. Run the Chatbot

```bash
cd finance_agent
python main.py
# API at http://localhost:8080
# Docs at http://localhost:8080/docs
# Chat UI at http://localhost:8080/chat-ui
```

## Architecture

```
Root Orchestrator (dynamic model routing)
├── Market Data Agent (Yahoo Finance MCP) - Stock prices, financials, options
├── Economic Agent (FRED API) - GDP, inflation, unemployment, rates
├── Headlines Agent (Google Search) - News, earnings, sentiment
├── Portfolio Agent (yfinance + numpy) - Risk, allocation, rebalancing
├── Data Store Agent (Vertex AI RAG) - Earnings call transcripts with citations
├── S&P 500 Validation Tools
└── Memory Tools (load_memory for cross-session recall)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive architecture documentation.

## Example Queries

**Market Data:**
- "What's Apple's current stock price?"
- "Compare AAPL and MSFT stock performance"
- "What is the P/E ratio for Tesla?"

**Economic Data:**
- "What's the current GDP growth rate?"
- "What's the inflation rate right now?"
- "Show me unemployment data"

**News Headlines:**
- "Get Tesla news"
- "What are analysts saying about Apple?"

**Portfolio Analysis:**
- "Analyze my portfolio: 100 AAPL, 50 MSFT, 200 GOOGL"
- "What's the risk profile of my holdings?"

**Earnings Call RAG:**
- "What did Apple's CEO say about AI in the Q3 2025 earnings call?"
- "What are Microsoft's revenue growth projections?"
- "Summarize Tesla's Q2 2025 earnings highlights"

**Memory Recall:**
- "What questions have I asked in the past?"

## Project Structure

```
Real_Time_Finance_AI_POC/
├── finance_agent/          # Primary application code
│   ├── main.py             # FastAPI server (port 8080)
│   ├── Dockerfile          # Cloud Run container
│   ├── mcp_servers/        # Bundled Yahoo Finance MCP
│   └── workflow_1/
│       ├── agents/         # Root + Market Data agents
│       ├── api_clients/    # FRED API client
│       ├── mcp_clients/    # Yahoo Finance MCP client
│       ├── memory/         # PostgresMemoryService
│       ├── arize_observability/
│       └── utils/          # S&P 500 validation, model routing
├── terraform/              # Cloud SQL, IAM, storage
├── docs/                   # Architecture documentation
└── tests/                  # Test suite
```

## Deployment

```bash
# Deploy to Cloud Run
python deploy_cloud_run.py --deploy

# View logs
python deploy_cloud_run.py --logs
```

## License

MIT

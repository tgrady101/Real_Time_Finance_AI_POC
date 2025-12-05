# Real-Time Financial Markets Intelligence Chatbot

S&P 500-focused AI chatbot with multi-agent architecture for comprehensive financial market analysis.

## Features

- **Multi-Agent Architecture**: Root Orchestrator with 6 specialized sub-agents (Market Data, Economic, Headlines, Portfolio, Vector Store, Utility)
- **Dynamic Model Routing**: Automatic selection between gemini-2.5-flash (simple) and gemini-3-pro-preview (complex)
- **Real-Time Data**: Yahoo Finance MCP Server for stock prices and fundamentals
- **Economic Data**: FRED API integration for GDP, inflation, unemployment, interest rates
- **News Headlines**: Google Search integration for real-time company news
- **Portfolio Analysis**: Risk metrics, allocation breakdown, rebalancing recommendations
- **S&P 500 Scope**: 500 companies with ticker validation and fuzzy matching
- **Persistent Memory**: PostgreSQL-backed cross-session recall with `load_memory` tool
- **Hybrid Vector Search**: Vertex AI Vector Search with 3072-dim dense embeddings (gemini-embedding-001) + BM25 sparse embeddings, RRF fusion (α=0.5)
- **Earnings Call RAG**: 70,846 chunks from 503 S&P 500 companies (Q2/Q3 2025 transcripts) with citations
- **Crash-Resistant Ingestion**: Modular pipeline with per-chunk checkpointing, atomic writes, parallel I/O
- **Observability**: Arize AX for LLM tracing + 15-evaluator test suite (9 L1 automated + 6 L3 LLM-as-Judge) 

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
├── Economic Agent (FRED API) - GDP, inflation, unemployment, rates, historical trends
├── Headlines Agent (Google Search) - News, earnings, sentiment
├── Portfolio Agent (yfinance + numpy) - Risk, allocation, rebalancing
├── Vector Store Agent (Vertex AI Vector Search) - Hybrid dense+sparse search with RRF
│   └── 70,846 earnings call chunks (503 companies, Q2/Q3 2025)
├── Utility Agent - S&P 500 ticker validation, company lookup
└── Memory Tools (load_memory for cross-session recall)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive architecture documentation.

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

**Earnings Call RAG (Hybrid Search):**
- "What did Apple's CEO say about AI in the Q3 2025 earnings call?"
- "Summarize Tesla's Q2 2025 earnings highlights"
- "Which companies mentioned supply chain challenges?"
- "Compare guidance from NVDA and AMD earnings calls"

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
│       ├── agents/         # Root + 6 specialized sub-agents
│       ├── api_clients/    # FRED API client (with date range support)
│       ├── mcp_clients/    # Yahoo Finance MCP client
│       ├── memory/         # PostgresMemoryService
│       ├── arize_observability/  # 15 evaluators (9 L1 + 6 L3)
│       └── utils/          # S&P 500 validation, model routing, RAG
├── ingestion/              # Hybrid vector search ingestion pipeline
│   ├── run_pipeline.py     # Main orchestrator (6 stages)
│   ├── pipeline/           # Modular components
│   │   ├── config.py       # Pipeline configuration
│   │   ├── checkpoint.py   # Crash-resistant checkpointing
│   │   ├── embeddings.py   # Dense embeddings (gemini-embedding-001)
│   │   ├── bm25.py         # Sparse BM25 encoder
│   │   ├── chunker.py      # Text chunking
│   │   └── export.py       # JSONL export + GCS upload
│   ├── embeddings_cache/   # Per-chunk embedding storage
│   └── downloaded_earnings_calls/  # 979 transcript files
├── terraform/              # Cloud SQL, Vector Search, GCS
├── ARCHITECTURE.md         # Comprehensive architecture documentation
└── pyproject.toml          # Python project configuration
```

## Deployment

```bash
# Deploy to Cloud Run
python deploy_cloud_run.py --deploy

# View logs
python deploy_cloud_run.py --logs
```


# Real-Time Finance AI - Copilot Instructions

## Project Overview
S&P 500 financial chatbot using **Google ADK** (Agent Development Kit) with multi-agent architecture, dynamic model routing, FastAPI REST API, and Arize observability. Production-ready on Cloud Run.

## Quick Reference

### Development Setup
```powershell
# ALWAYS activate venv first (from project root)
& .\.venv\Scripts\Activate.ps1

# Run locally (from finance_agent/)
cd finance_agent
python main.py
# API: http://localhost:8080 | Docs: /docs | Chat UI: /chat-ui

# Deploy to Cloud Run
python deploy_cloud_run.py --deploy

# Run evaluations (25 evaluators, 96.2% pass rate)
python -m finance_agent.workflow_1.arize_observability.evaluations
```

### Key Directories
- `finance_agent/main.py` - FastAPI server entry point
- `finance_agent/workflow_1/agents/` - All agent definitions (root, market_data, economic, headlines, portfolio, data_store)
- `finance_agent/workflow_1/utils/` - Utilities (sp500_validator, query_classifier, dynamic_model_callback)
- `finance_agent/workflow_1/memory/` - PostgresMemoryService for cross-session recall
- `ingestion/` - Data ingestion scripts (sp500_earnings_ingestion.py)
- `terraform/` - Cloud SQL + Vertex AI Data Store infrastructure

## Architecture Patterns

### Agent Creation (MUST follow this pattern)
All agents use `before_model_callback` for dynamic model routing. See `root_agent.py` for the canonical example:

```python
from google.adk.agents import LlmAgent
from workflow_1.config import Config
from workflow_1.utils.dynamic_model_callback import create_dynamic_model_callback

def create_my_agent(use_dynamic_routing: bool = True):
    dynamic_callback = create_dynamic_model_callback(
        fast_model=Config.MODEL_FAST,      # gemini-2.5-flash
        complex_model=Config.MODEL_COMPLEX, # gemini-3-pro-preview
    ) if use_dynamic_routing else None
    
    return LlmAgent(
        model=Config.MODEL_FAST,
        name="my_agent",
        instruction="...",
        tools=[...],
        sub_agents=[...],  # Optional: for delegation
        before_model_callback=dynamic_callback,
    )
```

### Multi-Agent Delegation
Root agent delegates to 5 specialized sub-agents via `sub_agents` parameter:
- `market_data_agent` → Yahoo Finance MCP (stock prices, financials, options)
- `economic_agent` → FRED API (GDP, inflation, unemployment, rates)
- `headlines_agent` → Google Search built-in tool (news, headlines)
- `portfolio_agent` → yfinance + numpy (risk metrics, rebalancing)
- `data_store_agent` → Vertex AI Discovery Engine RAG (Q2/Q3 2025 earnings call transcripts with citations)

### MCP Tool Integration
Yahoo Finance uses Model Context Protocol via stdio:
```python
from workflow_1.mcp_clients.yahoo_finance import create_yahoo_finance_toolset
agent = LlmAgent(tools=[create_yahoo_finance_toolset()])
```

### S&P 500 Validation (REQUIRED for stock queries)
Always validate tickers before delegating to market agents:
```python
from workflow_1.utils.sp500_validator import (
    find_ticker_by_name,      # "Apple" → "AAPL"
    get_sp500_company_info,   # "AAPL" → company details
    is_valid_sp500_ticker,    # "AAPL" → True
)
```

### Earnings Call Ingestion
Ingest S&P 500 earnings call transcripts to Vertex AI Data Store (Discovery Engine auto-generates embeddings):
```powershell
# Set API key (API Ninjas Developer plan: $39/month)
$env:API_NINJAS_KEY = "your-api-key"

# Run ingestion for Q2/Q3 2025 (default)
cd ingestion
python sp500_earnings_ingestion.py

# Custom quarters/year
python sp500_earnings_ingestion.py --year 2025 --quarters 2,3

# Test with limited companies
python sp500_earnings_ingestion.py --limit 10 --skip-import
```
The ingestion script downloads transcripts, chunks them, and imports to Vertex AI where embeddings are automatically generated during indexing.

## Memory System

**Dual-service architecture:**
- `InMemorySessionService` - Ephemeral sessions (Arize logs them)
- `PostgresMemoryService` - Persistent memories for cross-session recall

**Auto-detection:** Uses Postgres on Cloud Run (`K_SERVICE` env), InMemory locally. Override with `MEMORY_STORAGE=postgres|memory|auto`.

**Key patterns in `main.py`:**
- Cookie-based user identification (`finance_user_token`, 1-year duration)
- `after_agent_callback` auto-saves to memory after each turn
- `load_memory` tool enables agent to recall past conversations

## Dynamic Model Routing

The `before_model_callback` in `dynamic_model_callback.py`:
1. Extracts user query from `llm_request.contents`
2. Classifies complexity via `query_classifier.py`
3. **Modifies** `llm_request.model` to selected model
4. Logs to Arize via OpenTelemetry span attributes

**Simple → `MODEL_FAST`:** Price lookups, greetings, short queries (<8 words)
**Complex → `MODEL_COMPLEX`:** Analysis, comparisons, multi-ticker, "why/how" questions

## Environment Variables

```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global  # For gemini-3-pro-preview

# Models (defaults shown)
MODEL_FAST=gemini-2.5-flash
MODEL_COMPLEX=gemini-3-pro-preview

# Memory (auto-detects Cloud Run vs local)
MEMORY_STORAGE=auto
SESSION_DB_URL=postgresql://...  # Required for postgres mode

# Data Store (earnings call RAG)
DATA_STORE_ID=earnings-call-datastore

# API Keys
API_NINJAS_KEY=...  # For earnings call ingestion ($39/month)
ARIZE_API_KEY=...
ARIZE_SPACE_ID=...
ARIZE_ENABLED=true
```

## Code Conventions
- **Line length:** 100 characters
- **Python:** 3.10+ required
- **Error handling:** Return user-friendly messages, never expose stack traces
- **Imports:** Use relative imports within `workflow_1/` package
- **Config:** Always use `Config` class from `workflow_1/config.py`, never hardcode values
- **Callbacks:** Use `before_model_callback` for routing, `after_agent_callback` for side effects

## Testing & Evaluation
Evaluations in `arize_observability/evaluations.py` - 25 evaluators across 47 test cases (96.2% pass rate) covering:
- Market data (ticker validation, price format, tool usage)
- Economic data format
- Headlines (news detection, source citation)
- Portfolio (risk metrics, allocation, rebalancing)
- Data store (RAG grounding, citation inclusion, earnings accuracy)
- Model routing and agent delegation accuracy

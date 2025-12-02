# Real-Time Finance AI - Copilot Instructions

## Project Overview
S&P 500-focused financial chatbot using Google ADK with a multi-agent architecture. Core implementation is complete with dynamic model routing, FastAPI REST API, and Arize observability.

## Current Status: ✅ Production Ready

### Implemented
- **Root Agent** with dynamic model routing via `before_model_callback`
- **Market Data Agent** with Yahoo Finance MCP integration
- **Economic Agent** with FRED API integration
- **Headlines Agent** with Google Search (built-in tool, no API key)
- **Portfolio Agent** for portfolio analysis, risk metrics, and rebalancing
- **FastAPI REST API** deployed to Cloud Run
- **Arize AX observability** with custom span attributes for model routing
- **S&P 500 validation** from Wikipedia with fuzzy matching
- **Persistent Memory** with PostgreSQL + `load_memory` tool for cross-session recall
- **Comprehensive Evaluation Suite**: 19 evaluators achieving 96.9% pass rate (39/39 tests)

### Cloud Run Deployment
- **URL**: https://finance-agent-277774606239.us-central1.run.app
- **Chat UI**: https://finance-agent-277774606239.us-central1.run.app/chat-ui
- **API Docs**: https://finance-agent-277774606239.us-central1.run.app/docs

### Future Work
- Data Store Agent (Vertex AI RAG for SEC filings)

## Architecture

### Primary Application Code: `finance_agent/`
```
finance_agent/
├── main.py                          # FastAPI server (port 8080)
├── requirements.txt                 # Cloud Run dependencies
├── Dockerfile                       # Container build
├── mcp_servers/
│   └── yahoo_finance_server.py      # Bundled MCP server
└── workflow_1/
    ├── config.py                    # Environment configuration
    ├── agents/
    │   ├── root_agent.py            # Root orchestrator
    │   ├── market_data_agent.py     # Yahoo Finance sub-agent
    │   ├── economic_agent.py        # FRED API sub-agent
    │   ├── headlines_agent.py       # Google Search sub-agent
    │   └── portfolio_agent.py       # Portfolio analysis sub-agent
    ├── api_clients/
    │   ├── fred_api.py              # FRED economic data
    │   └── news_api.py              # NewsAPI client (DEPRECATED - use headlines_agent)
    ├── mcp_clients/
    │   └── yahoo_finance.py         # MCP toolset wrapper
    ├── memory/
    │   └── postgres_memory.py       # PostgreSQL memory service
    ├── utils/
    │   ├── sp500_validator.py       # S&P 500 ticker validation
    │   ├── query_classifier.py      # Complexity classification
    │   └── dynamic_model_callback.py # Per-request model routing
    └── arize_observability/
        ├── observability.py         # Arize tracing setup
        └── evaluations.py           # LLM evaluations
```

### Terraform Infrastructure: `terraform/`
Cloud SQL PostgreSQL for memory storage. Run `terraform apply` from this directory.

## Dynamic Model Routing

The system uses `before_model_callback` to select models per-request:

```python
# Simple queries → gemini-2.5-flash (faster, cheaper)
# Complex queries → gemini-3-pro-preview (more capable)

from google.adk.agents import LlmAgent
from workflow_1.utils.dynamic_model_callback import create_dynamic_model_callback

agent = LlmAgent(
    model="gemini-2.5-flash",  # Default
    name="my_agent",
    before_model_callback=create_dynamic_model_callback(
        fast_model="gemini-2.5-flash",
        complex_model="gemini-3-pro-preview",
    ),
)
```

### Classification Logic (`query_classifier.py`)

**Simple Patterns** (→ MODEL_FAST):
- Price lookups, greetings, yes/no questions
- Short queries (< 8 words)

**Complex Patterns** (→ MODEL_COMPLEX):
- Analysis, comparisons, "why/how" questions
- Multiple tickers, multi-step requests

## Key Patterns

### Agent Creation Pattern
```python
from google.adk.agents import LlmAgent
from workflow_1.config import Config
from workflow_1.utils.dynamic_model_callback import create_dynamic_model_callback

def create_my_agent(use_dynamic_routing: bool = True):
    dynamic_callback = None
    if use_dynamic_routing:
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    return LlmAgent(
        model=Config.MODEL_FAST,
        name="my_agent",
        instruction="...",
        tools=[...],
        before_model_callback=dynamic_callback,
    )
```

### MCP Toolset Pattern
```python
from workflow_1.mcp_clients.yahoo_finance import create_yahoo_finance_toolset

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="market_data_agent",
    tools=[create_yahoo_finance_toolset()],
)
```

### S&P 500 Validation
```python
from workflow_1.utils.sp500_validator import (
    find_ticker_by_name,      # "Apple" → "AAPL"
    get_sp500_company_info,   # "AAPL" → company details
    is_valid_sp500_ticker,    # "AAPL" → True
)
```

## Environment Variables

```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global        # For Gemini 3 Pro Preview

# Model Configuration
MODEL_FAST=gemini-2.5-flash
MODEL_COMPLEX=gemini-3-pro-preview

# Memory Storage (auto-detects local vs Cloud Run)
MEMORY_STORAGE=auto                 # Default: Postgres on Cloud Run, InMemory locally
MEMORY_STORAGE=postgres             # PostgresMemoryService (requires SESSION_DB_URL)
MEMORY_STORAGE=memory               # InMemoryMemoryService (no persistence)
SESSION_DB_URL=postgresql://...     # PostgreSQL connection URL for memory storage

# Sessions are ephemeral (InMemorySessionService) - Arize handles logging
# Only agent memories are persisted for cross-session recall

# Optional APIs
FRED_API_KEY=...
# NEWS_API_KEY=... (DEPRECATED - headlines_agent uses Google Search)

# Arize Observability
ARIZE_API_KEY=...
ARIZE_SPACE_ID=...
ARIZE_PROJECT_NAME=finance-chatbot
ARIZE_ENABLED=true
```

## Development Commands

```powershell
# Always activate venv first (from project root)
& .\.venv\Scripts\Activate.ps1

# Local development
cd finance_agent
python main.py
# API at http://localhost:8080
# Docs at http://localhost:8080/docs
# Chat UI at http://localhost:8080/chat-ui

# Test query
$body = @{message="What is Apple's stock price?"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8080/chat" -Method POST -ContentType "application/json" -Body $body

# Deploy to Cloud Run
python deploy_cloud_run.py --deploy
```

**Important:** Always activate the virtual environment before running any Python commands:
```powershell
& C:\Users\tgrad\Projects\Real_Time_Finance_AI_POC\.venv\Scripts\Activate.ps1
```

## Arize Observability

The system traces model routing decisions with custom span attributes:

```python
model_routing.selected_model    # Model used
model_routing.complexity        # "simple" or "complex"
model_routing.confidence        # 0.0 to 1.0
model_routing.reason            # Classification explanation
```

View traces at: https://app.arize.com

### Evaluation Suite (19 Evaluators, 96.9% Pass Rate)

```bash
# Run evaluations locally
python -m finance_agent.workflow_1.arize_observability.evaluations

# Upload to Arize AX
python -m finance_agent.workflow_1.arize_observability.evaluations --arize --name "my_experiment"
```

**Evaluator Categories:**
- Market Data: `sp500_ticker_validation`, `company_name_resolution`, `stock_price_format`, `tool_usage_verification`
- Economic Data: `economic_data_format`
- Response Quality: `response_contains_data`, `financial_accuracy`, `response_completeness`
- Routing: `model_routing_accuracy`, `agent_delegation_accuracy`

## Memory System

The agent uses a dual-service architecture:
- **Sessions**: `InMemorySessionService` (ephemeral) - Arize handles session logging
- **Memories**: `PostgresMemoryService` (persistent) - For cross-session recall

### Memory Features
- `load_memory` tool: Agent can search past conversations (from `google.adk.tools`)
- `after_agent_callback`: Auto-saves conversation to memory after each turn
- Smart recall detection: Queries with "past", "history", "asked", "questions" return recent memories
- User-scoped: Each user only sees their own memories (filtered by cookie-based user_id)

### Memory Service Auto-Detection
```python
# Local development → InMemoryMemoryService (no persistence)
# Cloud Run → PostgresMemoryService (persists to Cloud SQL)

# Override with MEMORY_STORAGE environment variable
MEMORY_STORAGE=auto     # Default: detect environment
MEMORY_STORAGE=postgres # Force PostgreSQL
MEMORY_STORAGE=memory   # Force in-memory
```

### User Identification (Cookie-Based)
Users are identified via persistent HTTP cookies:
- Cookie name: `finance_user_token`
- Cookie duration: 1 year
- First request: UUID generated and stored in cookie
- Subsequent requests: Cookie read to identify user
- All memories are scoped to the user's cookie token

```python
# Server-side (automatic)
@app.post("/chat")
async def chat(
    request: ChatRequest,
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),  # Read cookie
):
    user_id = get_or_create_user_id(finance_user_token, response)  # Set if new
    ...

# Client-side (JavaScript)
fetch('/chat', {
    method: 'POST',
    credentials: 'include',  # Include cookies
    body: JSON.stringify({ message, session_id })
});
```

### Cloud SQL Setup (Terraform)
The `memories` table is auto-created by `PostgresMemoryService`.
Infrastructure managed in `terraform/cloudsql.tf`.

## Code Conventions
- **Primary code**: `finance_agent/workflow_1/`
- **Line length**: 100 characters
- **Python**: 3.10+ required
- **Error handling**: Return user-friendly messages
- **Model routing**: Use `before_model_callback` pattern

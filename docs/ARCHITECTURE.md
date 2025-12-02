# Real-Time Finance AI - Architecture

A production-ready S&P 500 financial chatbot built on Google's Agent Development Kit (ADK) with multi-agent orchestration, dynamic model routing, MCP tool integration, comprehensive evaluation suite, and cloud-native deployment.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Multi-Agent Design](#multi-agent-design)
- [Dynamic Model Routing](#dynamic-model-routing)
- [Tool Integration (MCP)](#tool-integration-mcp)
- [Memory System](#memory-system)
- [Observability & Evaluations (Arize)](#observability--evaluations-arize)
- [Deployment Architecture](#deployment-architecture)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)

---

## Overview

This system is an S&P 500-focused financial intelligence chatbot that leverages:

- **Google ADK (Agent Development Kit)**: Framework for building AI agents with LLM-based orchestration
- **Multi-Agent Architecture**: Root agent orchestrates specialized sub-agents (market_data_agent, economic_agent, headlines_agent, portfolio_agent)
- **Dynamic Model Routing**: `before_model_callback` selects optimal model per-request
- **MCP (Model Context Protocol)**: Yahoo Finance integration via stdio-based MCP server
- **Persistent Memory**: PostgreSQL-backed cross-session recall with `load_memory` tool
- **Arize AX Observability**: OpenTelemetry tracing + comprehensive evaluation suite with 19 evaluators (96.9% pass rate)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                                  │
│                    "What's Apple's stock price?"                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server (Cloud Run)                      │
│                         POST /chat endpoint                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ADK Runner                                  │
│                   Session + Memory Management                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Root Agent (finance_assistant)                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │             before_model_callback (Dynamic Routing)            │ │
│  │   • Classify query complexity (SIMPLE vs COMPLEX)              │ │
│  │   • Select model: gemini-2.5-flash OR gemini-3-pro-preview    │ │
│  │   • Log routing decision to Arize via OpenTelemetry           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Tools: find_ticker_by_name, get_sp500_company_info, load_memory    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Sub-Agent: market_data_agent                 ││
│  │  • Yahoo Finance MCP Toolset                                    ││
│  │  • Stock prices, news, financials, options                      ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Response to User                              │
│           "Apple (AAPL) is currently trading at $180.50"            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Agent Framework** | Google ADK | LlmAgent orchestration, tool management, callbacks |
| **LLM Models** | Gemini 2.5 Flash, Gemini 3 Pro Preview | Fast vs complex query handling |
| **API Server** | FastAPI | REST API for chat, streaming, health checks |
| **Tool Protocol** | MCP (Model Context Protocol) | Yahoo Finance market data integration |
| **Economic Data** | FRED API | GDP, inflation, unemployment, interest rates |
| **News/Headlines** | Google Search (built-in) | Recent company news, headlines, sentiment |
| **Portfolio Analysis** | yfinance + numpy | Risk metrics, allocation, performance |
| **Memory** | PostgreSQL + `load_memory` tool | Cross-session recall |
| **Sessions** | InMemorySessionService | Ephemeral per-request sessions |
| **Observability** | Arize AX + OpenTelemetry | Tracing, model routing analytics |
| **Evaluations** | Custom + LLM-as-a-Judge | 19 evaluators (39 tests, 96.9% pass rate) |
| **Deployment** | Google Cloud Run | Serverless container hosting |
| **Infrastructure** | Terraform | Cloud SQL provisioning |

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD RUN                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI Application                            │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  /chat       │  │  /chat/stream│  │  /health     │                │  │
│  │  │  (POST)      │  │  (SSE)       │  │  (GET)       │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │           │                │                                          │  │
│  │           └────────┬───────┘                                          │  │
│  │                    ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │                      ADK Runner                               │   │  │
│  │  │  • InMemorySessionService (ephemeral)                        │   │  │
│  │  │  • PostgresMemoryService (persistent)                        │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │              Root Agent (LlmAgent)                            │   │  │
│  │  │  before_model_callback → Dynamic Model Selection             │   │  │
│  │  │  after_agent_callback  → Memory Auto-Save                    │   │  │
│  │  │                                                               │   │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │   │  │
│  │  │  │              market_data_agent (Sub-Agent)              │  │   │  │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │   │  │
│  │  │  │  │            MCPToolset (Yahoo Finance)            │  │  │   │  │
│  │  │  │  │  • get_stock_info                                │  │  │   │  │
│  │  │  │  │  • get_historical_stock_prices                   │  │  │   │  │
│  │  │  │  │  • get_yahoo_finance_news                        │  │  │   │  │
│  │  │  │  │  • get_financial_statement                       │  │  │   │  │
│  │  │  │  │  • get_option_chain                              │  │  │   │  │
│  │  │  │  │  • ... (9 tools total)                           │  │  │   │  │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │   │  │
│  │  │  └──────────────────────────────────────────────────────┘  │   │  │
│  │  │                                                               │   │  │
│  │  │  ┌──────────────────────────────────────────────────────┐  │   │  │
│  │  │  │              headlines_agent (Sub-Agent)                 │  │   │  │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │   │  │
│  │  │  │  │           Google Search Tool (built-in)            │  │  │   │  │
│  │  │  │  │  • Recent company news                             │  │  │   │  │
│  │  │  │  │  • Breaking market headlines                       │  │  │   │  │
│  │  │  │  │  • Earnings news coverage                          │  │  │   │  │
│  │  │  │  │  • Sentiment from news                             │  │  │   │  │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │   │  │
│  │  │  └──────────────────────────────────────────────────────┘  │   │  │
│  │  │                                                               │   │  │
│  │  │  ┌──────────────────────────────────────────────────────┐  │   │  │
│  │  │  │              economic_agent (Sub-Agent)                 │  │   │  │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │   │  │
│  │  │  │  │                 FRED API Tools                    │  │  │   │  │
│  │  │  │  │  • get_gdp_data                                  │  │  │   │  │
│  │  │  │  │  • get_inflation_data                            │  │  │   │  │
│  │  │  │  │  • get_unemployment_data                         │  │  │   │  │
│  │  │  │  │  • get_interest_rate_data                        │  │  │   │  │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │   │  │
│  │  │  └────────────────────────────────────────────────────────┘  │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────┐  ┌────────────────────────────────────────┐ │
│  │  Yahoo Finance MCP Server  │  │  Arize AX Cloud (OpenTelemetry)       │ │
│  │  (Bundled Python Process)  │  │  • Model routing decisions            │ │
│  │  stdio communication       │  │  • Agent execution traces             │ │
│  └────────────────────────────┘  │  • Evaluation experiments             │ │
│                                   └────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CLOUD SQL (PostgreSQL)                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  memories table                                                        │  │
│  │  • app_name, user_id, session_id                                      │  │
│  │  • content (conversation text)                                         │  │
│  │  • summary (first 500 chars)                                          │  │
│  │  • created_at, metadata (JSONB)                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent Design

Based on Google ADK's **Multi-Agent Systems** architecture, the chatbot uses a hierarchical agent structure with LLM-based delegation.

### Agent Hierarchy

```python
from google.adk.agents import LlmAgent

# Root orchestrator - decides which sub-agent handles each query
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="finance_assistant",
    instruction="Coordinate specialized agents for S&P 500 analysis...",
    tools=[find_ticker_by_name, get_sp500_company_info, load_memory],
    sub_agents=[market_data_agent, headlines_agent, economic_agent, portfolio_agent],
    before_model_callback=dynamic_model_callback,
    after_agent_callback=auto_save_to_memory,
)

# Sub-agent for market data - uses MCP tools
market_data_agent = LlmAgent(
    model="gemini-3-pro-preview",
    name="market_data_agent",
    instruction="Retrieve real-time stock data via Yahoo Finance...",
    tools=[create_yahoo_finance_toolset()],  # MCPToolset
)

# Sub-agent for news/headlines - uses Google Search
headlines_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="headlines_agent",
    instruction="Find recent news and headlines for companies...",
    tools=[google_search],  # Built-in Google Search tool
)

# Sub-agent for economic data - uses FRED API
economic_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="economic_agent",
    instruction="Retrieve economic indicators from FRED API...",
    tools=[get_gdp_data, get_inflation_data, get_unemployment_data, get_interest_rate_data],
)
```

### Delegation Flow

1. **User Query** → Root Agent
2. Root Agent **validates ticker** using S&P 500 tools (for market queries)
3. Root Agent **delegates** to appropriate sub-agent:
   - `market_data_agent` for stock/company data
   - `economic_agent` for economic indicators
4. Sub-agent calls **external tools** (MCP/API)
5. Response flows back through the chain

### ADK Concepts Used

| Concept | Implementation | Purpose |
|---------|----------------|---------|
| **LlmAgent** | Root + sub-agents | LLM-powered reasoning and delegation |
| **Tools** | Python functions, MCPToolset | External capabilities |
| **before_model_callback** | `dynamic_model_selector()` | Intercept/modify LLM requests |
| **after_agent_callback** | `auto_save_to_memory()` | Post-turn actions |
| **sub_agents** | `[market_data_agent, economic_agent]` | Hierarchical delegation |

### Sub-Agents

| Agent | Status | Data Source | Capabilities |
|-------|--------|-------------|--------------|
| **market_data_agent** | ✅ Built | Yahoo Finance MCP | Stock prices, news, financials, options |
| **economic_agent** | ✅ Built | FRED API | GDP, inflation, unemployment, interest rates |
| **headlines_agent** | ✅ Built | Google Search | Recent news, headlines, earnings coverage, sentiment |
| **portfolio_agent** | ✅ Built | yfinance + numpy | Portfolio value, allocation, risk metrics, rebalancing |
| **data_store_agent** | 🔜 Planned | Vertex AI RAG | SEC filings, earnings calls, document search |

#### Economic Agent (Built)

```python
# workflow_1/agents/economic_agent.py

economic_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="economic_agent",
    instruction="""Retrieve economic indicators from FRED API.
    
    Capabilities:
    - GDP growth (quarterly/annual)
    - Inflation metrics (CPI)
    - Unemployment rates
    - Federal Reserve interest rates
    """,
    tools=[
        get_gdp_data,           # GDP growth data
        get_inflation_data,     # CPI inflation data
        get_unemployment_data,  # Unemployment rate
        get_interest_rate_data, # Fed funds rate
    ],
)
```

**API Client**: `workflow_1/api_clients/fred_api.py`

#### Headlines Agent (Built)

```python
# workflow_1/agents/headlines_agent.py

from google.adk.tools import google_search

headlines_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="headlines_agent",
    instruction="""Find recent news and headlines for companies.
    
    Capabilities:
    - Recent news headlines for specific companies
    - Breaking market news
    - Earnings announcements and coverage
    - Company press releases
    - Sentiment analysis from news coverage
    """,
    tools=[google_search],  # Built-in Google Search tool
)
```

**Data Source**: Google Search (built-in tool, no API key required)

#### Portfolio Agent (Built)

```python
# workflow_1/agents/portfolio_agent.py

portfolio_agent = LlmAgent(
    model="gemini-3-pro-preview",  # Complex analysis
    name="portfolio_agent",
    instruction="""Analyze user portfolios and provide insights.
    
    Capabilities:
    - Portfolio value calculation
    - Sector/industry allocation breakdown
    - Risk metrics (beta, volatility, Sharpe ratio)
    - Diversification analysis
    - Rebalancing recommendations
    - Performance vs benchmarks (SPY, QQQ)
    """,
    tools=[
        analyze_portfolio,
        calculate_portfolio_risk,
        get_portfolio_performance,
        suggest_rebalancing,
    ],
)
```

**Data Source**: yfinance for prices, numpy for calculations
```

#### Data Store Agent (Planned)

```python
# workflow_1/agents/data_store_agent.py (TO BE BUILT)

data_store_agent = LlmAgent(
    model="gemini-3-pro-preview",  # Complex document analysis
    name="data_store_agent",
    instruction="""Search and analyze SEC filings and earnings calls.
    
    Capabilities:
    - Search 10-K, 10-Q, 8-K filings
    - Query earnings call transcripts
    - Extract financial metrics from documents
    - Compare filings across quarters
    - Summarize management commentary
    """,
    tools=[
        # Vertex AI RAG Engine integration
        search_documents,     # Semantic search over filings
        get_filing,           # Retrieve specific SEC filing
        summarize_earnings,   # Summarize earnings call
    ],
)
```

**Infrastructure exists**: `terraform/data_store.tf` (Vertex AI RAG corpus)

#### Agent Hierarchy Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Root Agent (finance_assistant)                       │
│  Tools: find_ticker_by_name, get_sp500_company_info, load_memory        │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ market_data     │  │ headlines_agent │  │ economic_agent          │  │
│  │ _agent ✅       │  │ ✅ Google Search│  │ ✅ FRED API             │  │
│  │                 │  │                 │  │                         │  │
│  │ Yahoo Finance   │  │ Recent news     │  │ Interest rates          │  │
│  │ MCP Toolset     │  │ Headlines       │  │ Inflation               │  │
│  │                 │  │ Earnings        │  │ GDP, Employment         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────────────────────────────────┐   │
│  │ portfolio_agent │  │ data_store_agent                            │   │
│  │ ✅ yfinance     │  │ 🔜 Vertex AI RAG                            │   │
│  │                 │  │                                             │   │
│  │ Risk metrics    │  │ SEC 10-K, 10-Q filings                      │   │
│  │ Allocation      │  │ Earnings call transcripts                   │   │
│  │ Rebalancing     │  │ Document Q&A                                │   │
│  └─────────────────┘  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dynamic Model Routing

The system uses ADK's `before_model_callback` to dynamically select models based on query complexity **before** each LLM call.

### How It Works

```python
# workflow_1/utils/dynamic_model_callback.py

async def dynamic_model_selector(
    callback_context: CallbackContext, 
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    1. Extract user query from llm_request.contents
    2. Classify complexity (SIMPLE vs COMPLEX)
    3. Modify llm_request.model to selected model
    4. Return None to continue with modified request
    """
    user_query = extract_latest_user_message(llm_request)
    complexity, reason, confidence = classify_query(user_query)
    
    # Select model based on complexity
    selected_model = complex_model if complexity == COMPLEX else fast_model
    
    # MODIFY the request's model - this is the key part!
    llm_request.model = selected_model
    
    # Log to Arize via OpenTelemetry
    current_span.set_attribute("model_routing.selected_model", selected_model)
    current_span.set_attribute("model_routing.complexity", complexity.value)
    
    return None  # Continue with modified request
```

### Classification Logic

| Pattern | Complexity | Model |
|---------|------------|-------|
| Price lookups ("What's AAPL's price?") | SIMPLE | gemini-2.5-flash |
| Greetings ("Hello") | SIMPLE | gemini-2.5-flash |
| Short queries (< 8 words) | SIMPLE | gemini-2.5-flash |
| Analysis ("Why is AAPL down?") | COMPLEX | gemini-3-pro-preview |
| Comparisons ("Compare AAPL and GOOGL") | COMPLEX | gemini-3-pro-preview |
| Multi-ticker queries | COMPLEX | gemini-3-pro-preview |

### Observability

Every routing decision is traced to Arize:

```python
# Custom span attributes for Arize dashboard
current_span.set_attribute("model_routing.selected_model", "gemini-2.5-flash")
current_span.set_attribute("model_routing.complexity", "simple")
current_span.set_attribute("model_routing.confidence", 0.85)
current_span.set_attribute("model_routing.reason", "Price lookup pattern detected")
```

---

## Tool Integration (MCP)

Yahoo Finance data is integrated via **Model Context Protocol (MCP)**, using ADK's `MCPToolset` with stdio-based communication.

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     market_data_agent                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    MCPToolset                              │  │
│  │  connection_params=StdioConnectionParams(...)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                        stdio (stdin/stdout)                      │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           yahoo_finance_server.py (MCP Server)             │  │
│  │  • Bundled in container at /mcp_servers/                   │  │
│  │  • Uses yfinance library internally                        │  │
│  │  • Exposes 9 tools via MCP protocol                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `get_stock_info` | Price, metrics, company details |
| `get_historical_stock_prices` | OHLCV data (1d to max period) |
| `get_yahoo_finance_news` | Latest news articles |
| `get_stock_actions` | Dividends, stock splits |
| `get_financial_statement` | Income, balance sheet, cash flow |
| `get_holder_info` | Institutional/insider holdings |
| `get_option_expiration_dates` | Options expirations |
| `get_option_chain` | Calls/puts for expiration |
| `get_recommendations` | Analyst ratings |

### MCPToolset Configuration

```python
# workflow_1/mcp_clients/yahoo_finance.py

from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

def create_yahoo_finance_toolset():
    """Create MCPToolset for Yahoo Finance MCP server."""
    server_script = str(MCP_SERVER_PATH / "yahoo_finance_server.py")
    
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,  # Python interpreter
                args=[server_script],
            ),
            timeout=30.0,
        ),
    )
```

---

## Memory System

The system uses a **dual-service architecture** for session and memory management:

| Service | Type | Persistence | Purpose |
|---------|------|-------------|---------|
| **SessionService** | InMemorySessionService | Ephemeral | Per-request conversation state |
| **MemoryService** | PostgresMemoryService | Persistent | Cross-session recall |

### Why This Design?

- **Sessions are NOT persisted**: Arize handles logging/monitoring
- **Memories ARE persisted**: Agent can recall past conversations via `load_memory` tool
- **Auto-detect environment**: PostgreSQL on Cloud Run, InMemory locally

### User Identification (Cookie-Based)

Users are identified via **persistent HTTP cookies** for memory isolation:

| Setting | Value | Purpose |
|---------|-------|---------|
| **Cookie Name** | `finance_user_token` | Identifies returning users |
| **Max Age** | 1 year (86400 × 365s) | Long-term persistence |
| **httpOnly** | `true` | Security - not accessible via JS |
| **sameSite** | `lax` | Cross-site GET allowed |

```python
# Server-side (automatic in main.py)
from fastapi import Cookie, Response
import uuid

USER_COOKIE_NAME = "finance_user_token"
USER_COOKIE_MAX_AGE = 86400 * 365  # 1 year

def get_or_create_user_id(user_token: Optional[str], response: Response) -> str:
    """Get user ID from cookie or create a new one."""
    if user_token:
        return user_token
    
    new_token = str(uuid.uuid4())
    response.set_cookie(
        key=USER_COOKIE_NAME,
        value=new_token,
        max_age=USER_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return new_token

# All endpoints use this:
@app.post("/chat")
async def chat(
    request: ChatRequest,
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
):
    user_id = get_or_create_user_id(finance_user_token, response)
    ...
```

### Memory Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Turn                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                after_agent_callback                              │
│   async def auto_save_to_memory(callback_context):              │
│       session = callback_context.session                         │
│       await memory_service.add_session_to_memory(session)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgresMemoryService.add_session_to_memory()       │
│   • Extract conversation content from session events             │
│   • Store in PostgreSQL 'memories' table                        │
│   • Include app_name, user_id, session_id, content, summary     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Future Conversation                           │
│   Agent uses load_memory tool:                                   │
│   "What did I ask about Apple last time?"                       │
│                              │                                   │
│                              ▼                                   │
│   PostgresMemoryService.search_memory(query="Apple")            │
│   → Returns matching memories from past sessions                 │
└─────────────────────────────────────────────────────────────────┘
```

### PostgresMemoryService Schema

```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    app_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    content TEXT NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_memories_app_user ON memories(app_name, user_id);
```

### Environment-Aware Selection

```python
# main.py

async def get_memory_service():
    storage_type = os.getenv("MEMORY_STORAGE", "auto")
    
    if storage_type == "auto":
        # K_SERVICE is set by Cloud Run
        is_cloud_run = os.getenv("K_SERVICE") is not None
        storage_type = "postgres" if is_cloud_run else "memory"
    
    if storage_type == "postgres":
        from workflow_1.memory import PostgresMemoryService
        return PostgresMemoryService()  # Uses SESSION_DB_URL env var
    else:
        from google.adk.memory import InMemoryMemoryService
        return InMemoryMemoryService()
```

---

## Observability & Evaluations (Arize)

The system uses **Arize AX** (production ML observability platform) with **OpenTelemetry** for comprehensive tracing plus a **custom evaluation suite** with 10 evaluators.

### Auto-Instrumented Traces

```python
# workflow_1/arize_observability/observability.py

from arize.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

tracer_provider = register(
    space_id=ARIZE_SPACE_ID,
    api_key=ARIZE_API_KEY,
    project_name="finance-chatbot",
)

# Auto-instruments all ADK operations
GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
```

### What's Traced

| Category | Traces |
|----------|--------|
| **Agent Execution** | Root agent, sub-agent invocations |
| **Model Routing** | Selected model, complexity, confidence |
| **Tool Calls** | MCP tool invocations, API calls |
| **LLM Requests** | Model input/output, token usage |
| **Memory Operations** | save, search, recall |

### Custom Span Attributes

```python
# Added in dynamic_model_callback.py
current_span.set_attribute("model_routing.selected_model", "gemini-2.5-flash")
current_span.set_attribute("model_routing.complexity", "simple")
current_span.set_attribute("model_routing.confidence", 0.85)
current_span.set_attribute("model_routing.reason", "Price lookup pattern")
current_span.set_attribute("model_routing.agent", "finance_assistant")
```

### Evaluation Suite (10 Evaluators)

The system includes a comprehensive evaluation suite that can run locally or log to Arize AX for experiment tracking.

#### Evaluator Categories

| Category | Evaluators | Purpose |
|----------|-----------|---------|
| **Market Data** | `sp500_ticker_validation`, `company_name_resolution`, `stock_price_format`, `tool_usage_verification` | Validate S&P 500 queries |
| **Economic Data** | `economic_data_format` | Validate GDP, inflation, unemployment responses |
| **Response Quality** | `response_contains_data`, `financial_accuracy`, `response_completeness` | Overall response quality |
| **Routing & Delegation** | `model_routing_accuracy`, `agent_delegation_accuracy` | Model selection and agent routing |

#### Smart Filtering

Evaluators automatically apply based on query type:
- Market queries → Market Data + Quality + Routing evaluators
- Economic queries → Economic Data + Quality + Routing evaluators
- General queries → Quality + Routing evaluators only

#### Running Evaluations

```bash
# Local evaluation (default)
python -m finance_agent.workflow_1.arize_observability.evaluations

# Log to Arize AX (creates dataset + experiment)
python -m finance_agent.workflow_1.arize_observability.evaluations --arize --name "my_experiment"

# List existing datasets in Arize
python -m finance_agent.workflow_1.arize_observability.evaluations --list-datasets

# Focused evaluations
python -m finance_agent.workflow_1.arize_observability.evaluations --market
python -m finance_agent.workflow_1.arize_observability.evaluations --economic
python -m finance_agent.workflow_1.arize_observability.evaluations --routing
```

#### Current Test Results (96.9% Overall - All Tests Pass)

```
======================================================================
FINANCE AGENT EVALUATION REPORT
======================================================================

Total Tests: 39

--- Market Data Evaluators ---
✅ sp500_ticker_validation:    100.0%  (17/17)
✅ company_name_resolution:    100.0%  (17/17)
✅ stock_price_format:         100.0%  (17/17)
✅ tool_usage_verification:    100.0%  (17/17)

--- Economic Data Evaluators ---
✅ economic_data_format:       100.0%  (11/11)

--- Response Quality Evaluators ---
✅ response_contains_data:     100.0%  (39/39)
✅ financial_accuracy:         100.0%  (39/39)
✅ response_completeness:       98.7%  (38/39)

--- Routing & Delegation Evaluators ---
✅ model_routing_accuracy:     100.0%  (39/39)
✅ agent_delegation_accuracy:  100.0%  (39/39)

======================================================================
OVERALL SCORE: 96.9% ✅ PASS (0 FAILURES)
======================================================================
```

#### Arize AX Integration

Experiments appear in: **app.arize.com → Datasets & Experiments**

```python
# workflow_1/arize_observability/evaluations.py

from arize.experimental.datasets import ArizeDatasetsClient

# Create dataset and run experiment
client = ArizeDatasetsClient(api_key=ARIZE_API_KEY)
dataset_id = client.create_dataset(
    space_id=ARIZE_SPACE_ID,
    dataset_name="finance_agent_test",
    dataset_type=GENERATIVE,
    data=test_df,
)

experiment_id, results_df = client.run_experiment(
    space_id=ARIZE_SPACE_ID,
    dataset_id=dataset_id,
    task=task_function,
    evaluators=[sp500_ticker_validation, financial_accuracy, ...],
    experiment_name="eval_test_dec1",
)
```

### Arize Dashboard

View traces & experiments at: https://app.arize.com

---

## Deployment Architecture

### Cloud Run Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Run                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Container: finance-agent                                  │  │
│  │  Memory: 2Gi, CPU: 2, Timeout: 300s                       │  │
│  │                                                            │  │
│  │  Environment Variables:                                    │  │
│  │  • GOOGLE_CLOUD_PROJECT                                   │  │
│  │  • GOOGLE_CLOUD_LOCATION=global                           │  │
│  │  • SESSION_DB_URL=postgresql://...                        │  │
│  │  • ARIZE_API_KEY, ARIZE_SPACE_ID                         │  │
│  │  • MEMORY_STORAGE=auto                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  URL: https://finance-agent-XXXXX.us-central1.run.app          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud SQL (PostgreSQL 15)                     │
│  Instance: finance-agent-sessions                                │
│  IP: 34.41.65.51                                                │
│  Database: sessions                                              │
│  User: finance_agent                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Terraform Infrastructure

```hcl
# terraform/cloudsql.tf

resource "google_sql_database_instance" "sessions" {
  name             = "finance-agent-sessions"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier            = "db-f1-micro"
    disk_size       = 10
    disk_type       = "PD_SSD"
    disk_autoresize = false
  }
}

resource "google_sql_database" "sessions_db" {
  name     = "sessions"
  instance = google_sql_database_instance.sessions.name
}

resource "google_sql_user" "sessions_user" {
  name     = "finance_agent"
  instance = google_sql_database_instance.sessions.name
  password = var.db_password
}
```

### Deployment Command

```bash
cd finance_agent
gcloud run deploy finance-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=...,SESSION_DB_URL=..." \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300
```

---

## Data Flow

### Complete Request Flow

```
1. USER → FastAPI /chat endpoint
2. FastAPI → get_runner(user_id, session_id)
3. Runner → get_memory_service() [PostgresMemoryService on Cloud Run]
4. Runner → get_session_service() [InMemorySessionService]
5. Runner → get_agent(memory_service) [Root Agent with callbacks]
6. Runner.run_async() → Root Agent
7. Root Agent → before_model_callback (classify query, select model)
8. Root Agent → LLM call with selected model
9. LLM decides → delegate to market_data_agent
10. market_data_agent → MCPToolset → Yahoo Finance MCP Server
11. MCP Server → yfinance → Market data
12. Response flows back: MCP → Agent → Runner → FastAPI → USER
13. after_agent_callback → PostgresMemoryService.add_session_to_memory()
```

### Sequence Diagram

```
User        FastAPI      Runner       RootAgent    MarketAgent    MCP Server
  │            │            │             │             │             │
  │──POST /chat─►│            │             │             │             │
  │            │─►get_runner()│             │             │             │
  │            │            │─►run_async()──►│             │             │
  │            │            │             │─►before_model_callback     │
  │            │            │             │   (select gemini-2.5-flash)│
  │            │            │             │─►LLM call──────────────────►│
  │            │            │             │◄─delegate to market_data───│
  │            │            │             │─────────────►│             │
  │            │            │             │             │─►MCPToolset─►│
  │            │            │             │             │             │
  │            │            │             │             │◄─stock data──│
  │            │            │             │◄────────────│             │
  │            │            │◄────────────│             │             │
  │            │            │─►after_agent_callback     │             │
  │            │            │   (save to memory)        │             │
  │◄──response──│◄───────────│             │             │             │
```

---

## Project Structure

```
Real_Time_Finance_AI_POC/
├── finance_agent/                    # Primary application code
│   ├── main.py                       # FastAPI server (port 8080)
│   ├── agent.py                      # Agent factory module
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Container build
│   ├── mcp_servers/
│   │   └── yahoo_finance_server.py   # Bundled MCP server
│   └── workflow_1/
│       ├── config.py                 # Environment configuration
        ├── agents/
        │   ├── root_agent.py         # Root orchestrator + callbacks
        │   ├── market_data_agent.py  # Yahoo Finance sub-agent
        │   ├── economic_agent.py     # FRED API sub-agent
        │   ├── headlines_agent.py    # Google Search sub-agent
        │   └── portfolio_agent.py    # Portfolio analysis sub-agent
│       ├── api_clients/
│       │   ├── fred_api.py           # FRED economic data client
│       │   └── news_api.py           # NewsAPI client (legacy - using Google Search)
│       ├── mcp_clients/
│       │   └── yahoo_finance.py      # MCPToolset wrapper
│       ├── memory/
│       │   └── postgres_memory.py    # PostgreSQL memory service
│       ├── utils/
│       │   ├── sp500_validator.py    # S&P 500 ticker validation
│       │   ├── query_classifier.py   # Complexity classification
│       │   └── dynamic_model_callback.py  # Per-request model routing
│       └── arize_observability/
│           ├── observability.py      # Arize tracing setup
│           └── evaluations.py        # Evaluation framework (10 custom + 3 LLM-as-a-Judge)
├── terraform/                        # Infrastructure as code
│   ├── cloudsql.tf                   # Cloud SQL PostgreSQL
│   ├── variables.tf                  # Terraform variables
│   └── terraform.tfvars              # Variable values
├── tests/                            # Test files
│   ├── demo.py                       # Demo script
│   └── test_api_clients.py           # API client tests
├── docs/                             # Documentation
│   └── ARCHITECTURE.md               # This file
└── .github/
    └── copilot-instructions.md       # Development guidelines
```

---

## Environment Variables

```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global        # For Gemini 3 Pro Preview

# Model Configuration
MODEL_FAST=gemini-2.5-flash
MODEL_COMPLEX=gemini-3-pro-preview

# Memory Storage
MEMORY_STORAGE=auto                 # auto | postgres | memory
SESSION_DB_URL=postgresql://...     # For PostgresMemoryService

# API Keys (Optional)
FRED_API_KEY=...                    # For Economic Agent (FRED API)
# NEWS_API_KEY - DEPRECATED (headlines_agent uses Google Search instead)

# Arize Observability
ARIZE_API_KEY=...
ARIZE_SPACE_ID=...
ARIZE_PROJECT_NAME=finance-chatbot
ARIZE_ENABLED=true
```

---

## Key Design Decisions

1. **Dynamic Model Routing over Static Routing**: `before_model_callback` allows per-request optimization without code changes.

2. **MCP over Direct yfinance**: Encapsulates Yahoo Finance in a separate process, isolating dependencies.

3. **InMemorySessionService + PostgresMemoryService**: Sessions are ephemeral (Arize logs them), memories persist for recall.

4. **ADK's Multi-Agent Architecture**: Enables specialized agents without monolithic instruction prompts.

5. **OpenTelemetry Tracing**: First-class observability with Arize for production monitoring.

6. **Comprehensive Evaluation Framework**: 10 custom code evaluators + 3 LLM-as-a-Judge for quality assurance before deployment.

7. **Smart Evaluator Filtering**: Evaluators skip inapplicable tests based on query type (e.g., stock evaluators skip economic queries).

---

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/)
- [ADK Callbacks](https://google.github.io/adk-docs/agents/callbacks/)
- [MCP Tools Integration](https://google.github.io/adk-docs/tools/mcp-tools/)
- [ADK Memory Service](https://google.github.io/adk-docs/sessions/memory/)
- [Arize AX Documentation](https://docs.arize.com/arize)
- [Arize Phoenix (Open Source)](https://docs.arize.com/phoenix)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)

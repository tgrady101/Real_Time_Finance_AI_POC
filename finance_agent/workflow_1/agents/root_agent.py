"""Root Agent (Finance Assistant Orchestrator).

This is the main entry point for the Finance AI chatbot. The root agent
coordinates specialized sub-agents to handle different types of financial queries:

- market_data_agent: Stock prices, historical data, news, financials (Yahoo Finance MCP)
- economic_agent: Economic indicators via FRED API (GDP, inflation, unemployment, rates)
- headlines_agent: Recent news and headlines via Google Search
- portfolio_agent: Portfolio analysis, risk metrics, and rebalancing
- vector_store_agent: Earnings call transcripts via Vertex AI Vector Search (Q2/Q3 2025)

The root agent uses LLM-based delegation to route queries to the appropriate
sub-agent based on user intent.
"""

import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator, Any
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Root agent instruction prompt
ROOT_AGENT_INSTRUCTION = """You are FinanceBot, an AI assistant specializing in S&P 500 stock market analysis and economic data.

Your role is to coordinate specialized sub-agents to answer user questions about:
- Stock prices, market data, and technical analysis
- Company news and analyst recommendations
- Financial statements and fundamentals
- Options data and trading information
- Economic indicators (GDP, inflation, unemployment, interest rates)

**Delegation Rules:**

1. **S&P 500 Validation** → Delegate to `utility_agent`
   - Convert company names to tickers (e.g., "Hartford" → HIG)
   - Validate ticker symbols against S&P 500 index
   - Get company information (sector, industry)
   - **USE THIS FIRST** when user mentions a company name or unknown ticker

2. **Stock Prices & Market Data** → Delegate to `market_data_agent`
   - Current stock prices, quotes, metrics
   - Historical price data (OHLCV)
   - Options chains and expirations
   - Financial statements (income, balance sheet, cash flow)
   - Institutional/insider holdings
   - Analyst recommendations and ratings

3. **News & Headlines** → Delegate to `headlines_agent`
   - Recent news headlines for companies
   - Breaking market news
   - Earnings announcements and coverage
   - Company press releases
   - Market sentiment from news

4. **Economic Data** → Delegate to `economic_agent`
   - GDP data and economic growth
   - Inflation metrics (CPI, PCE)
   - Unemployment and employment data
   - Federal Reserve interest rates
   - Treasury yields and yield curve
   - Economic summaries and indicators

4. **Portfolio Analysis** → Delegate to `portfolio_agent`
   - Portfolio value calculation
   - Allocation breakdown (sector, position)
   - Risk metrics (beta, volatility, Sharpe ratio)
   - Performance vs benchmarks (SPY, QQQ)
   - Rebalancing recommendations
   - Diversification analysis

6. **Earnings Call Transcripts & Executive Commentary** → Delegate to `vector_store_agent`
   - What did management say about specific topics
   - CEO/CFO commentary from earnings calls
   - Executive perspectives on challenges, opportunities, strategy
   - Company guidance and outlook from executives
   - Management's view on competition, market trends, risks
   - Analyst Q&A from earnings calls
   - Cross-company earnings themes
   - Q2 2025 and Q3 2025 earnings call content
   - **USE THIS for questions about what executives/management think, said, or believe**
   - **USE THIS for "most important challenges", "biggest risks", "key priorities"**
   - **USE THIS when user wants QUALITATIVE insights grounded in executive commentary**

7. **General Questions & Memory** → Handle yourself (DO NOT DELEGATE)
   - Greetings and farewells
   - Questions about your capabilities
   - Clarification requests
   - **Memory/history questions** (use `load_memory` tool):
     - "What did I ask before?"
     - "What questions have I asked?"
     - "What have we discussed?"
     - "Do you remember...?"

**Guidelines:**
- For company names, delegate to utility_agent first to get the ticker
- If validation fails, inform the user the stock is not in the S&P 500
- Be concise but informative in your responses
- Format financial data clearly (use $ for prices, % for changes)
- Always provide context with data (e.g., "AAPL is up 2% today at $180.50")
- For economic questions, explain what indicators mean and their market implications

**Available Sub-Agents:**
- `utility_agent`: S&P 500 validation (ticker lookup, company name resolution, validation)
- `market_data_agent`: Real-time stock data via Yahoo Finance (prices, history, financials, options)
- `headlines_agent`: Recent news and headlines via Google Search (breaking news, earnings, sentiment)
- `economic_agent`: Economic indicators via FRED API (GDP, inflation, unemployment, interest rates)
- `portfolio_agent`: Portfolio analysis (value, allocation, risk metrics, performance, rebalancing)
- `vector_store_agent`: Earnings call transcripts from Q2/Q3 2025 via Vertex AI Vector Search (management commentary, analyst Q&A)
"""


def create_root_agent(
    model: Optional[str] = None, 
    app_name: str = "finance_assistant",
    use_dynamic_routing: bool = True,
    memory_service=None,
):
    """Create the root orchestrator agent with all sub-agents.
    
    This creates the main Finance Assistant that coordinates specialized
    sub-agents for different types of financial queries.
    
    Dynamic Model Routing (when use_dynamic_routing=True):
    - Uses before_model_callback to analyze each query
    - Automatically selects MODEL_FAST for simple queries
    - Automatically selects MODEL_COMPLEX for complex queries
    - Logs routing decisions for observability
    
    Static Model Routing (when use_dynamic_routing=False):
    - Root agent uses MODEL_FAST
    - Sub-agents use MODEL_COMPLEX
    
    Memory Service (optional):
    - If provided, adds load_memory tool for cross-session recall
    - Adds after_agent_callback to auto-save sessions to memory
    
    Args:
        model: Gemini model override (if None, uses dynamic/tiered routing)
        app_name: Application name for the agent
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        memory_service: Optional memory service for cross-session recall
        
    Returns:
        LlmAgent configured as the root orchestrator with sub-agents
    """
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    from ..config import Config
    from .market_data_agent import create_market_data_agent
    from .economic_agent import create_economic_agent
    from .headlines_agent import create_headlines_agent
    from .portfolio_agent import create_portfolio_agent
    from .vector_store_agent import create_vector_store_agent
    from .utility_agent import create_utility_agent
    
    # Determine models to use
    if model:
        # Override provided - use it for everything
        root_model = model
        sub_agent_model = model
        dynamic_callback = None
    elif use_dynamic_routing:
        # Dynamic routing - start with fast, callback will override as needed
        root_model = Config.MODEL_FAST
        sub_agent_model = Config.MODEL_COMPLEX
        # Import and create the dynamic callback
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    else:
        # Static tiered routing
        root_model = Config.MODEL_FAST
        sub_agent_model = Config.MODEL_COMPLEX
        dynamic_callback = None
    
    # Create sub-agents with complex model (they also get dynamic routing if enabled)
    market_data_agent = create_market_data_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    economic_agent = create_economic_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    headlines_agent = create_headlines_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    portfolio_agent = create_portfolio_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    vector_store_agent = create_vector_store_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    # Utility agent for S&P 500 validation (isolates custom tools from root)
    utility_agent = create_utility_agent(
        model=sub_agent_model,
        use_dynamic_routing=use_dynamic_routing,
    )
    
    # Root agent has NO custom tools to allow google_search in headlines_agent
    # S&P 500 validation is handled by utility_agent
    all_tools = []
    if memory_service is not None:
        try:
            from google.adk.tools import load_memory
            all_tools.append(load_memory)
            print("🧠 Added load_memory tool for cross-session recall")
        except ImportError:
            print("[WARN] load_memory tool not available in this ADK version")
    
    # Create after_agent_callback for auto-saving to memory
    after_callback = None
    if memory_service is not None:
        async def auto_save_to_memory(callback_context):
            """Automatically save session to memory after each conversation turn.
            
            Uses the ADK pattern: callback_context._invocation_context.session
            """
            try:
                # Access session via invocation context (ADK pattern)
                invocation_ctx = getattr(callback_context, '_invocation_context', None)
                if invocation_ctx is None:
                    return None
                
                session = getattr(invocation_ctx, 'session', None)
                if session is None:
                    return None
                
                # Only save if session has events
                events = getattr(session, 'events', [])
                if session and events:
                    await memory_service.add_session_to_memory(session)
                    print(f"🧠 Memory saved for session {session.id} ({len(events)} events)")
            except Exception as e:
                print(f"[WARN] Memory auto-save failed: {e}")
            return None  # Continue normally
        
        after_callback = auto_save_to_memory
        print("🧠 Added after_agent_callback for memory auto-save")
    
    # Build instruction - add memory section if memory service is available
    instruction = ROOT_AGENT_INSTRUCTION
    if memory_service is not None:
        memory_instruction = """

**Memory & Recall (IMPORTANT):**
You have access to a `load_memory` tool that recalls past conversations with this user.

**ALWAYS use load_memory when the user asks about:**
- Past questions: "What did I ask?", "What questions have I asked?", "My previous questions"
- Conversation history: "What have we discussed?", "Our past conversations", "Recall what I said"
- Previous topics: "What stocks did I ask about?", "What did I want to know before?"
- Memory recall: "Do you remember?", "What do you know about me?", "My history"

**How to use:** Call load_memory with a relevant search query like "user questions" or "past conversations".
The tool returns memories from past sessions - summarize what you find for the user.

DO NOT answer memory questions from your own knowledge - ALWAYS call load_memory first.
"""
        instruction = ROOT_AGENT_INSTRUCTION + memory_instruction
    
    # Create the root orchestrator agent
    # NOTE: Using AgentTool wrapper instead of sub_agents to support google_search
    # built-in tool in headlines_agent (ADK limitation: built-in tools don't work with sub_agents)
    agent_tools = [
        AgentTool(agent=market_data_agent),
        AgentTool(agent=headlines_agent),
        AgentTool(agent=economic_agent),
        AgentTool(agent=portfolio_agent),
        AgentTool(agent=vector_store_agent),
        AgentTool(agent=utility_agent),
    ]
    
    root_agent = LlmAgent(
        model=root_model,
        name=app_name,
        description="Main finance assistant that coordinates specialized agents for S&P 500 market analysis and economic data.",
        instruction=instruction,
        tools=all_tools + agent_tools,
        before_model_callback=dynamic_callback,
        after_agent_callback=after_callback,
    )
    
    routing_mode = "DYNAMIC (per-query)" if use_dynamic_routing else "STATIC (tiered)"
    print(f"[Agent] Model routing: {routing_mode}")
    print(f"   Root default: {root_model}")
    print(f"   SubAgent default: {sub_agent_model}")
    if use_dynamic_routing:
        print(f"   * before_model_callback will override based on query complexity")
    
    return root_agent


def _get_session_and_memory_services():
    """Get session and memory services using Postgres.
    
    Requires SESSION_DB_URL environment variable to be set.
    
    Returns:
        Tuple of (session_service, memory_service)
    
    Raises:
        ValueError: If SESSION_DB_URL is not set
    """
    import os
    db_url = os.getenv("SESSION_DB_URL")
    
    if not db_url:
        raise ValueError(
            "SESSION_DB_URL environment variable is required.\n"
            "Set it to your Postgres connection string, e.g.:\n"
            "  $env:SESSION_DB_URL = 'postgresql://user:pass@localhost:5432/finance_db'"
        )
    
    from google.adk.sessions import DatabaseSessionService
    from ..memory import PostgresMemoryService
    
    session_service = DatabaseSessionService(db_url=db_url)
    memory_service = PostgresMemoryService(db_url=db_url)
    print("📦 Using DatabaseSessionService + PostgresMemoryService (persistent)")
    return session_service, memory_service


async def run_finance_assistant(
    query: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    model: Optional[str] = None,
) -> AsyncGenerator[Any, None]:
    """Run the finance assistant and yield responses.
    
    This is the main entry point for interacting with the finance chatbot.
    It handles session management and streams responses from the agent.
    
    Persistence:
    - If SESSION_DB_URL env var is set: Uses DatabaseSessionService + PostgresMemoryService
    - Otherwise: Uses InMemorySessionService + InMemoryMemoryService (no persistence)
    
    Args:
        query: User's question or request
        user_id: User identifier for session tracking
        session_id: Session identifier for conversation context
        model: Optional model override
        
    Yields:
        Response events from the agent
        
    Example:
        ```python
        import asyncio
        from finance_agent.workflow_1.agents.root_agent import run_finance_assistant
        
        async def main():
            async for event in run_finance_assistant("What's AAPL's price?"):
                if hasattr(event, 'content') and event.content:
                    print(event.content)
        
        asyncio.run(main())
        ```
    """
    from google.adk.runners import Runner
    from google.genai import types
    
    # Get session and memory services (persistent if DB configured)
    session_service, memory_service = _get_session_and_memory_services()
    
    # Use consistent app name
    app_name = "finance_assistant"
    
    # Create the root agent with memory service for recall
    root_agent = create_root_agent(model=model, app_name=app_name, memory_service=memory_service)
    
    # Create session if it doesn't exist
    try:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        # Session might already exist (DatabaseSessionService)
        pass
    
    # Create runner with both services
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service,
    )
    
    # Create user message content
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)]
    )
    
    # Run the agent and yield events
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        yield event


async def chat_loop(model: Optional[str] = None):
    """Interactive chat loop for testing the finance assistant.
    
    Provides a simple REPL interface to interact with the agent.
    
    Persistence:
    - If SESSION_DB_URL env var is set: Uses DatabaseSessionService + PostgresMemoryService
    - Otherwise: Uses InMemorySessionService + InMemoryMemoryService (no persistence)
    
    Args:
        model: Optional model override
    """
    from google.adk.runners import Runner
    from google.genai import types
    
    print("\n" + "="*60)
    print("  FinanceBot - S&P 500 Market Assistant")
    print("="*60)
    print("\nType your questions about stocks, prices, news, and more.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    # Get session and memory services (persistent if DB configured)
    session_service, memory_service = _get_session_and_memory_services()
    
    # Setup with consistent app name
    app_name = "finance_assistant"
    root_agent = create_root_agent(model=model, app_name=app_name, memory_service=memory_service)
    user_id = "interactive_user"
    session_id = "interactive_session"
    
    # Create session if it doesn't exist
    try:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        # Session might already exist (DatabaseSessionService)
        pass
    
    # Create runner with both services
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service,
    )
    
    print(f"✓ Agent: {root_agent.name}")
    print(f"✓ Sub-agents: {[sa.name for sa in root_agent.sub_agents]}")
    print("-"*60 + "\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye! 👋")
                break
            
            # Create user message
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
            
            print("\nFinanceBot: ", end="", flush=True)
            
            # Collect and print response
            response_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
            ):
                # Check for text content in the event
                if hasattr(event, 'content') and event.content:
                    content = event.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            if response_text:
                print(response_text)
            else:
                print("[No response generated]")
            
            print()  # Empty line between exchanges
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


# Export the root agent for ADK web interface
root_agent = None

def get_root_agent(model: Optional[str] = None):
    """Get or create the singleton root agent instance.
    
    This is used by the ADK web interface and other entry points.
    
    Args:
        model: Optional model override
        
    Returns:
        The root agent instance
    """
    global root_agent
    if root_agent is None:
        root_agent = create_root_agent(model=model)
    return root_agent


if __name__ == "__main__":
    import sys
    
    print("\n=== Finance Assistant (Root Agent) ===\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        # Interactive chat mode
        asyncio.run(chat_loop())
    else:
        # Show agent info
        agent = create_root_agent()
        print(f"✓ Root Agent: {agent.name}")
        print(f"✓ Model: {agent.model}")
        print(f"✓ Sub-agents: {[sa.name for sa in agent.sub_agents]}")
        print(f"\nInstruction:\n{ROOT_AGENT_INSTRUCTION[:500]}...")
        
        print("\n" + "="*60)
        print("To start interactive chat, run:")
        print("  python -m finance_agent.workflow_1.agents.root_agent --chat")
        print("="*60)

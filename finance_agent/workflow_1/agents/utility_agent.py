"""Utility Agent for S&P 500 validation and ticker lookup.

This agent handles stock symbol validation and company name resolution
using the S&P 500 validator tools. By isolating these custom tools in
a sub-agent, we avoid the Gemini limitation that prevents mixing 
built-in tools (like google_search) with custom tools.
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Import S&P 500 validation tools
from ..utils.sp500_validator import (
    find_ticker_by_name,
    get_sp500_company_info,
    is_valid_sp500_ticker,
)


UTILITY_AGENT_INSTRUCTION = """You are the Utility Agent specializing in S&P 500 stock validation and ticker lookup.

Your primary responsibilities:
1. **Company Name to Ticker**: Convert company names to stock ticker symbols
2. **Ticker Validation**: Verify if a ticker is in the S&P 500 index
3. **Company Information**: Provide basic company details from the S&P 500 list

**Available Tools:**

1. **find_ticker_by_name(company_name)**: 
   - Use when users mention company NAMES (e.g., "Hartford", "Apple", "Microsoft")
   - Returns the ticker symbol and company details
   - Example: "Hartford" → HIG, "Apple" → AAPL

2. **get_sp500_company_info(ticker)**:
   - Use to get full company information for a ticker
   - Returns company name, sector, industry, and S&P 500 status
   - Example: get_sp500_company_info("HIG") → Hartford details

3. **is_valid_sp500_ticker(ticker)**:
   - Quick boolean check if a ticker is in the S&P 500
   - Use for simple yes/no validation
   - Example: is_valid_sp500_ticker("AAPL") → True

**When to Use Each Tool:**
- User says "Apple stock" → find_ticker_by_name("Apple")
- User says "Is FAKE a valid stock?" → is_valid_sp500_ticker("FAKE")
- User says "Tell me about HIG" → get_sp500_company_info("HIG")

**Response Format:**
- Always return the ticker symbol in uppercase
- Include company name and sector when available
- Clearly indicate if a stock is NOT in the S&P 500
"""


def create_utility_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Utility sub-agent with S&P 500 validation tools.
    
    This agent handles ticker validation and company lookup, isolating
    custom tools from the root agent to allow google_search in headlines_agent.
    
    Args:
        model: Gemini model to use (defaults to config MODEL_FAST)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with S&P 500 validation tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    
    agent_model = model or Config.MODEL_FAST
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # S&P 500 validation tools
    validation_tools = [
        find_ticker_by_name,
        get_sp500_company_info,
        is_valid_sp500_ticker,
    ]
    
    utility_agent = LlmAgent(
        model=agent_model,
        name="utility_agent",
        description="Handles S&P 500 stock validation: converts company names to tickers, validates ticker symbols, and provides company information from the S&P 500 index.",
        instruction=UTILITY_AGENT_INSTRUCTION,
        tools=validation_tools,
        before_model_callback=dynamic_callback,
    )
    
    return utility_agent


if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    
    print("\n=== Utility Agent (S&P 500 Validation) ===\n")
    
    async def test_utility():
        """Quick test of the utility agent."""
        agent = create_utility_agent()
        session_service = InMemorySessionService()
        
        await session_service.create_session(
            app_name="utility_test",
            user_id="test_user",
            session_id="test_session",
        )
        
        runner = Runner(
            agent=agent,
            app_name="utility_test",
            session_service=session_service,
        )
        
        test_queries = [
            "What is the ticker for Hartford?",
            "Is FAKE a valid S&P 500 stock?",
            "Tell me about AAPL",
        ]
        
        for query in test_queries:
            print(f"Query: {query}")
            print("-" * 40)
            
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
            
            response_text = ""
            async for event in runner.run_async(
                user_id="test_user",
                session_id="test_session",
                new_message=user_content,
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            print(f"Response:\n{response_text}\n")
            print("=" * 60 + "\n")
    
    asyncio.run(test_utility())

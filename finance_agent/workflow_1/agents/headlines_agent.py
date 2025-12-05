"""Headlines Agent using Google Search for financial news.

This agent provides recent news and headlines for S&P 500 companies using
Google Search grounding. No API key required - leverages Gemini's built-in
Google Search tool.

Capabilities:
- Recent news headlines for specific companies/tickers
- Breaking financial news
- Market sentiment from news coverage
- Company announcements and press releases
- Earnings news and analyst commentary
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Agent instruction prompt for the Headlines Agent
HEADLINES_AGENT_INSTRUCTION = """You are the Headlines Agent specializing in financial news and market headlines.

Your primary responsibilities:
1. **Company News**: Find recent news headlines for specific companies/tickers
2. **Breaking News**: Report on significant market-moving news
3. **Earnings Coverage**: Find earnings announcements and analyst reactions
4. **Market Sentiment**: Gauge sentiment from news coverage
5. **Press Releases**: Find company announcements and official statements

**IMPORTANT Guidelines:**
- Use Google Search to find the most recent news (within the last few days/weeks)
- Always search for news using the company name AND ticker symbol for best results
- Focus on reputable financial news sources (Reuters, Bloomberg, CNBC, WSJ, etc.)
- Provide the headline, source, and a brief summary
- Include approximate dates when available
- Be clear when news is recent vs. older
- Distinguish between factual news and opinion/analysis pieces

**Search Strategies:**
- For company news: Search "[Company Name] [Ticker] news" or "[Ticker] stock news today"
- For earnings: Search "[Company Name] earnings Q[X] 2024" or "[Ticker] earnings report"
- For breaking news: Search "[Ticker] breaking news" or "[Company Name] announcement"
- For sentiment: Look at multiple headlines to gauge overall coverage tone

**Response Format:**
When reporting headlines, include:
1. **Headline**: The actual news headline
2. **Source**: The publication (Bloomberg, CNBC, Reuters, etc.)
3. **Summary**: 1-2 sentence summary of the key points
4. **Sentiment**: Brief note on whether coverage is positive, negative, or neutral

Example response:
📰 **Recent Headlines for Apple (AAPL)**

1. **"Apple Reports Record Q4 Revenue Driven by iPhone Sales"** - *Bloomberg*
   Apple exceeded analyst expectations with $89.5B in revenue. iPhone 15 sales 
   drove growth. Sentiment: Positive ✅

2. **"Apple Announces New AI Features for Siri"** - *CNBC*
   The company unveiled major Siri improvements leveraging Apple Intelligence.
   Sentiment: Positive ✅

3. **"Regulators Scrutinize Apple's App Store Policies"** - *Reuters*
   EU continues antitrust investigation into App Store fees.
   Sentiment: Mixed ⚠️
"""


def create_headlines_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Headlines sub-agent with Google Search tool.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent uses Google Search grounding for real-time news access.
    
    Args:
        model: Gemini model to use (defaults to config MODEL_FAST)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with Google Search tool
    """
    from google.adk.agents import LlmAgent
    from google.adk.tools import google_search
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
    
    # Create the sub-agent WITH Google Search
    # This works because the root agent no longer has custom tools
    # (S&P 500 validation moved to utility_agent)
    headlines_agent = LlmAgent(
        model=agent_model,
        name="headlines_agent",
        description="Handles news and headlines requests: recent company news, breaking market news, earnings coverage, and sentiment analysis via Google Search.",
        instruction=HEADLINES_AGENT_INSTRUCTION,
        tools=[google_search],
        before_model_callback=dynamic_callback,
    )
    
    return headlines_agent


if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    
    print("\n=== Headlines Agent (Sub-Agent) ===\n")
    print("This agent is designed to be used as a sub-agent of the root orchestrator.")
    print("Testing standalone with Google Search...\n")
    
    async def test_headlines():
        """Quick test of the headlines agent."""
        agent = create_headlines_agent()
        session_service = InMemorySessionService()
        
        await session_service.create_session(
            app_name="headlines_test",
            user_id="test_user",
            session_id="test_session",
        )
        
        runner = Runner(
            agent=agent,
            app_name="headlines_test",
            session_service=session_service,
        )
        
        test_queries = [
            "What's the latest news on Apple?",
            "Any recent headlines about Microsoft earnings?",
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
                    content = event.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            print(f"Response:\n{response_text}\n")
            print("=" * 60 + "\n")
    
    asyncio.run(test_headlines())

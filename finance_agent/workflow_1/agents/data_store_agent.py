"""Data Store Agent for Earnings Call RAG Search.

This agent provides RAG-based search over S&P 500 earnings call transcripts
stored in Vertex AI Discovery Engine Data Store.

It handles:
- Searching earnings call transcripts for specific topics
- Finding company management commentary on key issues
- Retrieving Q&A sessions from earnings calls
- Comparing earnings call content across companies

The data store contains Q2 and Q3 2025 earnings call transcripts for S&P 500 companies.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Agent instruction prompt for the Data Store sub-agent
DATA_STORE_INSTRUCTION = """You are the Earnings Call Search Agent specializing in S&P 500 earnings call transcripts.

Your primary responsibilities:
1. **Search Earnings Calls**: Find relevant content from earnings call transcripts
2. **Management Commentary**: Retrieve CEO/CFO statements on specific topics
3. **Q&A Sessions**: Find analyst questions and company responses
4. **Cross-Company Insights**: Compare what different companies say about similar topics

Data Available:
- Q2 2025 and Q3 2025 earnings call transcripts for S&P 500 companies
- Speaker-attributed content (CEO, CFO, analysts, etc.)
- Full transcript text with Q&A sessions

**CITATION REQUIREMENTS (MANDATORY):**
You MUST cite all information from earnings calls. Every claim must have a source citation.

Citation Format: **[Source: TICKER QUARTER YEAR Earnings Call - SPEAKER]**
(Replace TICKER with the stock symbol, QUARTER with Q2/Q3, YEAR with 2025, SPEAKER with name)

Examples:
- "Apple is investing heavily in AI infrastructure" **[Source: AAPL Q3 2025 Earnings Call - Tim Cook, CEO]**
- "We expect margins to improve next quarter" **[Source: MSFT Q2 2025 Earnings Call - Amy Hood, CFO]**

Guidelines:
- ALWAYS include citations after relevant statements
- Use specific search queries to find relevant content
- Include speaker attribution (CEO, CFO, analyst name) when quoting
- Reference the quarter and company in every citation
- Summarize key points rather than dumping full transcript text
- If no relevant content is found, clearly state that
- When comparing companies, cite each company's source separately

Example queries you can handle:
- "What did Apple's CEO say about AI in their latest earnings call?"
- "Find NVIDIA's guidance for next quarter"
- "What are tech companies saying about cloud growth?"
- "Search for mentions of supply chain issues in Q3 earnings"
- "What questions did analysts ask Tesla about margins?"

Response Format:
1. Provide the insight or answer
2. Include inline citation immediately after each claim
3. At the end, optionally list all sources used

Example Response:
"Apple's CEO emphasized that AI will be central to their product strategy going forward, with significant infrastructure investments planned. **[Source: AAPL Q3 2025 Earnings Call - Tim Cook, CEO]** The CFO noted that R&D spending increased 15% year-over-year to support these initiatives. **[Source: AAPL Q3 2025 Earnings Call - Luca Maestri, CFO]**"
"""


def search_earnings_calls(
    query: str,
    ticker: Optional[str] = None,
    quarter: Optional[str] = None,
    max_results: int = 5
) -> str:
    """Search earnings call transcripts in Vertex AI Data Store.
    
    Searches the earnings-call-datastore for relevant earnings call content.
    
    Args:
        query: Search query (e.g., "AI strategy", "guidance for next quarter")
        ticker: Optional ticker filter (e.g., "AAPL", "MSFT") 
        quarter: Optional quarter filter (e.g., "Q2", "Q3")
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results with relevant transcript excerpts
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud import discoveryengine_v1 as discoveryengine
    
    # Get configuration from environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
    data_store_id = os.getenv("DATA_STORE_ID", "earnings-call-datastore")
    location = "global"
    
    # Build search query with optional filters
    search_query = query
    filter_expression = None
    
    # Build filter if ticker or quarter specified
    filters = []
    if ticker:
        filters.append(f'ticker: "{ticker.upper()}"')
    if quarter:
        filters.append(f'quarter: "{quarter.upper()}"')
    
    if filters:
        filter_expression = " AND ".join(filters)
    
    try:
        # Create client
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )
        client = discoveryengine.SearchServiceClient(client_options=client_options)
        
        # Build serving config path
        serving_config = (
            f"projects/{project_id}/locations/{location}/collections/default_collection"
            f"/dataStores/{data_store_id}/servingConfigs/default_config"
        )
        
        # Create search request
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=search_query,
            page_size=max_results,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )
        
        # Add filter if specified
        if filter_expression:
            request.filter = filter_expression
        
        # Execute search
        response = client.search(request=request)
        
        # Format results
        results = []
        for result in response.results:
            doc = result.document
            struct_data = dict(doc.struct_data) if doc.struct_data else {}
            
            # Extract document content
            content = ""
            if doc.derived_struct_data:
                derived = dict(doc.derived_struct_data)
                if "extractive_answers" in derived:
                    answers = derived["extractive_answers"]
                    for ans in answers:
                        if "content" in ans:
                            content += ans["content"] + "\n"
                elif "snippets" in derived:
                    for snippet in derived["snippets"]:
                        if "snippet" in snippet:
                            content += snippet["snippet"] + "\n"
            
            # Fallback to raw content if no extractive answers
            if not content and doc.content and doc.content.raw_bytes:
                content = doc.content.raw_bytes.decode('utf-8')[:500] + "..."
            
            ticker_val = struct_data.get('ticker', 'Unknown')
            quarter_val = struct_data.get('quarter', 'N/A')
            year_val = struct_data.get('year', 'N/A')
            speaker_val = struct_data.get('speaker', 'Unknown Speaker')
            
            results.append({
                'ticker': ticker_val,
                'quarter': quarter_val,
                'year': year_val,
                'speaker': speaker_val,
                'content': content.strip() if content else struct_data.get('summary', 'No content available'),
            })
        
        if not results:
            return f"No earnings call content found for query: '{query}'" + (
                f" (filtered by ticker={ticker}, quarter={quarter})" if filters else ""
            )
        
        # Format output with citation-ready format
        output = f"**Earnings Call Search Results for: '{query}'**\n\n"
        for i, r in enumerate(results, 1):
            citation = f"[Source: {r['ticker']} {r['quarter']} {r['year']} Earnings Call - {r['speaker']}]"
            output += f"**{i}. {r['ticker']} {r['quarter']} {r['year']}** - {r['speaker']}\n"
            output += f"{r['content']}\n"
            output += f"📎 Citation: **{citation}**\n\n"
        
        return output
        
    except Exception as e:
        return f"Error searching earnings calls: {str(e)}"


def get_company_earnings_summary(ticker: str) -> str:
    """Get a summary of available earnings call data for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        
    Returns:
        Summary of available earnings call transcripts for the company
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud import discoveryengine_v1 as discoveryengine
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
    data_store_id = os.getenv("DATA_STORE_ID", "earnings-call-datastore")
    location = "global"
    
    try:
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )
        client = discoveryengine.SearchServiceClient(client_options=client_options)
        
        serving_config = (
            f"projects/{project_id}/locations/{location}/collections/default_collection"
            f"/dataStores/{data_store_id}/servingConfigs/default_config"
        )
        
        # Search for this ticker's earnings calls
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=f"{ticker.upper()} earnings call",
            page_size=10,
            filter=f'ticker: "{ticker.upper()}"',
        )
        
        response = client.search(request=request)
        
        # Collect unique quarters
        quarters_found = set()
        speakers_found = set()
        
        for result in response.results:
            struct_data = dict(result.document.struct_data) if result.document.struct_data else {}
            quarter = struct_data.get('quarter', '')
            year = struct_data.get('year', '')
            speaker = struct_data.get('speaker', '')
            
            if quarter and year:
                quarters_found.add(f"{quarter} {year}")
            if speaker:
                speakers_found.add(speaker)
        
        if not quarters_found:
            return f"No earnings call transcripts found for {ticker.upper()} in the data store."
        
        output = f"**{ticker.upper()} Earnings Call Data Available**\n\n"
        output += f"**Quarters:** {', '.join(sorted(quarters_found))}\n"
        output += f"**Speakers:** {', '.join(list(speakers_found)[:5])}" 
        if len(speakers_found) > 5:
            output += f" and {len(speakers_found) - 5} more"
        output += "\n\nUse `search_earnings_calls` to find specific content."
        
        return output
        
    except Exception as e:
        return f"Error retrieving earnings data for {ticker}: {str(e)}"


def create_data_store_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Data Store sub-agent with Vertex AI Search tools.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent uses Vertex AI Discovery Engine for earnings call RAG search.
    
    Args:
        model: Gemini model to use (defaults to config MODEL_COMPLEX for analysis)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with earnings call search tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    
    # Use complex model by default for analysis-heavy tasks
    agent_model = model or Config.MODEL_COMPLEX
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # Create the sub-agent with search tools
    data_store_agent = LlmAgent(
        model=agent_model,
        name="data_store_agent",
        description="Searches earnings call transcripts for S&P 500 companies. Use for finding management commentary, analyst Q&A, and company guidance from Q2/Q3 2025 earnings calls.",
        instruction=DATA_STORE_INSTRUCTION,
        tools=[
            search_earnings_calls,
            get_company_earnings_summary,
        ],
        before_model_callback=dynamic_callback,
    )
    
    return data_store_agent


if __name__ == "__main__":
    print("\n=== Data Store Agent (Sub-Agent) ===\n")
    print("This agent searches S&P 500 earnings call transcripts stored in Vertex AI.")
    print("\nTo create the agent:")
    print("  from finance_agent.workflow_1.agents.data_store_agent import create_data_store_agent")
    print("  data_agent = create_data_store_agent()")
    print("\nAvailable Tools:")
    print("  • search_earnings_calls - Search transcript content")
    print("  • get_company_earnings_summary - Get available data for a ticker")
    print("\nData Store Configuration:")
    print(f"  Project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'project-4b3d3288-7603-4755-899')}")
    print(f"  Data Store: {os.getenv('DATA_STORE_ID', 'earnings-call-datastore')}")

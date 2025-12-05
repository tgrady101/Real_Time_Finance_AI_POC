"""Economic Agent using FRED API for economic data.

This agent provides economic indicators and macroeconomic data through the FRED API.
It handles:
- GDP and economic growth data
- Inflation indicators (CPI, PCE)
- Unemployment rates and employment data
- Federal Reserve interest rates
- Treasury yields
- Consumer sentiment

The agent provides context about how economic data relates to stock market performance.
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Agent instruction prompt for the Economic Data sub-agent
ECONOMIC_AGENT_INSTRUCTION = """You are the Economic Data Agent specializing in macroeconomic indicators and Federal Reserve data.

Your primary responsibilities:
1. **GDP Data**: Retrieve GDP growth rates and economic output data
2. **Inflation Metrics**: Get CPI, PCE, and other inflation indicators
3. **Employment Data**: Unemployment rates, payroll numbers, labor market data
4. **Interest Rates**: Federal funds rate, Treasury yields, rate decisions
5. **Economic Summaries**: Provide comprehensive economic overviews
6. **Historical Trends**: Analyze economic indicator trends for specific years or date ranges

Guidelines:
- Always explain what economic indicators mean in plain language
- Provide context about how indicators relate to stock markets
- Include the date of the most recent data point
- Compare current values to historical norms when relevant
- Explain trends (rising, falling, stable) and their implications

**IMPORTANT: For historical or trend queries:**
- Use `get_indicator_trend` or `get_cpi_trend` for queries about specific years
- Examples: "CPI trend in 2024", "unemployment in 2023", "how did GDP change last year"
- Pass the year parameter (e.g., year=2024) for year-specific queries
- Use start_date/end_date for custom date ranges

Available FRED Series:
- GDP: Gross Domestic Product (quarterly, billions $)
- GDPC1: Real GDP (inflation-adjusted)
- CPIAUCSL: Consumer Price Index (monthly, inflation indicator)
- PCE: Personal Consumption Expenditures
- UNRATE: Unemployment Rate (monthly, %)
- PAYEMS: Total Nonfarm Payrolls (monthly, thousands)
- DFF: Federal Funds Effective Rate (daily, %)
- FEDFUNDS: Federal Funds Target Rate
- DGS10: 10-Year Treasury Yield (daily, %)
- DGS2: 2-Year Treasury Yield (daily, %)
- T10Y2Y: 10-Year minus 2-Year Treasury (yield curve)
- UMCSENT: Consumer Sentiment Index

Example responses:
- "The current unemployment rate is 3.8%, which is historically low. This suggests a tight labor market."
- "The Fed funds rate is currently 5.25-5.50%. The 10-year Treasury yield is 4.2%, indicating..."
- "GDP grew 2.1% in Q3 2024, showing moderate economic expansion."
- "CPI is up 3.2% year-over-year, above the Fed's 2% target but down from 2022 peaks."
- "In 2024, CPI rose from 308.42 to 315.61, a 2.3% increase year-over-year."
"""


# FRED tool functions that wrap the API client
def get_gdp_data(periods: int = 4) -> str:
    """Get recent GDP (Gross Domestic Product) data.
    
    Args:
        periods: Number of quarters to retrieve (default: 4 for 1 year)
        
    Returns:
        Formatted string with GDP data and context
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        observations = client.get_series_observations('GDP', limit=periods)
        
        if not observations:
            return "No GDP data available."
        
        result = "📊 **GDP Data (Gross Domestic Product)**\n\n"
        for obs in observations:
            date = obs.get('date', 'N/A')
            value = obs.get('value', 'N/A')
            if value != '.':
                result += f"- {date}: ${float(value):,.1f} billion\n"
        
        # Add context
        if len(observations) >= 2:
            latest = float(observations[0]['value']) if observations[0]['value'] != '.' else None
            previous = float(observations[1]['value']) if observations[1]['value'] != '.' else None
            if latest and previous:
                change = ((latest - previous) / previous) * 100
                result += f"\nQuarterly change: {change:+.2f}%"
        
        return result
    except Exception as e:
        return f"Error retrieving GDP data: {str(e)}"


def get_inflation_data(periods: int = 12) -> str:
    """Get recent CPI (Consumer Price Index) inflation data.
    
    Args:
        periods: Number of months to retrieve (default: 12 for 1 year)
        
    Returns:
        Formatted string with inflation data and context
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        observations = client.get_series_observations('CPIAUCSL', limit=periods)
        
        if not observations:
            return "No inflation data available."
        
        result = "📈 **CPI Inflation Data**\n\n"
        
        # Show last few months
        for obs in observations[:6]:
            date = obs.get('date', 'N/A')
            value = obs.get('value', 'N/A')
            if value != '.':
                result += f"- {date}: {float(value):.1f}\n"
        
        # Calculate year-over-year inflation if we have enough data
        if len(observations) >= 12:
            current = float(observations[0]['value']) if observations[0]['value'] != '.' else None
            year_ago = float(observations[11]['value']) if observations[11]['value'] != '.' else None
            if current and year_ago:
                yoy_inflation = ((current - year_ago) / year_ago) * 100
                result += f"\n**Year-over-Year Inflation: {yoy_inflation:.1f}%**"
                result += f"\n(Fed target: 2.0%)"
        
        return result
    except Exception as e:
        return f"Error retrieving inflation data: {str(e)}"


def get_unemployment_data(periods: int = 12) -> str:
    """Get recent unemployment rate data.
    
    Args:
        periods: Number of months to retrieve (default: 12 for 1 year)
        
    Returns:
        Formatted string with unemployment data and context
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        observations = client.get_series_observations('UNRATE', limit=periods)
        
        if not observations:
            return "No unemployment data available."
        
        result = "👥 **Unemployment Rate**\n\n"
        
        # Show recent months
        for obs in observations[:6]:
            date = obs.get('date', 'N/A')
            value = obs.get('value', 'N/A')
            if value != '.':
                result += f"- {date}: {float(value):.1f}%\n"
        
        # Add context
        latest = float(observations[0]['value']) if observations[0]['value'] != '.' else None
        if latest:
            if latest < 4.0:
                result += "\n*This is historically low, indicating a tight labor market.*"
            elif latest < 5.0:
                result += "\n*This is considered healthy employment.*"
            elif latest < 7.0:
                result += "\n*This is elevated, suggesting economic stress.*"
            else:
                result += "\n*This is high, indicating significant job losses.*"
        
        return result
    except Exception as e:
        return f"Error retrieving unemployment data: {str(e)}"


def get_interest_rates() -> str:
    """Get current Federal Reserve and Treasury interest rates.
    
    Returns:
        Formatted string with interest rate data and context
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        
        # Get various rates
        fed_funds = client.get_latest_value('DFF')
        treasury_10y = client.get_latest_value('DGS10')
        treasury_2y = client.get_latest_value('DGS2')
        spread = client.get_latest_value('T10Y2Y')
        
        result = "🏦 **Interest Rates**\n\n"
        
        if fed_funds and fed_funds.get('value') != '.':
            result += f"**Federal Funds Rate**: {float(fed_funds['value']):.2f}%\n"
            result += f"  (as of {fed_funds['date']})\n\n"
        
        if treasury_2y and treasury_2y.get('value') != '.':
            result += f"**2-Year Treasury Yield**: {float(treasury_2y['value']):.2f}%\n"
        
        if treasury_10y and treasury_10y.get('value') != '.':
            result += f"**10-Year Treasury Yield**: {float(treasury_10y['value']):.2f}%\n"
        
        if spread and spread.get('value') != '.':
            spread_val = float(spread['value'])
            result += f"\n**Yield Curve (10Y-2Y)**: {spread_val:.2f}%\n"
            if spread_val < 0:
                result += "*⚠️ Yield curve is inverted - historically a recession indicator*"
            elif spread_val < 0.5:
                result += "*Yield curve is flat - suggests economic uncertainty*"
            else:
                result += "*Yield curve is normal - suggests economic stability*"
        
        return result
    except Exception as e:
        return f"Error retrieving interest rate data: {str(e)}"


def get_economic_summary() -> str:
    """Get a comprehensive summary of key economic indicators.
    
    Returns:
        Formatted string with all major economic indicators
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    from datetime import datetime
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        summary = client.get_economic_summary()
        
        result = "📊 **Economic Summary**\n"
        result += f"*As of {datetime.now().strftime('%B %d, %Y')}*\n\n"
        
        # GDP
        if summary.get('gdp'):
            result += f"**GDP**: ${summary['gdp']:,.1f} billion\n"
        
        # Inflation
        if summary.get('inflation_cpi'):
            result += f"**CPI Index**: {summary['inflation_cpi']:.1f}\n"
        
        # Unemployment
        if summary.get('unemployment_rate'):
            result += f"**Unemployment Rate**: {summary['unemployment_rate']:.1f}%\n"
        
        # Interest rates
        rates = summary.get('interest_rates', {})
        if rates.get('fed_funds_rate'):
            result += f"**Fed Funds Rate**: {rates['fed_funds_rate']:.2f}%\n"
        if rates.get('treasury_10y'):
            result += f"**10-Year Treasury**: {rates['treasury_10y']:.2f}%\n"
        
        result += "\n---\n"
        result += "*Use specific queries for detailed data and historical trends.*"
        
        return result
    except Exception as e:
        return f"Error retrieving economic summary: {str(e)}"


def get_fred_series(series_id: str, periods: int = 10) -> str:
    """Get data for any FRED series by ID.
    
    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'UNRATE', 'DGS10')
        periods: Number of observations to retrieve (default: 10)
        
    Returns:
        Formatted string with series data
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        observations = client.get_series_observations(series_id.upper(), limit=periods)
        
        if not observations:
            return f"No data available for series '{series_id}'."
        
        result = f"📊 **FRED Series: {series_id.upper()}**\n\n"
        
        for obs in observations:
            date = obs.get('date', 'N/A')
            value = obs.get('value', 'N/A')
            if value != '.':
                result += f"- {date}: {value}\n"
        
        return result
    except Exception as e:
        return f"Error retrieving series {series_id}: {str(e)}"


def get_indicator_trend(
    indicator: str, 
    year: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """Get historical trend data for an economic indicator.
    
    Use this for queries about historical data, trends over time, or specific years.
    Examples: "CPI trend in 2024", "unemployment rate last year", "GDP from 2023 to 2024"
    
    Args:
        indicator: Name of indicator ('cpi', 'gdp', 'unemployment', 'fed_funds_rate', 
                   'treasury_10y') or FRED series ID
        year: Specific year to get data for (e.g., 2024)
        start_date: Start date in YYYY-MM-DD format (use with end_date)
        end_date: End date in YYYY-MM-DD format (use with start_date)
        
    Returns:
        Formatted string with trend data and analysis
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        trend_data = client.get_indicator_trend(
            indicator=indicator,
            year=year,
            start_date=start_date,
            end_date=end_date
        )
        
        if trend_data.get('error'):
            return trend_data['error']
        
        # Format the response
        indicator_name = trend_data.get('indicator', indicator).upper()
        
        result = f"📊 **{indicator_name} Trend Analysis**\n"
        if year:
            result += f"*Year: {year}*\n\n"
        else:
            result += f"*Period: {trend_data['period_start']} to {trend_data['period_end']}*\n\n"
        
        # Summary statistics
        result += "**Summary:**\n"
        result += f"- Starting Value: {trend_data['start_value']}\n"
        result += f"- Ending Value: {trend_data['end_value']}\n"
        result += f"- Change: {trend_data['absolute_change']:+.2f} ({trend_data['percent_change']:+.2f}%)\n"
        result += f"- Min: {trend_data['min_value']} | Max: {trend_data['max_value']} | Avg: {trend_data['avg_value']}\n"
        result += f"- Data Points: {trend_data['data_points']}\n\n"
        
        # Show sample observations (first 6 and last 3 if many)
        observations = trend_data.get('observations', [])
        if observations:
            result += "**Data Points:**\n"
            if len(observations) <= 12:
                for obs in observations:
                    date = obs.get('date', 'N/A')
                    value = obs.get('value', 'N/A')
                    if value != '.':
                        result += f"- {date}: {value}\n"
            else:
                # Show first 6
                for obs in observations[:6]:
                    date = obs.get('date', 'N/A')
                    value = obs.get('value', 'N/A')
                    if value != '.':
                        result += f"- {date}: {value}\n"
                result += f"  ... ({len(observations) - 9} more data points) ...\n"
                # Show last 3
                for obs in observations[-3:]:
                    date = obs.get('date', 'N/A')
                    value = obs.get('value', 'N/A')
                    if value != '.':
                        result += f"- {date}: {value}\n"
        
        return result
    except Exception as e:
        return f"Error retrieving trend data for {indicator}: {str(e)}"


def get_cpi_trend(year: Optional[int] = None) -> str:
    """Get CPI (inflation) trend data for a specific year.
    
    Use this when asked about CPI or inflation trends, especially for specific years.
    Examples: "CPI trend in 2024", "inflation last year", "how did CPI change in 2023"
    
    Args:
        year: Year to retrieve CPI data for (e.g., 2024, 2023)
        
    Returns:
        Formatted string with CPI trend analysis
    """
    from ..api_clients.fred_api import FREDAPIClient
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        return "Error: FRED_API_KEY not configured. Cannot retrieve economic data."
    
    try:
        client = FREDAPIClient(api_key)
        trend_data = client.get_cpi_trend(year=year)
        
        if trend_data.get('error'):
            return trend_data['error']
        
        result = "📈 **CPI (Consumer Price Index) Trend**\n"
        if year:
            result += f"*Year: {year}*\n\n"
        else:
            result += f"*Period: {trend_data['period_start']} to {trend_data['period_end']}*\n\n"
        
        result += "**Summary:**\n"
        result += f"- Starting CPI: {trend_data['start_value']:.2f}\n"
        result += f"- Ending CPI: {trend_data['end_value']:.2f}\n"
        result += f"- Change: {trend_data['absolute_change']:+.2f} ({trend_data['percent_change']:+.2f}%)\n"
        result += f"- Range: {trend_data['min_value']:.2f} to {trend_data['max_value']:.2f}\n"
        result += f"- Monthly Data Points: {trend_data['data_points']}\n\n"
        
        # Interpret the trend
        pct_change = trend_data['percent_change']
        if year:
            if pct_change > 5:
                result += f"*⚠️ High inflation: CPI rose {pct_change:.1f}% in {year}, well above Fed's 2% target.*\n"
            elif pct_change > 3:
                result += f"*Elevated inflation: CPI rose {pct_change:.1f}% in {year}, above Fed's 2% target.*\n"
            elif pct_change > 2:
                result += f"*Moderate inflation: CPI rose {pct_change:.1f}% in {year}, near Fed's 2% target.*\n"
            else:
                result += f"*Low inflation: CPI rose only {pct_change:.1f}% in {year}.*\n"
        
        # Show monthly data
        observations = trend_data.get('observations', [])
        if observations:
            result += "\n**Monthly Values:**\n"
            for obs in observations:
                date = obs.get('date', 'N/A')
                value = obs.get('value', 'N/A')
                if value != '.':
                    result += f"- {date}: {float(value):.2f}\n"
        
        return result
    except Exception as e:
        return f"Error retrieving CPI trend: {str(e)}"


def create_economic_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Economic Data sub-agent with FRED API tools.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent uses FRED API for macroeconomic data.
    
    Args:
        model: Gemini model to use (defaults to config GEMINI_MODEL)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with FRED API tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    
    agent_model = model or Config.GEMINI_MODEL
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # FRED API tools
    economic_tools = [
        get_gdp_data,
        get_inflation_data,
        get_unemployment_data,
        get_interest_rates,
        get_economic_summary,
        get_fred_series,
        get_indicator_trend,  # For historical/trend queries with date ranges
        get_cpi_trend,        # Specialized CPI trend analysis
    ]
    
    # Create the sub-agent with FRED tools
    economic_agent = LlmAgent(
        model=agent_model,
        name="economic_agent",
        description="Handles macroeconomic data requests: GDP, inflation, unemployment, interest rates, and economic indicators via FRED API.",
        instruction=ECONOMIC_AGENT_INSTRUCTION,
        tools=economic_tools,
        before_model_callback=dynamic_callback,
    )
    
    return economic_agent


if __name__ == "__main__":
    import os
    
    print("\n=== Economic Agent (Sub-Agent) ===\n")
    print("This agent is designed to be used as a sub-agent of the root orchestrator.")
    print("Use the root_agent.py to run the full finance assistant.")
    
    # Test the tools if API key is available
    if os.getenv('FRED_API_KEY'):
        print("\n--- Testing FRED API Tools ---\n")
        
        print("1. Economic Summary:")
        print(get_economic_summary())
        
        print("\n2. Interest Rates:")
        print(get_interest_rates())
        
        print("\n3. Unemployment:")
        print(get_unemployment_data(periods=6))
    else:
        print("\n⚠️ FRED_API_KEY not set - cannot test tools")
        print("Set FRED_API_KEY in your .env file to enable economic data")
    
    print("\nAvailable Tools:")
    tools = [
        "get_gdp_data",
        "get_inflation_data",
        "get_unemployment_data",
        "get_interest_rates",
        "get_economic_summary",
        "get_fred_series",
        "get_indicator_trend",  # NEW: Historical trends with date ranges
        "get_cpi_trend",        # NEW: CPI-specific trend analysis
    ]
    for tool in tools:
        print(f"  • {tool}")

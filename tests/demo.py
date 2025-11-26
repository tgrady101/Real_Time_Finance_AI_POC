"""Quick demo script showing all capabilities.

Run this to see the chatbot components in action.
"""

import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for stdout (fix for Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Find project root and add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env from project root
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

def demo_market_data():
    """Demo market data capabilities."""
    print("\n" + "="*60)
    print(" 📈 MARKET DATA DEMO")
    print("="*60)
    
    from src.workflow_1.api_clients.ninja_api import get_ninja_client
    from src.workflow_1.utils.sp500_validator import validate_ticker_with_message
    
    client = get_ninja_client()
    
    # Validate and get stock data
    tickers = ["AAPL", "MSFT", "GOOGL", "INVALID"]
    
    for ticker in tickers:
        is_valid, message = validate_ticker_with_message(ticker)
        print(f"\n{message}")
        
        if is_valid:
            try:
                data = client.get_stock_price(ticker)
                print(f"💰 Current Price: ${data['price']}")
                print(f"📊 Volume: {data.get('volume', 'N/A'):,}")
            except Exception as e:
                print(f"❌ Error fetching price: {e}")

def demo_economic_data():
    """Demo economic data capabilities."""
    print("\n" + "="*60)
    print(" 📊 ECONOMIC DATA DEMO")
    print("="*60)
    
    from src.workflow_1.api_clients.fred_api import get_fred_client
    
    client = get_fred_client()
    
    print("\n🇺🇸 U.S. Economic Indicators:")
    
    # Get economic summary
    summary = client.get_economic_summary()
    
    if summary['gdp']:
        print(f"\n💵 GDP: ${summary['gdp']:,.2f} billion")
    
    if summary['inflation_cpi']:
        print(f"📈 CPI (Inflation): {summary['inflation_cpi']:.2f}")
    
    if summary['unemployment_rate']:
        print(f"👥 Unemployment Rate: {summary['unemployment_rate']}%")
    
    rates = summary['interest_rates']
    if rates['fed_funds_rate']:
        print(f"🏦 Fed Funds Rate: {rates['fed_funds_rate']}%")
    if rates['treasury_10y']:
        print(f"📜 10-Year Treasury: {rates['treasury_10y']}%")

def demo_company_analysis():
    """Demo full company analysis."""
    print("\n" + "="*60)
    print(" 🏢 COMPANY ANALYSIS DEMO")
    print("="*60)
    
    from src.workflow_1.api_clients.ninja_api import get_ninja_client
    from src.workflow_1.api_clients.news_api import get_news_client
    
    ninja = get_ninja_client()
    news = get_news_client()
    
    ticker = "TSLA"
    
    print(f"\n🔍 Analyzing {ticker}...")
    
    # Get company info
    info = ninja.get_ticker_info(ticker)
    print(f"\n📋 Company: {info.get('name')}")
    print(f"👨‍💼 CEO: {info.get('chief_executive_officer')}")
    print(f"🏭 Industry: {info.get('industry')}")
    print(f"💰 Market Cap: ${info.get('latest_market_cap', 0):,.0f}")
    
    # Get current price
    price = ninja.get_stock_price(ticker)
    print(f"\n💵 Current Price: ${price['price']}")
    
    # Get recent news
    print(f"\n📰 Recent News:")
    articles = news.get_company_news(info.get('name'), ticker, limit=3)
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article.get('title', 'No title')}")
        print(f"   Source: {article.get('source', {}).get('name', 'Unknown')}")

def demo_sp500_validation():
    """Demo S&P 500 validation with fuzzy matching."""
    print("\n" + "="*60)
    print(" ✅ S&P 500 VALIDATION DEMO")
    print("="*60)
    
    from src.workflow_1.utils.sp500_validator import get_validator
    
    validator = get_validator()
    
    print(f"\n📊 Loaded {len(validator.get_all_tickers())} S&P 500 companies")
    
    # Test cases
    test_tickers = [
        ("AAPL", "Valid ticker"),
        ("GOOGL", "Valid ticker (Alphabet)"),
        ("APPL", "Typo - should suggest AAPL"),
        ("TSLL", "Typo - should suggest TSLA"),
        ("BITCOIN", "Not in S&P 500"),
    ]
    
    for ticker, description in test_tickers:
        print(f"\n🔍 Testing: {ticker} ({description})")
        message = validator.get_validation_message(ticker)
        print(f"   {message}")

def main():
    """Run all demos."""
    print("\n" + "="*60)
    print(" 🎯 Real-Time Finance AI - Capability Demo")
    print("="*60)
    
    demos = [
        ("S&P 500 Validation", demo_sp500_validation),
        ("Market Data", demo_market_data),
        ("Economic Data", demo_economic_data),
        ("Company Analysis", demo_company_analysis),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(" ✨ Demo Complete!")
    print("="*60)
    print("\nNext: Build the agents to make this conversational! 🤖\n")

if __name__ == "__main__":
    main()

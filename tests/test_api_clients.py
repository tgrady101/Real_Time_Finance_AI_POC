"""Integration tests for API clients.

Tests real API calls to verify everything works.
Run with: python tests/test_api_clients.py
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

def test_ninja_api():
    """Test Ninja API client."""
    print("\n" + "="*60)
    print(" Testing Ninja API Client")
    print("="*60)
    
    try:
        from src.workflow_1.api_clients.ninja_api import NinjaAPIClient
        
        client = NinjaAPIClient()
        print("✓ Client initialized")
        
        # Test stock price
        print("\n1. Testing stock price (AAPL)...")
        price_data = client.get_stock_price('AAPL')
        print(f"   ✓ Price: ${price_data.get('price')}")
        print(f"   ✓ Exchange: {price_data.get('exchange')}")
        
        # Test ticker info
        print("\n2. Testing company info (MSFT)...")
        info = client.get_ticker_info('MSFT')
        print(f"   ✓ Company: {info.get('name')}")
        print(f"   ✓ CEO: {info.get('chief_executive_officer')}")
        
        # Test crypto price
        print("\n3. Testing crypto price (BTCUSD)...")
        crypto = client.get_crypto_price('BTCUSD')
        price = float(crypto.get('price', 0))
        print(f"   ✓ BTC Price: ${price:,.2f}")
        
        print("\n✅ Ninja API tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ninja API test failed: {e}")
        return False

def test_fred_api():
    """Test FRED API client."""
    print("\n" + "="*60)
    print(" Testing FRED API Client")
    print("="*60)
    
    try:
        from src.workflow_1.api_clients.fred_api import FREDAPIClient
        
        client = FREDAPIClient()
        print("✓ Client initialized")
        
        # Test GDP
        print("\n1. Testing GDP data...")
        gdp = client.get_latest_gdp()
        if gdp:
            print(f"   ✓ Latest GDP: ${gdp:,.2f} billion")
        
        # Test unemployment
        print("\n2. Testing unemployment rate...")
        unemp = client.get_latest_unemployment()
        if unemp:
            print(f"   ✓ Unemployment Rate: {unemp}%")
        
        # Test interest rates
        print("\n3. Testing interest rates...")
        rates = client.get_interest_rates()
        if rates['fed_funds_rate']:
            print(f"   ✓ Fed Funds Rate: {rates['fed_funds_rate']}%")
        if rates['treasury_10y']:
            print(f"   ✓ 10-Year Treasury: {rates['treasury_10y']}%")
        
        print("\n✅ FRED API tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ FRED API test failed: {e}")
        return False

def test_sp500_validator():
    """Test S&P 500 validator."""
    print("\n" + "="*60)
    print(" Testing S&P 500 Validator")
    print("="*60)
    
    try:
        from src.workflow_1.utils.sp500_validator import SP500Validator
        
        validator = SP500Validator()
        print("✓ Validator initialized")
        
        # Load S&P 500 list
        print("\n1. Loading S&P 500 list...")
        validator.refresh_list()
        tickers = validator.get_all_tickers()
        print(f"   ✓ Loaded {len(tickers)} S&P 500 companies")
        
        # Test valid ticker
        print("\n2. Testing valid ticker (AAPL)...")
        is_valid = validator.is_valid_ticker('AAPL')
        company = validator.get_company_name('AAPL')
        print(f"   ✓ Valid: {is_valid}")
        print(f"   ✓ Company: {company}")
        
        # Test invalid ticker with suggestions
        print("\n3. Testing invalid ticker (APPL - typo)...")
        suggestions = validator.suggest_similar_tickers('APPL', max_suggestions=3)
        if suggestions:
            print(f"   ✓ Found {len(suggestions)} suggestions:")
            for ticker, name, score in suggestions:
                print(f"      - {ticker} ({name}) - {score:.2%} match")
        
        print("\n✅ S&P 500 Validator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ S&P 500 Validator test failed: {e}")
        return False

def test_news_api():
    """Test NewsAPI client."""
    print("\n" + "="*60)
    print(" Testing NewsAPI Client")
    print("="*60)
    
    try:
        from src.workflow_1.api_clients.news_api import NewsAPIClient
        
        client = NewsAPIClient()
        print("✓ Client initialized")
        
        if client.has_api_key:
            print("\n1. Testing company news (Apple)...")
            articles = client.get_company_news("Apple Inc", "AAPL", limit=3)
            print(f"   ✓ Retrieved {len(articles)} articles")
            
            if articles:
                print(f"\n   Latest article:")
                print(f"   - {articles[0].get('title', 'No title')}")
                print(f"   - Source: {articles[0].get('source', {}).get('name', 'Unknown')}")
        else:
            print("\n⚠️  NewsAPI key not configured (optional)")
            print("   Testing fallback mode...")
            articles = client.get_company_news("Tesla", "TSLA")
            print(f"   ✓ Fallback works: {len(articles)} placeholder returned")
        
        print("\n✅ NewsAPI tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ NewsAPI test failed: {e}")
        return False

def main():
    """Run all API client tests."""
    print("\n🧪 Real-Time Finance AI - API Client Tests")
    print("="*60)
    
    results = []
    
    # Test each client
    results.append(("Ninja API", test_ninja_api()))
    results.append(("FRED API", test_fred_api()))
    results.append(("S&P 500 Validator", test_sp500_validator()))
    results.append(("NewsAPI", test_news_api()))
    
    # Summary
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n{total_passed}/{len(results)} test suites passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Your setup is working correctly.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

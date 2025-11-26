"""Ninja API client for financial market data.

Provides unified interface to Ninja API endpoints for:
- Stock prices (real-time)
- Crypto prices
- Forex exchange rates
- Company ticker information
- S&P 500 constituent list
"""

import os
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


class NinjaAPIClient:
    """Client for Ninja API financial data endpoints."""
    
    BASE_URL = "https://api.api-ninjas.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Ninja API client.
        
        Args:
            api_key: API key for Ninja API (defaults to NINJA_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('NINJA_API_KEY')
        if not self.api_key:
            raise ValueError("Ninja API key not provided and NINJA_API_KEY not set in environment")
        
        self.session = requests.Session()
        self.session.headers.update({'X-Api-Key': self.api_key})
        
        # Rate limiting (conservative for free tier)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to Ninja API with error handling.
        
        Args:
            endpoint: API endpoint path (e.g., '/stockprice')
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.exceptions.HTTPError: On HTTP error
            ValueError: On invalid response
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                raise ValueError("Payment required: This endpoint requires a paid Ninja API plan")
            elif e.response.status_code == 404:
                raise ValueError(f"Data not found for the requested parameters")
            else:
                raise ValueError(f"Ninja API error: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error calling Ninja API: {str(e)}")
    
    # Stock price endpoints
    
    def get_stock_price(self, ticker: str) -> Dict[str, Any]:
        """Get real-time stock price information.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict with keys: ticker, name, price, exchange, updated, currency, volume
        """
        return self._make_request('/stockprice', {'ticker': ticker.upper()})
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company profile information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with company details including CEO, address, market cap, dividend, etc.
        """
        return self._make_request('/ticker', {'ticker': ticker.upper()})
    
    # Crypto endpoints
    
    def get_crypto_price(self, symbol: str) -> Dict[str, Any]:
        """Get current cryptocurrency price.
        
        Args:
            symbol: Crypto trading pair (e.g., 'BTCUSD', 'ETHBTC')
            
        Returns:
            Dict with keys: symbol, price, timestamp
        """
        return self._make_request('/cryptoprice', {'symbol': symbol.upper()})
    
    # Forex endpoints
    
    def get_exchange_rate(self, pair: str) -> Dict[str, Any]:
        """Get forex exchange rate.
        
        Args:
            pair: Currency pair in format 'CUR1_CUR2' (e.g., 'USD_EUR')
            
        Returns:
            Dict with keys: currency_pair, exchange_rate, timestamp
        """
        return self._make_request('/exchangerate', {'pair': pair.upper()})
    
    # S&P 500 endpoint
    
    def get_sp500_list(self) -> List[Dict[str, Any]]:
        """Get list of all S&P 500 companies.
        
        Returns:
            List of dicts with company information
        """
        return self._make_request('/sp500')
    
    # Batch operations
    
    def get_multiple_stock_prices(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices for multiple stocks (sequential requests).
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker to price data
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_stock_price(ticker)
            except ValueError as e:
                results[ticker] = {'error': str(e)}
        return results


# Global client instance
_client = None

def get_client() -> NinjaAPIClient:
    """Get the global Ninja API client instance."""
    global _client
    if _client is None:
        _client = NinjaAPIClient()
    return _client


if __name__ == "__main__":
    # Test the client
    client = NinjaAPIClient()
    
    print("\n=== Ninja API Client Test ===\n")
    
    # Test stock price
    print("1. Stock Price (AAPL):")
    price_data = client.get_stock_price('AAPL')
    print(f"   Price: ${price_data.get('price')}")
    print(f"   Exchange: {price_data.get('exchange')}")
    
    # Test ticker info
    print("\n2. Company Info (MSFT):")
    info = client.get_ticker_info('MSFT')
    print(f"   Company: {info.get('name')}")
    print(f"   CEO: {info.get('chief_executive_officer')}")
    print(f"   Market Cap: ${info.get('latest_market_cap'):,}")
    
    # Test crypto
    print("\n3. Crypto Price (BTCUSD):")
    crypto = client.get_crypto_price('BTCUSD')
    print(f"   Price: ${crypto.get('price')}")
    
    # Test forex
    print("\n4. Exchange Rate (USD_EUR):")
    forex = client.get_exchange_rate('USD_EUR')
    print(f"   Rate: {forex.get('exchange_rate')}")
    
    print("\nAll tests completed successfully!")

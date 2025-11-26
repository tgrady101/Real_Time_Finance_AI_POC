"""S&P 500 ticker validation utility.

Validates ticker symbols against the current S&P 500 list.
Uses API Ninjas S&P 500 endpoint as primary source with Wikipedia as fallback.
"""

import os
import requests
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import difflib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class SP500Validator:
    """Validates ticker symbols against S&P 500 constituent list."""
    
    def __init__(self):
        self.api_key = os.getenv('NINJA_API_KEY')
        self.sp500_list: Dict[str, str] = {}  # ticker -> company_name
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(days=7)  # Refresh weekly
        
    def _fetch_from_api_ninjas(self) -> Dict[str, str]:
        """Fetch S&P 500 list from API Ninjas."""
        if not self.api_key:
            print("Warning: NINJA_API_KEY not set, skipping API Ninjas fetch")
            return {}
        
        url = "https://api.api-ninjas.com/v1/sp500"
        headers = {'X-Api-Key': self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to dictionary: ticker -> company name
            sp500_dict = {}
            for company in data:
                ticker = company.get('ticker', '').upper()
                name = company.get('name', '')
                if ticker and name:
                    sp500_dict[ticker] = name
            
            return sp500_dict
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch from API Ninjas: {e}")
            return {}

    def _fetch_from_wikipedia(self) -> Dict[str, str]:
        """Fetch S&P 500 list from Wikipedia as fallback."""
        try:
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # Add User-Agent to avoid 403 Forbidden
            storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            tables = pd.read_html(url, storage_options=storage_options)
            df = tables[0]
            
            sp500_dict = {}
            for _, row in df.iterrows():
                ticker = str(row['Symbol']).upper()
                name = str(row['Security'])
                sp500_dict[ticker] = name
                
            return sp500_dict
        except ImportError:
            print("Warning: pandas/lxml not installed, trying BeautifulSoup for Wikipedia fallback")
            return self._fetch_from_wikipedia_bs4()
        except Exception as e:
            print(f"Warning: Failed to fetch from Wikipedia (pandas): {e}")
            return self._fetch_from_wikipedia_bs4()

    def _fetch_from_wikipedia_bs4(self) -> Dict[str, str]:
        """Fetch S&P 500 list from Wikipedia using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            
            sp500_dict = {}
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    ticker = cols[0].text.strip().upper()
                    name = cols[1].text.strip()
                    sp500_dict[ticker] = name
            
            return sp500_dict
        except Exception as e:
            print(f"Warning: Failed to fetch from Wikipedia (BS4): {e}")
            return {}
    
    def _needs_refresh(self) -> bool:
        """Check if the S&P 500 list needs to be refreshed."""
        if not self.sp500_list:
            return True
        if not self.last_refresh:
            return True
        return datetime.now() - self.last_refresh > self.refresh_interval
    
    def refresh_list(self, force: bool = False):
        """Refresh the S&P 500 constituent list."""
        if not force and not self._needs_refresh():
            return
        
        print("Refreshing S&P 500 list...")
        
        # Try Wikipedia first (primary source)
        self.sp500_list = self._fetch_from_wikipedia()
        
        # Fallback to API Ninjas if Wikipedia fails
        if not self.sp500_list:
            print("Wikipedia fetch failed or returned empty, trying API Ninjas fallback...")
            self.sp500_list = self._fetch_from_api_ninjas()
        
        if self.sp500_list:
            self.last_refresh = datetime.now()
            print(f"Loaded {len(self.sp500_list)} S&P 500 companies")
        else:
            print("Warning: Failed to load S&P 500 list from all sources")
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker is in the S&P 500.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            True if ticker is in S&P 500, False otherwise
        """
        if not ticker:
            return False
        
        # Ensure list is loaded
        if self._needs_refresh():
            self.refresh_list()
        
        return ticker.upper() in self.sp500_list
    
    def get_company_name(self, ticker: str) -> Optional[str]:
        """Get company name for a valid S&P 500 ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name if ticker is valid, None otherwise
        """
        if not self.is_valid_ticker(ticker):
            return None
        
        return self.sp500_list.get(ticker.upper())
    
    def suggest_similar_tickers(self, ticker: str, max_suggestions: int = 3) -> List[Tuple[str, str, float]]:
        """Find similar tickers using fuzzy matching (for typos).
        
        Args:
            ticker: Invalid or potentially misspelled ticker
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of tuples: (ticker, company_name, similarity_score)
        """
        if self._needs_refresh():
            self.refresh_list()
        
        if not self.sp500_list:
            return []
        
        ticker_upper = ticker.upper()
        
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            ticker_upper, 
            self.sp500_list.keys(), 
            n=max_suggestions, 
            cutoff=0.6
        )
        
        suggestions = []
        for match in matches:
            company_name = self.sp500_list[match]
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, ticker_upper, match).ratio()
            suggestions.append((match, company_name, ratio))
        
        return suggestions
    
    def get_validation_message(self, ticker: str) -> str:
        """Get a user-friendly validation message for a ticker.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            Helpful message for the user
        """
        if self.is_valid_ticker(ticker):
            company_name = self.get_company_name(ticker)
            return f"✓ {ticker} is a valid S&P 500 ticker ({company_name})"
        
        # Ticker not valid, provide suggestions
        suggestions = self.suggest_similar_tickers(ticker)
        
        if suggestions:
            msg = f"✗ {ticker} is not in the S&P 500. Did you mean:\n"
            for tick, name, score in suggestions:
                msg += f"  - {tick} ({name})\n"
            return msg.strip()
        else:
            return (f"✗ {ticker} is not in the S&P 500. "
                   f"This chatbot only supports S&P 500 companies (500 tickers).")
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all S&P 500 tickers.
        
        Returns:
            List of ticker symbols
        """
        if self._needs_refresh():
            self.refresh_list()
        
        return sorted(list(self.sp500_list.keys()))


# Global validator instance
_validator = None

def get_validator() -> SP500Validator:
    """Get the global S&P 500 validator instance."""
    global _validator
    if _validator is None:
        _validator = SP500Validator()
        _validator.refresh_list()
    return _validator


# Convenience functions
def is_valid_sp500_ticker(ticker: str) -> bool:
    """Quick validation function."""
    return get_validator().is_valid_ticker(ticker)


def validate_ticker_with_message(ticker: str) -> Tuple[bool, str]:
    """Validate and get user message.
    
    Returns:
        Tuple of (is_valid, message)
    """
    validator = get_validator()
    is_valid = validator.is_valid_ticker(ticker)
    message = validator.get_validation_message(ticker)
    return is_valid, message


if __name__ == "__main__":
    # Test the validator
    validator = SP500Validator()
    validator.refresh_list()
    
    print("\n=== S&P 500 Validator Test ===\n")
    
    test_tickers = ["AAPL", "GOOGL", "INVALID", "APPL", "TSL"]
    
    for ticker in test_tickers:
        print(f"\nTesting: {ticker}")
        print(validator.get_validation_message(ticker))
    
    print(f"\n\nTotal S&P 500 companies loaded: {len(validator.get_all_tickers())}")

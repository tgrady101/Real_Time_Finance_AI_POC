"""S&P 500 ticker validation utility.

Validates ticker symbols against the current S&P 500 list.
Uses Wikipedia as the authoritative source for S&P 500 constituents.
"""

import requests
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import difflib


class SP500Validator:
    """Validates ticker symbols against S&P 500 constituent list from Wikipedia."""
    
    def __init__(self):
        self.sp500_list: Dict[str, str] = {}  # ticker -> company_name
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(days=7)  # Refresh weekly

    def _find_sp500_table(self, tables):
        """Find the S&P 500 constituents table using multiple detection strategies.
        
        Returns:
            DataFrame or None: The S&P 500 constituents table if found
        """
        import pandas as pd
        
        candidates = []
        
        for i, table in enumerate(tables):
            score = 0
            cols_lower = [str(c).lower() for c in table.columns]
            
            # Strategy 1: Look for 'symbol' column (strong signal)
            if any('symbol' in c for c in cols_lower):
                score += 50
            
            # Strategy 2: Look for 'security' column (strong signal)
            if any('security' in c for c in cols_lower):
                score += 40
            
            # Strategy 3: Look for 'gics' (sector classification - unique to S&P 500 table)
            if any('gics' in c for c in cols_lower):
                score += 30
            
            # Strategy 4: Look for 'cik' column (SEC identifier)
            if any('cik' in c for c in cols_lower):
                score += 20
            
            # Strategy 5: Row count should be ~500-510 (S&P 500 has ~503 companies)
            if 490 <= len(table) <= 520:
                score += 40
            elif 400 <= len(table) <= 600:
                score += 20
            
            # Strategy 6: Should have at least 5 columns
            if len(table.columns) >= 5:
                score += 10
            
            # Strategy 7: First column values should look like tickers (1-5 uppercase letters)
            if len(table) > 0:
                first_col = table.iloc[:, 0].astype(str)
                ticker_like = first_col.str.match(r'^[A-Z]{1,5}$').sum()
                if ticker_like > len(table) * 0.8:  # 80%+ look like tickers
                    score += 30
            
            if score > 0:
                candidates.append((score, i, table))
        
        if candidates:
            # Return table with highest score
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_idx, best_table = candidates[0]
            if best_score >= 50:  # Minimum confidence threshold
                return best_table
        
        return None

    def _fetch_from_wikipedia(self) -> Dict[str, str]:
        """Fetch S&P 500 list from Wikipedia using pandas."""
        try:
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # Add User-Agent to avoid 403 Forbidden
            storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            tables = pd.read_html(url, storage_options=storage_options)
            
            # Use smart table detection
            df = self._find_sp500_table(tables)
            
            if df is None:
                print("Warning: Could not find S&P 500 table - trying BeautifulSoup")
                return self._fetch_from_wikipedia_bs4()
            
            sp500_dict = {}
            # Handle different possible column names
            ticker_col = None
            name_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    ticker_col = col
                elif 'security' in col_lower or 'company' in col_lower or 'name' in col_lower:
                    name_col = col
            
            # Fallback to positional if column names not found
            if ticker_col is None:
                ticker_col = df.columns[0]
            if name_col is None:
                name_col = df.columns[1]
            
            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).upper().strip()
                name = str(row[name_col]).strip()
                if ticker and name and ticker != 'NAN' and len(ticker) <= 5:
                    sp500_dict[ticker] = name
            
            # Validate we got reasonable data
            if len(sp500_dict) < 400:
                print(f"Warning: Only found {len(sp500_dict)} tickers, expected ~500")
                return self._fetch_from_wikipedia_bs4()
                
            return sp500_dict
        except ImportError:
            print("pandas not installed, using BeautifulSoup fallback")
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
            
            if table is None:
                print("Warning: Could not find S&P 500 constituents table on Wikipedia")
                return {}
            
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
        """Refresh the S&P 500 constituent list from Wikipedia."""
        if not force and not self._needs_refresh():
            return
        
        print("Refreshing S&P 500 list from Wikipedia...")
        self.sp500_list = self._fetch_from_wikipedia()
        
        if self.sp500_list:
            self.last_refresh = datetime.now()
            print(f"Loaded {len(self.sp500_list)} S&P 500 companies from Wikipedia")
        else:
            print("Warning: Failed to load S&P 500 list from Wikipedia")
    
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
    
    def find_ticker_by_company_name(self, company_name: str, max_suggestions: int = 3) -> List[Tuple[str, str, float]]:
        """Find tickers by company name using fuzzy matching.
        
        Args:
            company_name: Company name or partial name (e.g., "Hartford", "Apple")
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of tuples: (ticker, company_name, similarity_score)
        """
        if self._needs_refresh():
            self.refresh_list()
        
        if not self.sp500_list:
            return []
        
        search_lower = company_name.lower().strip()
        suggestions = []
        
        # First, check for exact substring matches in company names
        for ticker, name in self.sp500_list.items():
            name_lower = name.lower()
            if search_lower in name_lower:
                # Higher score for matches at word boundaries
                words = name_lower.split()
                if any(search_lower in word or word in search_lower for word in words):
                    ratio = 0.95
                else:
                    ratio = 0.85
                suggestions.append((ticker, name, ratio))
        
        # If no substring matches, use fuzzy matching on company names
        if not suggestions:
            company_names = list(self.sp500_list.values())
            matches = difflib.get_close_matches(
                company_name,
                company_names,
                n=max_suggestions,
                cutoff=0.4
            )
            
            for match in matches:
                for ticker, name in self.sp500_list.items():
                    if name == match:
                        ratio = difflib.SequenceMatcher(None, company_name.lower(), name.lower()).ratio()
                        suggestions.append((ticker, name, ratio))
                        break
        
        # Sort by score and return top matches
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:max_suggestions]

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
        
        ticker_upper = ticker.upper().strip()
        
        matches = difflib.get_close_matches(
            ticker_upper, 
            self.sp500_list.keys(), 
            n=max_suggestions, 
            cutoff=0.6
        )
        
        suggestions = []
        for match in matches:
            company_name = self.sp500_list[match]
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


# Convenience functions for use as agent tools
def is_valid_sp500_ticker(ticker: str) -> bool:
    """Check if a ticker is in the S&P 500.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'HIG')
        
    Returns:
        True if ticker is in S&P 500, False otherwise
    """
    return get_validator().is_valid_ticker(ticker)


def validate_ticker_with_message(ticker: str) -> Tuple[bool, str]:
    """Validate a ticker and get a user-friendly message.
    
    Args:
        ticker: Stock ticker symbol to validate
    
    Returns:
        Tuple of (is_valid, message)
    """
    validator = get_validator()
    is_valid = validator.is_valid_ticker(ticker)
    message = validator.get_validation_message(ticker)
    return is_valid, message


def find_ticker_by_name(company_name: str) -> str:
    """Find S&P 500 ticker by company name.
    
    Use this to convert company names like "Hartford", "Apple", "Microsoft" 
    to their ticker symbols (HIG, AAPL, MSFT).
    
    Args:
        company_name: Company name or partial name (e.g., "Hartford", "Apple")
        
    Returns:
        Result string with ticker if found, or suggestions if not found
    """
    validator = get_validator()
    matches = validator.find_ticker_by_company_name(company_name, max_suggestions=3)
    
    if matches:
        if len(matches) == 1 or matches[0][2] > 0.9:
            # High confidence match
            ticker, name, _ = matches[0]
            return f"Found: {ticker} ({name})"
        else:
            # Multiple possible matches
            result = f"Possible matches for '{company_name}':\n"
            for ticker, name, score in matches:
                result += f"  - {ticker}: {name}\n"
            return result.strip()
    else:
        return f"No S&P 500 company found matching '{company_name}'"


def get_sp500_company_info(ticker: str) -> str:
    """Get S&P 500 company info for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Company info if valid S&P 500 ticker, error message otherwise
    """
    validator = get_validator()
    if validator.is_valid_ticker(ticker):
        name = validator.get_company_name(ticker)
        return f"{ticker.upper()} is in the S&P 500: {name}"
    else:
        return validator.get_validation_message(ticker)


if __name__ == "__main__":
    # Test the validator
    validator = SP500Validator()
    validator.refresh_list()
    
    print("\n=== S&P 500 Validator Test (Wikipedia Source) ===\n")
    
    # Test ticker validation
    test_tickers = ["AAPL", "GOOGL", "HIG", "INVALID", "APPL", "TSL"]
    print("--- Ticker Validation ---")
    for ticker in test_tickers:
        print(f"\n{ticker}: {validator.get_validation_message(ticker)}")
    
    # Test company name lookup
    print("\n--- Company Name Lookup ---")
    test_names = ["Hartford", "Apple", "Microsoft", "Google", "Tesla", "Invalid Corp"]
    for name in test_names:
        result = find_ticker_by_name(name)
        print(f"\n'{name}': {result}")
    
    print(f"\n\nTotal S&P 500 companies loaded: {len(validator.get_all_tickers())}")

"""FRED API client for economic data.

Provides access to Federal Reserve Economic Data (FRED) for:
- GDP data
- Inflation indicators (CPI, PCE)
- Unemployment rates
- Interest rates
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from ..arize_observability import instrument

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


class FREDAPIClient:
    """Client for Federal Reserve Economic Data (FRED) API."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common economic series IDs
    SERIES = {
        'gdp': 'GDP',  # Gross Domestic Product
        'gdp_real': 'GDPC1',  # Real GDP
        'cpi': 'CPIAUCSL',  # Consumer Price Index
        'pce': 'PCE',  # Personal Consumption Expenditures
        'unemployment': 'UNRATE',  # Unemployment Rate
        'fed_funds_rate': 'DFF',  # Federal Funds Rate
        'treasury_10y': 'DGS10',  # 10-Year Treasury Rate
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED API client.
        
        Args:
            api_key: FRED API key (defaults to FRED_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not provided and FRED_API_KEY not set in environment")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to FRED API.
        
        Args:
            endpoint: API endpoint (e.g., 'series/observations')
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"FRED API error: {str(e)}")
    
    def get_series_observations(
        self, 
        series_id: str, 
        limit: int = 10,
        sort_order: str = 'desc'
    ) -> List[Dict[str, Any]]:
        """Get observations for a FRED series.
        
        Args:
            series_id: FRED series ID
            limit: Number of most recent observations to return
            sort_order: 'asc' or 'desc'
            
        Returns:
            List of observations with date and value
        """
        params = {
            'series_id': series_id,
            'limit': limit,
            'sort_order': sort_order
        }
        
        data = self._make_request('series/observations', params)
        return data.get('observations', [])
    
    def get_latest_value(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent value for a series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dict with date and value, or None if no data
        """
        observations = self.get_series_observations(series_id, limit=1)
        if observations:
            return observations[0]
        return None
    
    # Convenience methods for common indicators
    
    def get_gdp(self, periods: int = 4) -> List[Dict[str, Any]]:
        """Get recent GDP data.
        
        Args:
            periods: Number of quarters to retrieve
            
        Returns:
            List of GDP observations
        """
        return self.get_series_observations(self.SERIES['gdp'], limit=periods)
    
    @instrument(name="fred_get_latest_gdp")
    def get_latest_gdp(self) -> Optional[float]:
        """Get the most recent GDP value.
        
        Returns:
            GDP value in billions or None
        """
        latest = self.get_latest_value(self.SERIES['gdp'])
        if latest and latest.get('value') != '.':
            return float(latest['value'])
        return None
    
    def get_inflation_rate(self, periods: int = 12) -> List[Dict[str, Any]]:
        """Get recent CPI data (inflation indicator).
        
        Args:
            periods: Number of months to retrieve
            
        Returns:
            List of CPI observations
        """
        return self.get_series_observations(self.SERIES['cpi'], limit=periods)
    
    @instrument(name="fred_get_latest_inflation")
    def get_latest_inflation(self) -> Optional[float]:
        """Get the most recent CPI value.
        
        Returns:
            CPI value or None
        """
        latest = self.get_latest_value(self.SERIES['cpi'])
        if latest and latest.get('value') != '.':
            return float(latest['value'])
        return None
    
    def get_unemployment_rate(self, periods: int = 12) -> List[Dict[str, Any]]:
        """Get recent unemployment rate data.
        
        Args:
            periods: Number of months to retrieve
            
        Returns:
            List of unemployment observations
        """
        return self.get_series_observations(self.SERIES['unemployment'], limit=periods)
    
    @instrument(name="fred_get_latest_unemployment")
    def get_latest_unemployment(self) -> Optional[float]:
        """Get the most recent unemployment rate.
        
        Returns:
            Unemployment rate percentage or None
        """
        latest = self.get_latest_value(self.SERIES['unemployment'])
        if latest and latest.get('value') != '.':
            return float(latest['value'])
        return None
    
    def get_interest_rates(self) -> Dict[str, Optional[float]]:
        """Get current interest rates (Fed funds and 10-year Treasury).
        
        Returns:
            Dict with fed_funds_rate and treasury_10y
        """
        fed_funds = self.get_latest_value(self.SERIES['fed_funds_rate'])
        treasury = self.get_latest_value(self.SERIES['treasury_10y'])
        
        return {
            'fed_funds_rate': float(fed_funds['value']) if fed_funds and fed_funds.get('value') != '.' else None,
            'treasury_10y': float(treasury['value']) if treasury and treasury.get('value') != '.' else None,
            'fed_funds_date': fed_funds.get('date') if fed_funds else None,
            'treasury_date': treasury.get('date') if treasury else None,
        }
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get a summary of key economic indicators.
        
        Returns:
            Dict with latest values for GDP, inflation, unemployment, rates
        """
        return {
            'gdp': self.get_latest_gdp(),
            'inflation_cpi': self.get_latest_inflation(),
            'unemployment_rate': self.get_latest_unemployment(),
            'interest_rates': self.get_interest_rates(),
            'as_of_date': datetime.now().strftime('%Y-%m-%d')
        }


# Global client instance
_client = None

def get_client() -> FREDAPIClient:
    """Get the global FRED API client instance."""
    global _client
    if _client is None:
        _client = FREDAPIClient()
    return _client


if __name__ == "__main__":
    # Test the client
    client = FREDAPIClient()
    
    print("\n=== FRED API Client Test ===\n")
    
    print("1. Latest GDP:")
    gdp = client.get_latest_gdp()
    print(f"   ${gdp:,.2f} billion")
    
    print("\n2. Latest Unemployment Rate:")
    unemp = client.get_latest_unemployment()
    print(f"   {unemp}%")
    
    print("\n3. Latest CPI (Inflation):")
    cpi = client.get_latest_inflation()
    print(f"   {cpi}")
    
    print("\n4. Interest Rates:")
    rates = client.get_interest_rates()
    print(f"   Fed Funds Rate: {rates['fed_funds_rate']}%")
    print(f"   10-Year Treasury: {rates['treasury_10y']}%")
    
    print("\n5. Economic Summary:")
    summary = client.get_economic_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\nAll tests completed successfully!")

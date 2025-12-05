"""FRED API client for economic data.

Provides access to Federal Reserve Economic Data (FRED) for:
- GDP data
- Inflation indicators (CPI, PCE)
- Unemployment rates
- Interest rates
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
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
        sort_order: str = 'desc',
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get observations for a FRED series.
        
        Args:
            series_id: FRED series ID
            limit: Number of most recent observations to return
            sort_order: 'asc' or 'desc'
            observation_start: Start date in YYYY-MM-DD format (optional)
            observation_end: End date in YYYY-MM-DD format (optional)
            
        Returns:
            List of observations with date and value
        """
        params = {
            'series_id': series_id,
            'limit': limit,
            'sort_order': sort_order
        }
        
        # Add date range filters if specified
        if observation_start:
            params['observation_start'] = observation_start
        if observation_end:
            params['observation_end'] = observation_end
        
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
    
    def get_series_for_year(
        self, 
        series_id: str, 
        year: int,
        sort_order: str = 'asc'
    ) -> List[Dict[str, Any]]:
        """Get all observations for a series within a specific year.
        
        Args:
            series_id: FRED series ID
            year: Year to retrieve data for (e.g., 2024)
            sort_order: 'asc' for chronological, 'desc' for reverse
            
        Returns:
            List of observations for that year
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Use a high limit to get all observations for the year
        return self.get_series_observations(
            series_id=series_id,
            limit=366,  # Max observations for daily data
            sort_order=sort_order,
            observation_start=start_date,
            observation_end=end_date
        )
    
    def get_series_for_date_range(
        self, 
        series_id: str, 
        start_date: str,
        end_date: str,
        sort_order: str = 'asc'
    ) -> List[Dict[str, Any]]:
        """Get all observations for a series within a date range.
        
        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sort_order: 'asc' for chronological, 'desc' for reverse
            
        Returns:
            List of observations for the date range
        """
        return self.get_series_observations(
            series_id=series_id,
            limit=1000,  # High limit for long date ranges
            sort_order=sort_order,
            observation_start=start_date,
            observation_end=end_date
        )
    
    @instrument(name="fred_get_cpi_trend")
    def get_cpi_trend(
        self, 
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get CPI trend data for a specific year or date range.
        
        Args:
            year: Year to get CPI data for (e.g., 2024)
            start_date: Start date in YYYY-MM-DD format (alternative to year)
            end_date: End date in YYYY-MM-DD format (alternative to year)
            
        Returns:
            Dict with CPI observations and trend analysis
        """
        if year:
            observations = self.get_series_for_year(self.SERIES['cpi'], year)
        elif start_date and end_date:
            observations = self.get_series_for_date_range(
                self.SERIES['cpi'], start_date, end_date
            )
        else:
            # Default to last 12 months
            observations = self.get_inflation_rate(periods=12)
        
        if not observations:
            return {'error': 'No CPI data available for the specified period'}
        
        # Calculate statistics
        values = [float(obs['value']) for obs in observations if obs.get('value') != '.']
        if not values:
            return {'error': 'No valid CPI data points'}
        
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value
        pct_change = (change / first_value) * 100 if first_value else 0
        
        return {
            'observations': observations,
            'period_start': observations[0]['date'] if observations else None,
            'period_end': observations[-1]['date'] if observations else None,
            'start_value': first_value,
            'end_value': last_value,
            'absolute_change': round(change, 2),
            'percent_change': round(pct_change, 2),
            'data_points': len(values),
            'min_value': round(min(values), 2),
            'max_value': round(max(values), 2)
        }
    
    @instrument(name="fred_get_indicator_trend")
    def get_indicator_trend(
        self,
        indicator: str,
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get trend data for any economic indicator.
        
        Args:
            indicator: Indicator name ('gdp', 'cpi', 'unemployment', 'fed_funds_rate', 'treasury_10y')
            year: Year to get data for (e.g., 2024)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dict with observations and trend analysis
        """
        # Map common names to series IDs
        series_id = self.SERIES.get(indicator.lower())
        if not series_id:
            # Try using the indicator directly as a series ID
            series_id = indicator.upper()
        
        if year:
            observations = self.get_series_for_year(series_id, year)
        elif start_date and end_date:
            observations = self.get_series_for_date_range(
                series_id, start_date, end_date
            )
        else:
            observations = self.get_series_observations(series_id, limit=12)
        
        if not observations:
            return {'error': f'No data available for {indicator}'}
        
        values = [float(obs['value']) for obs in observations if obs.get('value') != '.']
        if not values:
            return {'error': f'No valid data points for {indicator}'}
        
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value
        pct_change = (change / first_value) * 100 if first_value else 0
        
        return {
            'indicator': indicator,
            'series_id': series_id,
            'observations': observations,
            'period_start': observations[0]['date'] if observations else None,
            'period_end': observations[-1]['date'] if observations else None,
            'start_value': first_value,
            'end_value': last_value,
            'absolute_change': round(change, 2),
            'percent_change': round(pct_change, 2),
            'data_points': len(values),
            'min_value': round(min(values), 2),
            'max_value': round(max(values), 2),
            'avg_value': round(sum(values) / len(values), 2)
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

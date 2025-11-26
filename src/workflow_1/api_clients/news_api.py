"""NewsAPI client for financial news.

Provides access to news articles for companies and market sentiment.
Falls back to free sources if NewsAPI key not available.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


class NewsAPIClient:
    """Client for NewsAPI financial news."""
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Financial news sources
    FINANCE_SOURCES = [
        'bloomberg', 'cnbc', 'reuters', 'the-wall-street-journal',
        'financial-times', 'business-insider'
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize NewsAPI client.
        
        Args:
            api_key: NewsAPI key (defaults to NEWS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.has_api_key = bool(self.api_key)
        
        if not self.has_api_key:
            print("Warning: NEWS_API_KEY not set. Using limited free functionality.")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to NewsAPI.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        if not self.has_api_key:
            raise ValueError("NewsAPI key required for this operation")
        
        url = f"{self.BASE_URL}/{endpoint}"
        params['apiKey'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"NewsAPI error: {str(e)}")
    
    def get_company_news(
        self, 
        company_name: str,
        ticker: Optional[str] = None,
        limit: int = 5,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get recent news articles about a company.
        
        Args:
            company_name: Company name to search for
            ticker: Optional ticker symbol to include in search
            limit: Maximum number of articles
            days_back: Number of days of history to search
            
        Returns:
            List of news articles
        """
        if not self.has_api_key:
            return self._get_free_news_placeholder(company_name, ticker)
        
        # Build search query
        query = company_name
        if ticker:
            query = f"{company_name} OR {ticker}"
        
        # Calculate date range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': limit
        }
        
        try:
            data = self._make_request('everything', params)
            return data.get('articles', [])
        except ValueError as e:
            print(f"NewsAPI error: {e}")
            return self._get_free_news_placeholder(company_name, ticker)
    
    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get general market/business news.
        
        Args:
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        if not self.has_api_key:
            return self._get_free_news_placeholder("market", None)
        
        params = {
            'category': 'business',
            'language': 'en',
            'pageSize': limit,
            'country': 'us'
        }
        
        try:
            data = self._make_request('top-headlines', params)
            return data.get('articles', [])
        except ValueError:
            return []
    
    def _get_free_news_placeholder(
        self, 
        company_name: str, 
        ticker: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return placeholder when NewsAPI key not available.
        
        This would be replaced with actual free news scraping in production.
        """
        return [{
            'title': f"News about {company_name} ({ticker})" if ticker else f"News about {company_name}",
            'description': "NewsAPI key not configured. To enable news fetching, set NEWS_API_KEY in your .env file.",
            'url': f"https://www.google.com/search?q={company_name}+news",
            'publishedAt': datetime.now().isoformat(),
            'source': {'name': 'Manual Search Required'}
        }]
    
    def format_articles_for_display(self, articles: List[Dict[str, Any]]) -> str:
        """Format news articles for display.
        
        Args:
            articles: List of article dicts
            
        Returns:
            Formatted string
        """
        if not articles:
            return "No recent news articles found."
        
        output = []
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            source = article.get('source', {}).get('name', 'Unknown')
            date = article.get('publishedAt', '')
            description = article.get('description', '')
            url = article.get('url', '')
            
            # Format date
            try:
                published_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                date_str = published_date.strftime('%Y-%m-%d')
            except:
                date_str = date
            
            output.append(f"{i}. {title}")
            output.append(f"   Source: {source} | Date: {date_str}")
            if description:
                output.append(f"   {description[:150]}...")
            if url:
                output.append(f"   URL: {url}")
            output.append("")
        
        return "\n".join(output)


# Global client instance
_client = None

def get_client() -> NewsAPIClient:
    """Get the global NewsAPI client instance."""
    global _client
    if _client is None:
        _client = NewsAPIClient()
    return _client


if __name__ == "__main__":
    # Test the client
    client = NewsAPIClient()
    
    print("\n=== NewsAPI Client Test ===\n")
    
    if client.has_api_key:
        print("1. Company News (Apple):")
        articles = client.get_company_news("Apple Inc", "AAPL", limit=3)
        print(client.format_articles_for_display(articles))
        
        print("\n2. Market News:")
        market_news = client.get_market_news(limit=3)
        print(client.format_articles_for_display(market_news))
    else:
        print("NewsAPI key not configured. Set NEWS_API_KEY to test.")
        print("\nPlaceholder news:")
        placeholder = client.get_company_news("Tesla", "TSLA")
        print(client.format_articles_for_display(placeholder))

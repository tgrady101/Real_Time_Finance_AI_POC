"""Package initialization for API clients."""

from .ninja_api import NinjaAPIClient, get_client as get_ninja_client
from .fred_api import FREDAPIClient, get_client as get_fred_client
from .news_api import NewsAPIClient, get_client as get_news_client

__all__ = [
    'NinjaAPIClient',
    'FREDAPIClient', 
    'NewsAPIClient',
    'get_ninja_client',
    'get_fred_client',
    'get_news_client',
]

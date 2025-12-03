"""Package initialization for API clients."""

from .fred_api import FREDAPIClient, get_client as get_fred_client

__all__ = [
    'FREDAPIClient', 
    'get_fred_client',
]

"""Package initialization for utilities."""

from .sp500_validator import (
    SP500Validator,
    get_validator,
    is_valid_sp500_ticker,
    validate_ticker_with_message,
    find_ticker_by_name,
    get_sp500_company_info,
)

from .query_classifier import (
    QueryComplexity,
    classify_query,
    get_model_for_query,
    QueryClassifier,
)

from .dynamic_model_callback import (
    create_dynamic_model_callback,
    get_default_dynamic_model_callback,
)

from .agentic_rag import (
    AgenticRAGConfig,
    AgenticSearchResult,
    agentic_search,
    rewrite_query,
    evaluate_results,
    optimized_earnings_search,
)

__all__ = [
    # S&P 500 validation
    'SP500Validator',
    'get_validator',
    'is_valid_sp500_ticker',
    'validate_ticker_with_message',
    'find_ticker_by_name',
    'get_sp500_company_info',
    # Query classification
    'QueryComplexity',
    'classify_query',
    'get_model_for_query',
    'QueryClassifier',
    # Dynamic model routing
    'create_dynamic_model_callback',
    'get_default_dynamic_model_callback',
    # Agentic RAG
    'AgenticRAGConfig',
    'AgenticSearchResult',
    'agentic_search',
    'rewrite_query',
    'evaluate_results',
    'optimized_earnings_search',
]

"""Arize Observability Module.

Provides tracing, evaluations, and monitoring for the finance agent.

Components:
- observability.py: Arize/Phoenix tracing setup
- evaluations.py: Comprehensive evaluation suite

Evaluator Categories:
- Market Data Evaluators: S&P 500 ticker validation, company name resolution, stock price format
- Economic Data Evaluators: GDP/inflation/unemployment response validation
- Routing Evaluators: Model routing accuracy, agent delegation accuracy
- Quality Evaluators: Response completeness, financial accuracy
- LLM-as-a-Judge: Phoenix Evals with Gemini for hallucination, quality, safety
"""

from .evaluations import (
    # Core evaluator functions
    sp500_ticker_validation,
    company_name_resolution,
    stock_price_format,
    response_contains_data,
    tool_usage_verification,
    economic_data_format,
    model_routing_accuracy,
    agent_delegation_accuracy,
    financial_accuracy,
    response_completeness,
    
    # Evaluator lists
    ALL_EVALUATORS,
    MARKET_DATA_EVALUATORS,
    ECONOMIC_EVALUATORS,
    ROUTING_EVALUATORS,
    QUALITY_EVALUATORS,
    
    # Test datasets
    get_finance_agent_test_dataset,
    get_market_data_test_dataset,
    get_economic_test_dataset,
    get_model_routing_test_dataset,
    get_agent_delegation_test_dataset,
    
    # Evaluation runners
    run_local_evaluations,
    run_market_data_evaluations,
    run_economic_evaluations,
    run_routing_evaluations,
    print_evaluation_report,
    
    # LLM-as-a-Judge evaluators
    run_llm_evaluations,
    run_hallucination_eval,
    run_financial_quality_eval,
    run_economic_interpretation_eval,
    run_investment_safety_eval,
    run_all_llm_evaluations,
    
    # Arize integration
    log_evaluations_to_arize,
    create_arize_dataset,
    run_arize_experiment,
    run_quick_experiment,
)

from .observability import (
    setup_arize_tracing,
    instrument,
    is_tracing_enabled,
    add_custom_trace_attributes,
)

__all__ = [
    # Evaluator functions
    "sp500_ticker_validation",
    "company_name_resolution",
    "stock_price_format",
    "response_contains_data",
    "tool_usage_verification",
    "economic_data_format",
    "model_routing_accuracy",
    "agent_delegation_accuracy",
    "financial_accuracy",
    "response_completeness",
    
    # Evaluator lists
    "ALL_EVALUATORS",
    "MARKET_DATA_EVALUATORS",
    "ECONOMIC_EVALUATORS",
    "ROUTING_EVALUATORS",
    "QUALITY_EVALUATORS",
    
    # Test datasets
    "get_finance_agent_test_dataset",
    "get_market_data_test_dataset",
    "get_economic_test_dataset",
    "get_model_routing_test_dataset",
    "get_agent_delegation_test_dataset",
    
    # Evaluation runners
    "run_local_evaluations",
    "run_market_data_evaluations",
    "run_economic_evaluations",
    "run_routing_evaluations",
    "print_evaluation_report",
    
    # LLM evaluators
    "run_llm_evaluations",
    "run_hallucination_eval",
    "run_financial_quality_eval",
    "run_economic_interpretation_eval",
    "run_investment_safety_eval",
    "run_all_llm_evaluations",
    
    # Arize integration
    "log_evaluations_to_arize",
    "create_arize_dataset",
    "run_arize_experiment",
    "run_quick_experiment",
    
    # Observability
    "setup_arize_tracing",
    "instrument",
    "is_tracing_enabled",
    "add_custom_trace_attributes",
]

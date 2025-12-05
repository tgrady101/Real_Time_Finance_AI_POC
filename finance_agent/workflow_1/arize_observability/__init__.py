"""Arize Observability Module.

Provides tracing, evaluations, and monitoring for the finance agent.

Components:
- observability.py: Arize/Phoenix tracing setup  
- evaluations.py: 3-Layer evaluation framework (Automated + LLM-as-Judge)

Usage:
    # Run full evaluation (all 49 queries, all layers)
    python -m finance_agent.workflow_1.arize_observability.evaluations
    
    # Run specific agent type
    python -m ... --market | --economic | --headlines | --vector-store | --portfolio | --utility
    
    # Custom settings
    python -m ... --delay 5.0 --name "my_experiment"
"""

from .evaluations import (
    # Test dataset
    get_test_dataset,
    
    # Layer 1: Automated Checks
    LAYER1_EVALUATORS,
    run_layer1_evaluations,
    
    # Layer 3: LLM-as-Judge
    run_layer3_evaluations,
    
    # Execution
    query_live_agent,
    log_to_arize,
    print_summary,
)

from .observability import (
    setup_arize_tracing,
    instrument,
    is_tracing_enabled,
    add_custom_trace_attributes,
)

__all__ = [
    # Test datasets
    "get_test_dataset",
    
    # Layer 1: Automated Checks
    "LAYER1_EVALUATORS",
    "run_layer1_evaluations",
    
    # Layer 3: LLM-as-Judge
    "run_layer3_evaluations",
    
    # Execution
    "query_live_agent",
    "log_to_arize",
    "print_summary",
    
    # Observability
    "setup_arize_tracing",
    "instrument",
    "is_tracing_enabled",
    "add_custom_trace_attributes",
]

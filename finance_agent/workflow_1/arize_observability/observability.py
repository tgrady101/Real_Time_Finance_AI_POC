"""
Arize Cloud Tracing Configuration

Sets up OpenTelemetry tracing for Google ADK agents with Arize cloud observability.
Captures:
- All ADK agent executions
- Tool calls (Yahoo Finance MCP, API clients)
- Agent-to-agent interactions
- Complete workflow execution
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Configuration - reads from .env file
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME")

# Global tracer provider
_TRACER_PROVIDER = None
_INITIALIZED = False


def setup_arize_tracing(project_name: Optional[str] = None):
    """
    Initialize Arize cloud tracing for the financial chatbot system.
    
    Call this BEFORE creating any agents to enable automatic tracing.
    
    This will trace:
    - All ADK agent executions
    - Tool calls (MCP tools, API calls, etc.)
    - Agent-to-agent delegation
    - LLM requests and responses
    
    Args:
        project_name: Optional project name override
    
    Returns:
        tracer_provider: The configured OpenTelemetry tracer provider
    """
    global _TRACER_PROVIDER, _INITIALIZED
    
    if _INITIALIZED:
        return _TRACER_PROVIDER
    
    _INITIALIZED = True
    
    print("\n" + "="*70)
    print("INITIALIZING ARIZE CLOUD TRACING")
    print("="*70)
    
    # Check for required configuration
    if not ARIZE_API_KEY:
        print("⚠️  WARNING: ARIZE_API_KEY not set in .env")
        print("   Tracing will not be sent to Arize cloud")
        print("   Get from: https://app.arize.com (Space Settings > API Keys)")
        return None
    
    if not ARIZE_SPACE_ID:
        print("⚠️  WARNING: ARIZE_SPACE_ID not set in .env")
        print("   Tracing will not be sent to Arize cloud")
        print("   Get from: https://app.arize.com (Space Settings)")
        return None
    
    if not ARIZE_PROJECT_NAME and not project_name:
        print("⚠️  WARNING: ARIZE_PROJECT_NAME not set in .env")
        print("   Tracing will not be sent to Arize cloud")
        print("   Set ARIZE_PROJECT_NAME in your .env file")
        return None
    
    try:
        from arize.otel import register
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        
        # At this point, all required vars are validated (not None/empty)
        proj_name = project_name or ARIZE_PROJECT_NAME
        assert ARIZE_SPACE_ID is not None
        assert ARIZE_API_KEY is not None
        assert proj_name is not None
        
        # Register Arize cloud tracer
        _TRACER_PROVIDER = register(
            space_id=ARIZE_SPACE_ID,
            api_key=ARIZE_API_KEY,
            project_name=proj_name,
        )
        
        # Instrument Google ADK for automatic tracing
        GoogleADKInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
        
        print(f"✅ Arize cloud tracing initialized!")
        print(f"   Project: {proj_name}")
        print(f"   Space ID: {ARIZE_SPACE_ID[:20]}...")
        print(f"\n   Auto-instrumentation captures:")
        print(f"   • ADK agent executions")
        print(f"   • Tool function calls")
        print(f"   • Agent workflow orchestration")
        print(f"   • LLM requests/responses")
        print(f"\n   View traces at: https://app.arize.com")
        print("="*70 + "\n")
        
        return _TRACER_PROVIDER
    
    except ImportError as e:
        print(f"❌ ERROR: Required package not installed: {e}")
        print("   Install: pip install arize-otel openinference-instrumentation-google-adk")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Arize tracing: {e}")
        return None


def is_tracing_enabled() -> bool:
    """Check if tracing is properly configured and enabled."""
    return _TRACER_PROVIDER is not None


def add_custom_trace_attributes():
    """
    Add custom attributes to traces for better filtering and analysis.
    """
    from opentelemetry import trace
    
    # Get the current span
    span = trace.get_current_span()
    
    if span:
        # Add custom attributes
        span.set_attribute("environment", os.getenv("ENVIRONMENT", "development"))
        span.set_attribute("gcp_project", os.getenv("GOOGLE_CLOUD_PROJECT", "unknown"))
        span.set_attribute("model", "gemini-2.0-flash-exp")
        span.set_attribute("datastore", os.getenv("DATA_STORE_ID", "finance-data"))


# Legacy compatibility - keep instrument decorator for API clients
def instrument(name: Optional[str] = None):
    """
    Decorator to manually instrument functions with OpenTelemetry tracing.
    
    Note: Google ADK agents are auto-instrumented, so this is primarily
    for API client methods and utility functions.
    
    Args:
        name: Optional name for the span. If not provided, function name is used.
    """
    def decorator(func):
        import functools
        from opentelemetry import trace
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracer
            tracer = trace.get_tracer(__name__)
            span_name = name or func.__name__
            
            with tracer.start_as_current_span(span_name) as span:
                # Record input arguments as attributes
                try:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                except Exception:
                    pass
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise e
        return wrapper
    return decorator

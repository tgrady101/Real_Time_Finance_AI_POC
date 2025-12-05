"""Dynamic Model Callback for per-request model selection.

This callback intercepts each LLM request and can dynamically change 
the model based on query complexity, cost constraints, or other factors.

Based on Google ADK's before_model_callback pattern.

Includes OpenTelemetry tracing for Arize observability.
"""

from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from .query_classifier import classify_query, QueryComplexity

# OpenTelemetry for Arize tracing
try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def create_dynamic_model_callback(fast_model: str, complex_model: str):
    """Create a before_model_callback that dynamically selects models.
    
    This callback analyzes the user's query and switches to the appropriate
    model BEFORE the LLM call is made.
    
    Args:
        fast_model: Model for simple queries (e.g., gemini-2.5-flash)
        complex_model: Model for complex queries (e.g., gemini-3-pro-preview)
        
    Returns:
        A callback function compatible with LlmAgent.before_model_callback
    """
    
    async def dynamic_model_selector(
        callback_context: CallbackContext, 
        llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Dynamically select model based on query complexity.
        
        This callback:
        1. Extracts the latest user message
        2. Classifies its complexity
        3. Modifies llm_request.model to use the appropriate model
        4. Returns None to continue with the (now modified) request
        """
        # Extract latest user message
        user_query = ""
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                if content.role == 'user' and content.parts:
                    if content.parts[0].text:
                        user_query = content.parts[0].text
                        break
        
        if not user_query:
            # No user query found, use fast model as default
            llm_request.model = fast_model
            return None
        
        # Classify query complexity
        complexity, reason, confidence = classify_query(user_query)
        
        # Select model based on complexity
        if complexity == QueryComplexity.COMPLEX:
            selected_model = complex_model
        else:
            selected_model = fast_model
        
        # Log the routing decision
        agent_name = callback_context.agent_name
        print(f"[ROUTE] [{agent_name}] Dynamic routing: {selected_model} ({confidence:.0%})")
        print(f"   Query: {user_query[:60]}...")
        print(f"   Reason: {reason[:50]}...")
        
        # Store routing info in state for observability
        callback_context.state["model_routing"] = {
            "selected_model": selected_model,
            "complexity": complexity.value,
            "confidence": confidence,
            "reason": reason,
        }
        
        # Add OpenTelemetry span attributes for Arize tracing
        if OTEL_AVAILABLE:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("model_routing.selected_model", selected_model)
                current_span.set_attribute("model_routing.complexity", complexity.value)
                current_span.set_attribute("model_routing.confidence", confidence)
                current_span.set_attribute("model_routing.reason", reason)
                current_span.set_attribute("model_routing.agent", agent_name)
                current_span.set_attribute("model_routing.query_preview", user_query[:100])
        
        # MODIFY the request's model - this is the key part!
        llm_request.model = selected_model
        
        # Return None to proceed with the (modified) request
        return None
    
    return dynamic_model_selector


# Pre-configured callback using Config values
def get_default_dynamic_model_callback():
    """Get the default dynamic model callback using Config settings."""
    from ..config import Config
    return create_dynamic_model_callback(
        fast_model=Config.MODEL_FAST,
        complex_model=Config.MODEL_COMPLEX,
    )


if __name__ == "__main__":
    print("Dynamic Model Callback Module")
    print("=" * 50)
    print("""
Usage:
    from workflow_1.utils.dynamic_model_callback import get_default_dynamic_model_callback
    
    agent = LlmAgent(
        model="gemini-2.5-flash",  # Default, will be overridden
        name="my_agent",
        instruction="...",
        before_model_callback=get_default_dynamic_model_callback()
    )
    
The callback will:
1. Analyze each user query
2. Classify as SIMPLE or COMPLEX
3. Override the model in the request
4. Log the routing decision
""")

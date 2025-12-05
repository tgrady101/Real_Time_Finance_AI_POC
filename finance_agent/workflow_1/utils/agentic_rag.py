"""Agentic RAG Query Optimization.

Implements iterative query rewriting with LLM-as-Judge evaluation to optimize
chunk retrieval quality for earnings call searches.

Pattern:
1. Rewrite query for optimal retrieval (expand terms, add context)
2. Execute search with optimized query
3. Evaluate result quality using LLM-as-Judge
4. If quality below threshold, refine query based on feedback
5. Repeat until quality threshold met or max iterations reached
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AgenticRAGConfig:
    """Configuration for agentic RAG query optimization."""
    max_iterations: int = 3
    quality_threshold: float = 0.7  # 0-1 scale
    rewriter_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"
    enable_logging: bool = True
    

DEFAULT_CONFIG = AgenticRAGConfig()


# =============================================================================
# Query Rewriter
# =============================================================================

QUERY_REWRITE_PROMPT = """You are a query optimization expert for financial earnings call search.

TASK: Rewrite the user query to maximize retrieval of relevant earnings call content.

ORIGINAL QUERY: {original_query}
ITERATION: {iteration} of {max_iterations}
{feedback_section}

OPTIMIZATION STRATEGIES:
1. **Expand financial terms**: "AI" -> "artificial intelligence, machine learning, AI strategy"
2. **Add speaker context**: If asking about strategy, add "CEO, CFO, management"
3. **Include synonyms**: "guidance" -> "guidance, outlook, forecast, expectations"
4. **Specify timeframe**: Add quarter/year context if relevant
5. **Extract key entities**: Company names, tickers, specific metrics
{iteration_specific_guidance}

OUTPUT FORMAT (JSON):
{{
    "optimized_query": "your optimized search query",
    "extracted_ticker": "TICKER or null if not specified",
    "extracted_company": "Company Name or null",
    "extracted_quarter": "Q2_2025 or Q3_2025 or null",
    "search_terms": ["list", "of", "key", "search", "terms"],
    "reasoning": "Brief explanation of optimization strategy"
}}

Return ONLY valid JSON, no markdown code blocks."""


ITERATION_GUIDANCE = {
    0: """
FIRST ITERATION: Focus on query expansion and term disambiguation.
- Expand acronyms and abbreviations
- Add relevant synonyms
- Extract any mentioned entities (companies, tickers)""",
    
    1: """
SECOND ITERATION: Based on initial results, focus on precision.
- Narrow down to most relevant terms
- Add specific financial terminology
- Consider alternative phrasings executives might use""",
    
    2: """
FINAL ITERATION: Maximum precision attempt.
- Use exact phrases executives commonly say
- Focus on the most distinctive terms
- Consider earnings call-specific language patterns"""
}


def rewrite_query(
    original_query: str,
    iteration: int,
    feedback: Optional[str] = None,
    config: AgenticRAGConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """Rewrite query for optimal retrieval.
    
    Args:
        original_query: The user's original query
        iteration: Current iteration number (0-indexed)
        feedback: Optional feedback from previous iteration's evaluation
        config: Agentic RAG configuration
        
    Returns:
        Dict with optimized_query, extracted entities, and reasoning
    """
    # Build feedback section
    feedback_section = ""
    if feedback:
        feedback_section = f"""
PREVIOUS ITERATION FEEDBACK:
{feedback}

Use this feedback to improve the query. Focus on addressing the specific issues mentioned."""

    # Get iteration-specific guidance
    iteration_guidance = ITERATION_GUIDANCE.get(
        iteration, 
        ITERATION_GUIDANCE[2]  # Use final iteration guidance for any additional iterations
    )
    
    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=original_query,
        iteration=iteration + 1,
        max_iterations=config.max_iterations,
        feedback_section=feedback_section,
        iteration_specific_guidance=iteration_guidance
    )
    
    try:
        # Initialize Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)
        
        model = GenerativeModel(config.rewriter_model)
        generation_config = GenerationConfig(temperature=0.0)
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Parse JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        if config.enable_logging:
            print(f"   [QueryRewrite] Iteration {iteration + 1}: {result.get('optimized_query', '')[:60]}...")
            
        return result
        
    except Exception as e:
        # Fallback: return original query with basic expansion
        if config.enable_logging:
            print(f"   [QueryRewrite] Error: {e}, using fallback")
        return {
            "optimized_query": original_query,
            "extracted_ticker": None,
            "extracted_company": None,
            "extracted_quarter": None,
            "search_terms": original_query.split(),
            "reasoning": f"Fallback due to error: {e}"
        }


# =============================================================================
# Result Quality Evaluator (LLM-as-Judge)
# =============================================================================

QUALITY_EVALUATION_PROMPT = """You are evaluating the quality of search results from an earnings call database.

ORIGINAL USER QUERY: {original_query}
SEARCH QUERY USED: {search_query}

SEARCH RESULTS:
{results}

Evaluate the results on these criteria (0.0 to 1.0 each):

1. **RELEVANCE** (0.0-1.0): Do the results directly address the user's question?
   - 1.0: Perfectly relevant, directly answers the query
   - 0.7: Mostly relevant, some useful information
   - 0.4: Partially relevant, tangentially related
   - 0.0: Irrelevant results

2. **COMPLETENESS** (0.0-1.0): Do results provide sufficient information to answer the query?
   - 1.0: Complete answer with supporting details
   - 0.7: Good coverage, minor gaps
   - 0.4: Partial coverage, significant gaps
   - 0.0: Insufficient information

3. **SPECIFICITY** (0.0-1.0): Are results specific to the requested topic/company?
   - 1.0: Highly specific, exactly what was asked
   - 0.7: Specific but includes some tangential content
   - 0.4: General/broad results
   - 0.0: Wrong topic or company

4. **SOURCE_QUALITY** (0.0-1.0): Are sources properly attributed with speaker/quarter?
   - 1.0: Clear attribution with speaker, quarter, year
   - 0.7: Partial attribution
   - 0.4: Minimal attribution
   - 0.0: No attribution

OUTPUT FORMAT (JSON):
{{
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "specificity": 0.0-1.0,
    "source_quality": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "feedback": "Specific feedback for query improvement if score < 0.7",
    "missing_aspects": ["list", "of", "what's", "missing"],
    "suggested_terms": ["terms", "that", "might", "help"]
}}

Return ONLY valid JSON."""


def evaluate_results(
    original_query: str,
    search_query: str,
    results: str,
    config: AgenticRAGConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """Evaluate search result quality using LLM-as-Judge.
    
    Args:
        original_query: The user's original query
        search_query: The query used for search (may be rewritten)
        results: Formatted search results string
        config: Agentic RAG configuration
        
    Returns:
        Dict with quality scores and feedback
    """
    # Handle empty results
    if not results or "No earnings call content found" in results:
        return {
            "relevance": 0.0,
            "completeness": 0.0,
            "specificity": 0.0,
            "source_quality": 0.0,
            "overall_score": 0.0,
            "feedback": "No results returned. Try broader search terms or different company/quarter filters.",
            "missing_aspects": ["any relevant content"],
            "suggested_terms": []
        }
    
    prompt = QUALITY_EVALUATION_PROMPT.format(
        original_query=original_query,
        search_query=search_query,
        results=results[:3000]  # Truncate for token limits
    )
    
    try:
        # Initialize Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)
        
        model = GenerativeModel(config.judge_model)
        generation_config = GenerationConfig(temperature=0.0)
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Parse JSON response
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        # Ensure overall_score is calculated
        if "overall_score" not in result:
            scores = [
                result.get("relevance", 0),
                result.get("completeness", 0),
                result.get("specificity", 0),
                result.get("source_quality", 0)
            ]
            result["overall_score"] = sum(scores) / len(scores)
        
        if config.enable_logging:
            print(f"   [QualityEval] Score: {result['overall_score']:.2f} "
                  f"(R:{result.get('relevance', 0):.1f} C:{result.get('completeness', 0):.1f} "
                  f"S:{result.get('specificity', 0):.1f})")
            
        return result
        
    except Exception as e:
        if config.enable_logging:
            print(f"   [QualityEval] Error: {e}, using heuristic evaluation")
        
        # Fallback: heuristic evaluation
        has_content = len(results) > 100
        has_speaker = any(s in results for s in ["CEO", "CFO", "COO"])
        has_quarter = "Q2" in results or "Q3" in results
        
        score = (0.4 if has_content else 0) + (0.3 if has_speaker else 0) + (0.3 if has_quarter else 0)
        
        return {
            "relevance": score,
            "completeness": score,
            "specificity": score,
            "source_quality": 1.0 if has_speaker else 0.5,
            "overall_score": score,
            "feedback": "Heuristic evaluation used",
            "missing_aspects": [],
            "suggested_terms": []
        }


# =============================================================================
# Agentic Search Orchestrator
# =============================================================================

@dataclass
class AgenticSearchResult:
    """Result from agentic search with metadata."""
    results: str
    final_query: str
    original_query: str
    iterations_used: int
    final_quality_score: float
    quality_history: List[Dict[str, Any]]
    query_history: List[str]
    

def agentic_search(
    query: str,
    search_function,  # The actual search function to call
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 5,
    config: AgenticRAGConfig = DEFAULT_CONFIG
) -> AgenticSearchResult:
    """Perform agentic search with iterative query optimization.
    
    This function wraps a search function and iteratively optimizes the query
    using LLM-based rewriting and quality evaluation.
    
    Args:
        query: Original user query
        search_function: Function to call for search (e.g., search_earnings_calls)
        ticker: Optional ticker filter
        company: Optional company filter
        quarter: Optional quarter filter
        year: Optional year filter
        max_results: Maximum results to return
        config: Agentic RAG configuration
        
    Returns:
        AgenticSearchResult with results and optimization metadata
    """
    if config.enable_logging:
        print(f"\n[AgenticRAG] Starting optimization for: '{query}'")
        print(f"   Max iterations: {config.max_iterations}, Threshold: {config.quality_threshold}")
    
    quality_history = []
    query_history = [query]
    best_results = None
    best_score = 0.0
    best_query = query
    
    for iteration in range(config.max_iterations):
        if config.enable_logging:
            print(f"\n   --- Iteration {iteration + 1}/{config.max_iterations} ---")
        
        # Get feedback from previous iteration (if any)
        feedback = None
        if quality_history:
            last_eval = quality_history[-1]
            feedback = last_eval.get("feedback", "")
            if last_eval.get("missing_aspects"):
                feedback += f"\nMissing: {', '.join(last_eval['missing_aspects'])}"
            if last_eval.get("suggested_terms"):
                feedback += f"\nSuggested terms: {', '.join(last_eval['suggested_terms'])}"
        
        # Rewrite query
        rewrite_result = rewrite_query(
            original_query=query,
            iteration=iteration,
            feedback=feedback,
            config=config
        )
        
        optimized_query = rewrite_result.get("optimized_query", query)
        query_history.append(optimized_query)
        
        # Extract any entities from rewrite (to supplement filters)
        extracted_ticker = ticker or rewrite_result.get("extracted_ticker")
        extracted_company = company or rewrite_result.get("extracted_company")
        extracted_quarter = quarter or rewrite_result.get("extracted_quarter")
        
        # Execute search
        try:
            results = search_function(
                query=optimized_query,
                ticker=extracted_ticker,
                company=extracted_company,
                quarter=extracted_quarter,
                year=year,
                max_results=max_results
            )
        except Exception as e:
            if config.enable_logging:
                print(f"   [Search] Error: {e}")
            results = f"Search error: {e}"
        
        # Evaluate quality
        evaluation = evaluate_results(
            original_query=query,
            search_query=optimized_query,
            results=results,
            config=config
        )
        quality_history.append(evaluation)
        
        current_score = evaluation.get("overall_score", 0)
        
        # Track best results
        if current_score > best_score:
            best_score = current_score
            best_results = results
            best_query = optimized_query
        
        # Check if quality threshold met
        if current_score >= config.quality_threshold:
            if config.enable_logging:
                print(f"\n[AgenticRAG] Quality threshold met! Score: {current_score:.2f}")
            break
    
    if config.enable_logging:
        print(f"\n[AgenticRAG] Complete. Best score: {best_score:.2f}, Iterations: {len(quality_history)}")
    
    return AgenticSearchResult(
        results=best_results or results,
        final_query=best_query,
        original_query=query,
        iterations_used=len(quality_history),
        final_quality_score=best_score,
        quality_history=quality_history,
        query_history=query_history
    )


# =============================================================================
# Convenience Function for Direct Use
# =============================================================================

def optimized_earnings_search(
    query: str,
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 5,
    max_iterations: int = 3,
    quality_threshold: float = 0.7,
    enable_logging: bool = False
) -> str:
    """Optimized earnings call search with agentic query rewriting.
    
    This is a convenience wrapper that uses agentic_search with the 
    search_earnings_calls function.
    
    Args:
        query: Search query for earnings calls
        ticker: Optional ticker filter
        company: Optional company filter  
        quarter: Optional quarter filter
        year: Optional year filter
        max_results: Maximum results to return
        max_iterations: Max query optimization iterations
        quality_threshold: Score threshold to stop optimization (0-1)
        enable_logging: Whether to print optimization progress
        
    Returns:
        Formatted search results with best quality found
    """
    # Import search function here to avoid circular imports
    from finance_agent.workflow_1.agents.vector_store_agent import search_earnings_calls
    
    config = AgenticRAGConfig(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        enable_logging=enable_logging
    )
    
    result = agentic_search(
        query=query,
        search_function=search_earnings_calls,
        ticker=ticker,
        company=company,
        quarter=quarter,
        year=year,
        max_results=max_results,
        config=config
    )
    
    # Append optimization metadata as a note
    if enable_logging:
        metadata = (
            f"\n\n---\n*Query optimized in {result.iterations_used} iterations. "
            f"Quality score: {result.final_quality_score:.2f}*"
        )
        return result.results + metadata
    
    return result.results

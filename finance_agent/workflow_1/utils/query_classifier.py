"""Query complexity classifier for dynamic model routing.

Analyzes user queries to determine whether they need a fast/simple model
or a complex/powerful model, enabling cost-effective model routing.

Complexity Signals:
- SIMPLE (use MODEL_FAST): Single data point lookups, yes/no questions, greetings
- COMPLEX (use MODEL_COMPLEX): Multi-step analysis, comparisons, reasoning, predictions
"""

import re
from typing import Tuple
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels for model routing."""
    SIMPLE = "simple"      # Use MODEL_FAST
    COMPLEX = "complex"    # Use MODEL_COMPLEX


# Keywords/patterns that indicate SIMPLE queries (fast model)
SIMPLE_PATTERNS = [
    # Greetings and meta
    r'\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
    r'\b(what can you do|help|capabilities)\b',
    
    # Single data point lookups
    r'\b(what is|what\'s|whats|get|show|tell me)\s+(the\s+)?(price|stock price|current price)\b',
    r'\b(price of|quote for|ticker for)\b',
    r'\b(is .+ (in|part of) (the\s+)?s&p|s&p 500)\b',
    
    # Simple yes/no validations
    r'\b(is .+ valid|check if|verify)\b',
    r'\b(list|show)\s+(all\s+)?(tickers?|stocks?|companies)\b',
]

# Keywords/patterns that indicate COMPLEX queries (powerful model)
COMPLEX_PATTERNS = [
    # Multi-step analysis
    r'\b(analyze|analysis|evaluate|assessment)\b',
    r'\b(compare|comparison|versus|vs\.?|better than)\b',
    r'\b(why|how come|explain|reasoning)\b',
    
    # Financial analysis
    r'\b(valuation|undervalued|overvalued|fair value)\b',
    r'\b(trend|pattern|technical analysis|moving average)\b',
    r'\b(risk|volatility|beta|sharpe)\b',
    r'\b(forecast|predict|projection|outlook)\b',
    r'\b(recommend|should i|investment advice)\b',
    
    # Deep financials
    r'\b(fundamentals?|financial health|balance sheet analysis)\b',
    r'\b(earnings|revenue growth|profit margin|cash flow analysis)\b',
    r'\b(p/e ratio|pe ratio|price.to.earnings).*(good|bad|high|low|compare)\b',
    
    # Options complexity
    r'\b(options strategy|call spread|put spread|straddle|strangle)\b',
    r'\b(implied volatility|greeks|delta|gamma|theta|vega)\b',
    r'\b(options? analysis|options? chain analysis)\b',
    
    # Portfolio/multi-stock
    r'\b(portfolio|diversif|allocation|sector exposure)\b',
    r'\b(multiple|several|these|those)\s+(stocks?|companies|tickers?)\b',
    r'\b(top \d+|best \d+|worst \d+)\b',
    
    # Time-based analysis
    r'\b(over time|historically|year.over.year|yoy|quarter.over.quarter)\b',
    r'\b(performance|return|growth)\s+(over|since|from|last)\b',
]

# Word count thresholds
SIMPLE_WORD_THRESHOLD = 8   # Queries under this are likely simple
COMPLEX_WORD_THRESHOLD = 20  # Queries over this are likely complex


def classify_query(query: str) -> Tuple[QueryComplexity, str, float]:
    """Classify a query's complexity for model routing.
    
    Args:
        query: The user's query string
        
    Returns:
        Tuple of (complexity, reason, confidence)
        - complexity: QueryComplexity.SIMPLE or QueryComplexity.COMPLEX
        - reason: Human-readable explanation
        - confidence: 0.0 to 1.0 confidence score
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)
    
    # Score accumulators
    simple_score = 0.0
    complex_score = 0.0
    reasons = []
    
    # Check for simple patterns
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, query_lower):
            simple_score += 0.3
            reasons.append(f"simple pattern: {pattern[:30]}")
    
    # Check for complex patterns
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, query_lower):
            complex_score += 0.3
            reasons.append(f"complex pattern: {pattern[:30]}")
    
    # Word count heuristics
    if word_count <= SIMPLE_WORD_THRESHOLD:
        simple_score += 0.2
        reasons.append(f"short query ({word_count} words)")
    elif word_count >= COMPLEX_WORD_THRESHOLD:
        complex_score += 0.2
        reasons.append(f"long query ({word_count} words)")
    
    # Question complexity signals
    question_words = sum(1 for w in ['why', 'how', 'explain', 'analyze'] if w in query_lower)
    if question_words >= 2:
        complex_score += 0.2
        reasons.append("multiple analytical question words")
    
    # Multiple tickers/companies mentioned
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    potential_tickers = re.findall(ticker_pattern, query)
    if len(potential_tickers) >= 3:
        complex_score += 0.2
        reasons.append(f"multiple tickers mentioned ({len(potential_tickers)})")
    
    # Conjunctions suggesting multi-part queries
    conjunctions = sum(1 for w in ['and', 'also', 'plus', 'as well as', 'compared to'] if w in query_lower)
    if conjunctions >= 2:
        complex_score += 0.15
        reasons.append("multi-part query (conjunctions)")
    
    # Determine final classification
    if complex_score > simple_score:
        confidence = min(0.95, 0.5 + (complex_score - simple_score))
        reason = f"Complex: {', '.join(reasons[:3])}"
        return QueryComplexity.COMPLEX, reason, confidence
    elif simple_score > complex_score:
        confidence = min(0.95, 0.5 + (simple_score - complex_score))
        reason = f"Simple: {', '.join(reasons[:3])}"
        return QueryComplexity.SIMPLE, reason, confidence
    else:
        # Default to simple for ambiguous queries (cost savings)
        return QueryComplexity.SIMPLE, "Ambiguous - defaulting to simple", 0.5


def get_model_for_query(query: str, fast_model: str, complex_model: str) -> Tuple[str, QueryComplexity, str]:
    """Get the appropriate model for a query.
    
    Args:
        query: The user's query string
        fast_model: Model name for simple queries
        complex_model: Model name for complex queries
        
    Returns:
        Tuple of (model_name, complexity, reason)
    """
    complexity, reason, confidence = classify_query(query)
    
    if complexity == QueryComplexity.COMPLEX:
        return complex_model, complexity, reason
    else:
        return fast_model, complexity, reason


# Pre-compiled classifier for performance
class QueryClassifier:
    """Cached query classifier for repeated use."""
    
    def __init__(self, fast_model: str, complex_model: str):
        self.fast_model = fast_model
        self.complex_model = complex_model
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in SIMPLE_PATTERNS]
        self._complex_patterns = [re.compile(p, re.IGNORECASE) for p in COMPLEX_PATTERNS]
    
    def classify(self, query: str) -> Tuple[str, QueryComplexity, str]:
        """Classify query and return appropriate model."""
        return get_model_for_query(query, self.fast_model, self.complex_model)
    
    def is_complex(self, query: str) -> bool:
        """Quick check if query is complex."""
        complexity, _, _ = classify_query(query)
        return complexity == QueryComplexity.COMPLEX


if __name__ == "__main__":
    # Test the classifier
    test_queries = [
        # Simple queries
        "What is the price of Apple?",
        "Hi there!",
        "Get AAPL stock price",
        "Is MSFT in the S&P 500?",
        "Show me the ticker for Microsoft",
        
        # Complex queries
        "Compare Apple and Microsoft's P/E ratios and explain which is a better value",
        "Analyze NVDA's revenue growth over the past 5 years and predict future performance",
        "Why is Tesla's stock volatile and how does it compare to other EV makers?",
        "What options strategy would you recommend for AAPL given current implied volatility?",
        "Evaluate the risk-adjusted returns of my portfolio: AAPL, MSFT, GOOGL, AMZN",
    ]
    
    print("Query Classification Test Results")
    print("=" * 70)
    
    for query in test_queries:
        complexity, reason, confidence = classify_query(query)
        model = "FAST" if complexity == QueryComplexity.SIMPLE else "COMPLEX"
        print(f"\nQuery: {query[:60]}...")
        print(f"  → {model} (confidence: {confidence:.0%})")
        print(f"  → Reason: {reason}")

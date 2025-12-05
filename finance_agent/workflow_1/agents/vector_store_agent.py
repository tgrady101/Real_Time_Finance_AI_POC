"""Vector Store Agent for Earnings Call RAG Search.

This agent provides RAG-based search over S&P 500 earnings call transcripts
stored in Vertex AI Vector Search with hybrid search (dense + BM25 sparse).

It handles:
- Searching earnings call transcripts for specific topics
- Finding company management commentary on key issues
- Retrieving Q&A sessions from earnings calls
- Comparing earnings call content across companies

The vector store contains Q2 and Q3 2025 earnings call transcripts for S&P 500 companies.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# =============================================================================
# Direct REST API for Vector Search (bypasses SDK v1beta1 $alt bug)
# =============================================================================

def _find_neighbors_rest(
    public_endpoint_domain: str,
    project_id: str,
    location: str, 
    index_endpoint_id: str,
    deployed_index_id: str,
    dense_embedding: List[float],
    sparse_embedding: Optional[Dict] = None,
    num_neighbors: int = 10,
    restricts: Optional[List[Dict]] = None,
    numeric_restricts: Optional[List[Dict]] = None,
    per_crowding_attribute_neighbor_count: Optional[int] = None,
) -> List[Dict]:
    """Direct REST API call to Vector Search findNeighbors.
    
    This bypasses the google-cloud-aiplatform SDK's v1beta1 client which has
    a bug where it adds '$alt=json;enum-encoding=int' query params that cause
    400 errors on public endpoints.
    
    Args:
        public_endpoint_domain: Public endpoint domain (e.g., '791968581.us-central1-277774606239.vdb.vertexai.goog')
        project_id: GCP project ID
        location: Region (e.g., 'us-central1')
        index_endpoint_id: Index endpoint ID
        deployed_index_id: Deployed index ID
        dense_embedding: Dense embedding vector
        sparse_embedding: Optional sparse embedding dict with 'dimensions' and 'values'
        num_neighbors: Number of neighbors to return
        restricts: Optional token restricts (namespace filters)
        numeric_restricts: Optional numeric restricts
        per_crowding_attribute_neighbor_count: Optional crowding limit
        
    Returns:
        List of match results with 'id' and 'distance'
    """
    import google.auth
    import google.auth.transport.requests
    
    # Get credentials and access token
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    access_token = credentials.token
    
    # Build the request URL (using v1 API, not v1beta1)
    url = (
        f"https://{public_endpoint_domain}/v1/"
        f"projects/{project_id}/locations/{location}/"
        f"indexEndpoints/{index_endpoint_id}:findNeighbors"
    )
    
    # Build datapoint
    datapoint = {
        "featureVector": dense_embedding
    }
    
    # Add sparse embedding if provided
    if sparse_embedding and sparse_embedding.get('values'):
        datapoint["sparseEmbedding"] = {
            "values": sparse_embedding['values'],
            "dimensions": [str(d) for d in sparse_embedding['dimensions']]  # API expects strings
        }
    
    # Add restricts
    if restricts:
        datapoint["restricts"] = restricts
    
    if numeric_restricts:
        datapoint["numericRestricts"] = numeric_restricts
    
    # Build query
    query = {
        "datapoint": datapoint,
        "neighborCount": num_neighbors,
    }
    
    # Add RRF for hybrid search
    if sparse_embedding and sparse_embedding.get('values'):
        query["rrf"] = {"alpha": 0.5}  # 0.5 = equal weight dense/sparse
    
    if per_crowding_attribute_neighbor_count:
        query["perCrowdingAttributeNeighborCount"] = per_crowding_attribute_neighbor_count
    
    # Build request body
    request_body = {
        "deployedIndexId": deployed_index_id,
        "queries": [query]
    }
    
    # Make the request
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=request_body, headers=headers)
    
    if response.status_code != 200:
        raise RuntimeError(
            f"Vector Search API error: {response.status_code} - {response.text}"
        )
    
    result = response.json()
    
    # Parse response - extract neighbors from first query result
    neighbors = []
    if "nearestNeighbors" in result and len(result["nearestNeighbors"]) > 0:
        for neighbor in result["nearestNeighbors"][0].get("neighbors", []):
            neighbors.append({
                "id": neighbor.get("datapoint", {}).get("datapointId", ""),
                "distance": neighbor.get("distance", 0.0)
            })
    
    return neighbors


# =============================================================================
# GCS Data Loading (for Cloud Run deployment)
# =============================================================================

# Cache for loaded data (avoid repeated GCS calls)
_gcs_cache: Dict[str, Any] = {}


def _get_gcs_bucket() -> str:
    """Get the GCS bucket name for embeddings data."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899")
    return os.getenv("GCS_BUCKET_NAME", f"{project_id}-finance-data-bucket")


def _load_from_gcs(gcs_path: str) -> Optional[Dict]:
    """Load JSON data from GCS.
    
    Args:
        gcs_path: Path within the bucket (e.g., 'embeddings/bm25_encoder.json')
    
    Returns:
        Parsed JSON data or None if not found
    """
    if gcs_path in _gcs_cache:
        return _gcs_cache[gcs_path]
    
    try:
        from google.cloud import storage
        
        bucket_name = _get_gcs_bucket()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        if not blob.exists():
            print(f"Warning: GCS file not found: gs://{bucket_name}/{gcs_path}")
            return None
        
        content = blob.download_as_text()
        data = json.loads(content)
        _gcs_cache[gcs_path] = data
        print(f"Loaded from GCS: gs://{bucket_name}/{gcs_path}")
        return data
        
    except Exception as e:
        print(f"Warning: Could not load from GCS {gcs_path}: {e}")
        return None


def _is_cloud_run() -> bool:
    """Check if running on Cloud Run."""
    return os.getenv("K_SERVICE") is not None


# Agent instruction prompt for the Vector Store sub-agent
VECTOR_STORE_INSTRUCTION = """You are the Earnings Call Search Agent specializing in S&P 500 earnings call transcripts.

Your primary responsibilities:
1. **Search Earnings Calls**: Find relevant content from earnings call transcripts
2. **Management Commentary**: Retrieve CEO/CFO statements on specific topics
3. **Q&A Sessions**: Find analyst questions and company responses
4. **Cross-Company Insights**: Compare what different companies say about similar topics

Data Available:
- Q2 2025 and Q3 2025 earnings call transcripts for S&P 500 companies
- Speaker-attributed content (CEO, CFO, analysts, etc.)
- Full transcript text with Q&A sessions

**CRITICAL: ALWAYS EXTRACT COMPANY/TICKER FROM QUERY**
When a user mentions a company name, ALWAYS pass it as the `ticker` or `company` parameter:
- "Google strategic priorities" → search_earnings_calls(query="strategic priorities CEO", ticker="GOOGL")
- "Apple AI strategy" → search_earnings_calls(query="AI strategy", ticker="AAPL")
- "Microsoft cloud growth" → search_earnings_calls(query="cloud growth", ticker="MSFT")
- "Amazon AWS revenue" → search_earnings_calls(query="AWS revenue", ticker="AMZN")

Common company name → ticker mappings:
- Google/Alphabet → GOOGL
- Apple → AAPL
- Microsoft → MSFT
- Amazon → AMZN
- NVIDIA → NVDA
- Tesla → TSLA
- Meta/Facebook → META

**TOOL SELECTION:**
You have TWO search tools available:

1. **search_earnings_calls** (FAST): Direct hybrid search
   - Use for: Simple, direct queries with clear intent
   - Examples: search_earnings_calls("AI strategy", ticker="AAPL")
   - Best when: User specifies a specific company

2. **agentic_earnings_search** (OPTIMIZED): LLM-powered query optimization
   - Use for: Complex, vague, or multi-faceted queries across multiple companies
   - Examples: "How are tech companies addressing AI regulation concerns?"
   - Best when: Query spans multiple companies or is very broad

Guidelines:
- ALWAYS extract company names and pass as ticker parameter
- Use specific search queries to find relevant content
- Include speaker attribution (CEO, CFO, analyst name) when quoting
- Reference the quarter and company when discussing earnings content
- Summarize key points rather than dumping full transcript text
- If results mention "strategic priorities", "guidance", "outlook" etc., that IS relevant content
- When comparing companies, search each company separately

Example queries you can handle:
- "What did Apple's CEO say about AI?" → search_earnings_calls("CEO AI", ticker="AAPL")
- "Google strategic priorities" → search_earnings_calls("strategic priorities", ticker="GOOGL")
- "NVIDIA guidance for next quarter" → search_earnings_calls("guidance next quarter", ticker="NVDA")
"""


# =============================================================================
# Vector Search Configuration
# =============================================================================

def _get_vector_search_config() -> Dict[str, str]:
    """Get Vector Search configuration from environment."""
    return {
        'project_id': os.getenv("GOOGLE_CLOUD_PROJECT", "project-4b3d3288-7603-4755-899"),
        'location': os.getenv("VECTOR_SEARCH_LOCATION", "us-central1"),
        'index_endpoint_id': os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_ID", ""),
        'deployed_index_id': os.getenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID", "earnings_hybrid_3072"),
        'embedding_model': os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        'embedding_dimensions': int(os.getenv("EMBEDDING_DIMENSIONALITY", "3072")),
    }


def _load_bm25_encoder():
    """Load BM25 encoder for sparse embeddings.
    
    Loads from GCS on Cloud Run, local file for development.
    """
    # Try GCS first (Cloud Run) or if local file doesn't exist
    if _is_cloud_run():
        data = _load_from_gcs("embeddings/bm25_encoder.json")
        if data:
            return data
        raise ValueError(
            "BM25 encoder not found in GCS. "
            "Run: python hybrid_vector_search_ingestion.py --skip-download"
        )
    
    # Local development path
    bm25_path = Path(__file__).parent.parent.parent.parent / "ingestion" / "hybrid_embeddings" / "bm25_encoder.json"
    
    if not bm25_path.exists():
        # Try GCS as fallback for local development too
        data = _load_from_gcs("embeddings/bm25_encoder.json")
        if data:
            return data
        raise ValueError(
            f"BM25 encoder not found at {bm25_path} or in GCS. "
            "Run: python hybrid_vector_search_ingestion.py --skip-download"
        )
    
    try:
        with open(bm25_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Could not load BM25 encoder: {e}")


def _generate_query_embedding(query: str, config: Dict[str, str]) -> List[float]:
    """Generate dense embedding for a query using Gemini embedding model.
    
    Uses gemini-embedding-001 with 3072 dimensions (pre-normalized for DOT_PRODUCT).
    """
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    
    vertexai.init(project=config['project_id'], location=config['location'])
    
    # Explicitly use gemini-embedding-001 - the only model supporting 3072 dims
    embedding_model = config.get('embedding_model', 'gemini-embedding-001')
    if embedding_model not in ['gemini-embedding-001', 'text-embedding-005']:
        print(f"   [WARN] Embedding model '{embedding_model}' may not support 3072 dims, using gemini-embedding-001")
        embedding_model = 'gemini-embedding-001'
    
    model = TextEmbeddingModel.from_pretrained(embedding_model)
    
    # Use RETRIEVAL_QUERY task type for queries
    text_input = TextEmbeddingInput(query, task_type="RETRIEVAL_QUERY")
    embedding = model.get_embeddings(
        [text_input],
        output_dimensionality=config['embedding_dimensions']
    )
    
    return embedding[0].values


def _generate_sparse_embedding(query: str, bm25_data: Dict) -> Dict[str, List]:
    """Generate BM25 sparse embedding for a query."""
    import re
    import math
    
    if not bm25_data:
        return {"values": [], "dimensions": []}
    
    k1 = bm25_data.get("k1", 1.5)
    b = bm25_data.get("b", 0.75)
    vocab = bm25_data.get("vocab", {})
    idf = bm25_data.get("idf", {})
    avgdl = bm25_data.get("avgdl", 100.0)
    
    # Tokenize query
    text = query.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    doc_len = len(tokens)
    
    # Calculate term frequencies
    tf = {}
    for token in tokens:
        if token in vocab:
            tf[token] = tf.get(token, 0) + 1
    
    # Calculate BM25 scores
    dimensions = []
    values = []
    
    for token, freq in tf.items():
        if token in idf:
            token_idf = idf[token]
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
            score = token_idf * (numerator / denominator)
            
            if score > 0:
                dimensions.append(vocab[token])
                values.append(float(score))
    
    # Sort by dimension
    if dimensions:
        sorted_pairs = sorted(zip(dimensions, values))
        dimensions, values = zip(*sorted_pairs)
    
    return {"values": list(values), "dimensions": list(dimensions)}


def _load_chunk_metadata() -> Dict[str, Dict]:
    """Load chunk metadata from the most recent embeddings file."""
    hybrid_dir = Path(__file__).parent.parent.parent.parent / "ingestion" / "hybrid_embeddings"
    
    if not hybrid_dir.exists():
        return {}
    
    # Find most recent JSONL file
    jsonl_files = list(hybrid_dir.glob("hybrid_embeddings_*.jsonl"))
    if not jsonl_files:
        return {}
    
    latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    
    metadata = {}
    try:
        with open(latest_file, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc.get('id', '')
                # Extract metadata from restricts
                restricts = doc.get('restricts', [])
                meta = {}
                for r in restricts:
                    namespace = r.get('namespace', '')
                    allow_list = r.get('allow_list', [])
                    if allow_list:
                        meta[namespace] = allow_list[0]
                metadata[doc_id] = meta
    except Exception as e:
        print(f"Warning: Could not load chunk metadata: {e}")
    
    return metadata


def _load_chunk_content() -> Dict[str, str]:
    """Load chunk content from GCS (Cloud Run) or local file (development).
    
    This loads the chunk_content.json file which maps chunk IDs to their
    text content and metadata, needed for displaying search results.
    """
    # Try GCS first (Cloud Run) or if local file doesn't exist
    if _is_cloud_run():
        data = _load_from_gcs("embeddings/chunk_content.json")
        if data:
            return data
        raise ValueError(
            "Chunk content not found in GCS. "
            "Run: python run_pipeline.py --skip-download --chunks-only"
        )
    
    # Local development path
    chunk_file = Path(__file__).parent.parent.parent.parent / "ingestion" / "hybrid_embeddings" / "chunk_content.json"
    
    if not chunk_file.exists():
        # Try GCS as fallback for local development too
        data = _load_from_gcs("embeddings/chunk_content.json")
        if data:
            return data
        raise ValueError(
            f"Chunk content not found at {chunk_file} or in GCS. "
            "Run: python run_pipeline.py --skip-download --chunks-only"
        )
    
    try:
        with open(chunk_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Could not load chunk content: {e}")


def search_earnings_calls(
    query: str,
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 5
) -> str:
    """Search earnings call transcripts using Vertex AI Vector Search with hybrid search.
    
    Uses both dense embeddings (semantic search) and BM25 sparse embeddings (keyword match)
    for optimal retrieval quality.
    
    Args:
        query: Search query (e.g., "AI strategy", "guidance for next quarter")
        ticker: Optional ticker filter (e.g., "AAPL", "MSFT")
        company: Optional company name filter (e.g., "Apple", "Microsoft") - will be resolved to ticker
        quarter: Optional quarter filter (e.g., "Q2_2025", "Q3_2025")
        year: Optional year filter as integer (e.g., 2025) - supports range queries
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results with relevant transcript excerpts
    """
    from google.cloud import aiplatform
    
    # Note: Company name filtering is now done directly via restricts (namespace="company")
    # No need to resolve to ticker - both ticker and company restricts are indexed
    
    config = _get_vector_search_config()
    
    # Require Vector Search to be configured - no fallbacks
    if not config['index_endpoint_id']:
        raise ValueError(
            "VECTOR_SEARCH_INDEX_ENDPOINT_ID not configured. "
            "Set this environment variable or run terraform to deploy Vector Search."
        )
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project=config['project_id'], location=config['location'])
        
        # Generate query embeddings
        dense_embedding = _generate_query_embedding(query, config)
        
        # Load BM25 encoder and generate sparse embedding
        bm25_data = _load_bm25_encoder()
        sparse_embedding = _generate_sparse_embedding(query, bm25_data) if bm25_data else None
        
        # Get the index endpoint
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=config['index_endpoint_id']
        )
        
        # Import Namespace and NumericNamespace for filtering
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
            Namespace, NumericNamespace
        )
        
        # Build filter for categorical filtering (using Namespace objects)
        filter_list = []
        if ticker:
            filter_list.append(Namespace(
                name="ticker",
                allow_tokens=[ticker.upper()]
            ))
        
        # Add company name filter (normalized lowercase, without suffixes)
        # This allows filtering by "Apple" or "Microsoft" directly
        if company and not ticker:
            # Normalize company name to match ingestion format
            company_normalized = company.lower().strip().replace(',', '').replace('.', '')
            for suffix in [' inc', ' corp', ' corporation', ' company', ' co', ' ltd', ' llc', ' plc']:
                if company_normalized.endswith(suffix):
                    company_normalized = company_normalized[:-len(suffix)].strip()
            filter_list.append(Namespace(
                name="company",
                allow_tokens=[company_normalized]
            ))
        
        if quarter:
            # Quarter format: Q2_2025, Q3_2025, etc.
            q = quarter.upper()
            if not q.startswith('Q'):
                q = f"Q{q}"
            filter_list.append(Namespace(
                name="quarter", 
                allow_tokens=[q]
            ))
        
        # Build numeric filter for range queries (using NumericNamespace objects)
        numeric_filter_list = []
        if year:
            numeric_filter_list.append(NumericNamespace(
                name="year",
                value_int=int(year),
                op="EQUAL"  # Can also use GREATER_EQUAL, LESS_EQUAL for ranges
            ))
        
        # Determine if user is filtering by a specific company (via ticker OR company name)
        filtering_single_company = bool(ticker or company)
        
        # Convert Namespace objects to REST API format
        restricts = []
        for ns in filter_list:
            restricts.append({
                "namespace": ns.name,
                "allowList": ns.allow_tokens
            })
        
        numeric_restricts = []
        for nns in numeric_filter_list:
            numeric_restricts.append({
                "namespace": nns.name,
                "valueInt": str(nns.value_int),  # API expects string
                "op": nns.op
            })
        
        # Get public endpoint domain from index endpoint
        public_endpoint_domain = index_endpoint.public_endpoint_domain_name
        if not public_endpoint_domain:
            raise ValueError("Index endpoint does not have a public endpoint domain configured")
        
        # Execute search via direct REST API (bypasses SDK v1beta1 $alt bug)
        neighbors = _find_neighbors_rest(
            public_endpoint_domain=public_endpoint_domain,
            project_id=config['project_id'],
            location=config['location'],
            index_endpoint_id=config['index_endpoint_id'],
            deployed_index_id=config['deployed_index_id'],
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            num_neighbors=max_results * 2,  # Get more to filter
            restricts=restricts if restricts else None,
            numeric_restricts=numeric_restricts if numeric_restricts else None,
            per_crowding_attribute_neighbor_count=3 if not filtering_single_company else None,
        )
        
        # Load chunk content for retrieved IDs
        chunk_content = _load_chunk_content()
        
        # Format results
        formatted_results = []
        seen_ids = set()
        
        for match in neighbors:
            doc_id = match['id']
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            if doc_id in chunk_content:
                chunk = chunk_content[doc_id]
                formatted_results.append({
                    'ticker': chunk.get('ticker', 'Unknown'),
                    'quarter': chunk.get('quarter', 'N/A'),
                    'year': chunk.get('year', 'N/A'),
                    'speaker': chunk.get('speaker', 'Unknown Speaker'),
                    'content': chunk.get('content', '')[:800],  # Truncate for readability
                    'company': chunk.get('company', ''),
                    'distance': match['distance'],
                })
            
            if len(formatted_results) >= max_results:
                break
        
        if not formatted_results:
            return f"No earnings call content found for query: '{query}'" + (
                f" (filtered by ticker={ticker}, quarter={quarter})" if ticker or quarter else ""
            )
        
        # Format output with citation-ready format
        output = f"**Earnings Call Search Results for: '{query}'**\n\n"
        for i, r in enumerate(formatted_results, 1):
            output += f"**{i}. {r['ticker']} {r['quarter']} {r['year']}** - {r['speaker']}\n"
            output += f"{r['content']}\n\n"
        
        return output
        
    except Exception as e:
        # No fallback - surface errors for investigation
        raise RuntimeError(
            f"Vector Search query failed: {e}. "
            f"Check Vector Search configuration and index status."
        ) from e


# NOTE: _local_search_fallback removed - no fallbacks allowed.
# If Vector Search fails, we raise an exception for investigation.


def agentic_earnings_search(
    query: str,
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 5,
    quality_threshold: float = 0.7
) -> str:
    """Optimized earnings call search with LLM-based query rewriting and quality evaluation.
    
    This function uses an agentic approach to optimize search quality:
    1. Rewrites the query using LLM to expand terms and add context
    2. Executes the search with the optimized query
    3. Evaluates result quality using LLM-as-Judge
    4. If quality is below threshold, iteratively refines the query
    
    Use this for:
    - Complex or vague queries that might benefit from expansion
    - Queries where you want higher retrieval quality
    - When initial search results seem incomplete
    
    For simple, direct queries, use search_earnings_calls instead for speed.
    
    Args:
        query: Search query for earnings calls (will be optimized)
        ticker: Optional ticker filter (e.g., "AAPL", "MSFT")
        company: Optional company name filter (e.g., "Apple", "Microsoft")
        quarter: Optional quarter filter (e.g., "Q2_2025", "Q3_2025")
        year: Optional year filter as integer (e.g., 2025)
        max_results: Maximum number of results to return (default: 5)
        quality_threshold: Score threshold to stop optimization (0-1, default: 0.7)
        
    Returns:
        Formatted search results with best quality found, including optimization metadata
    """
    from ..utils.agentic_rag import agentic_search, AgenticRAGConfig
    
    config = AgenticRAGConfig(
        max_iterations=3,
        quality_threshold=quality_threshold,
        enable_logging=False  # Disable console logging for production
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
    
    # Append optimization metadata to help the agent understand what happened
    metadata = (
        f"\n\n---\n"
        f"*Query Optimization Summary:*\n"
        f"- Original: \"{result.original_query}\"\n"
        f"- Optimized: \"{result.final_query}\"\n"
        f"- Quality Score: {result.final_quality_score:.2f}/1.00\n"
        f"- Iterations: {result.iterations_used}\n"
    )
    
    return result.results + metadata


def get_company_earnings_summary(ticker: str) -> str:
    """Get a summary of available earnings call data for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        
    Returns:
        Summary of available earnings call transcripts for the company
        
    Raises:
        ValueError: If no chunk content is available
    """
    chunk_content = _load_chunk_content()
    
    if not chunk_content:
        raise ValueError(
            "No earnings call chunk content available. "
            "Run the ingestion pipeline (hybrid_vector_search_ingestion.py) first."
        )
    
    ticker_upper = ticker.upper()
    
    # Find all chunks for this ticker
    quarters_found = set()
    speakers_found = set()
    chunk_count = 0
    company_name = ""
    
    for chunk_id, chunk in chunk_content.items():
        if chunk.get('ticker', '').upper() == ticker_upper:
            quarter = chunk.get('quarter', '')
            year = chunk.get('year', '')
            speaker = chunk.get('speaker', '')
            
            if quarter and year:
                quarters_found.add(f"{quarter} {year}")
            if speaker and speaker != "Introduction":
                speakers_found.add(speaker)
            if not company_name:
                company_name = chunk.get('company', '')
            chunk_count += 1
    
    if not quarters_found:
        return f"No earnings call transcripts found for {ticker_upper} in the data store."
    
    output = f"**{ticker_upper} Earnings Call Data Available**"
    if company_name:
        output += f" ({company_name})"
    output += "\n\n"
    output += f"**Quarters:** {', '.join(sorted(quarters_found))}\n"
    output += f"**Total Chunks:** {chunk_count}\n"
    output += f"**Speakers:** {', '.join(list(speakers_found)[:5])}" 
    if len(speakers_found) > 5:
        output += f" and {len(speakers_found) - 5} more"
    output += "\n\nUse `search_earnings_calls` to find specific content."
    
    return output


def create_vector_store_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Vector Store sub-agent with Vertex AI Vector Search tools.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent uses Vertex AI Vector Search for earnings call RAG search.
    
    Args:
        model: Gemini model to use (defaults to config MODEL_COMPLEX for analysis)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with earnings call search tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    
    # Use complex model by default for analysis-heavy tasks
    agent_model = model or Config.MODEL_COMPLEX
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # Create the sub-agent with search tools
    vector_store_agent = LlmAgent(
        model=agent_model,
        name="vector_store_agent",
        description="Searches earnings call transcripts for S&P 500 companies. Use for finding management commentary, analyst Q&A, and company guidance from Q2/Q3 2025 earnings calls.",
        instruction=VECTOR_STORE_INSTRUCTION,
        tools=[
            search_earnings_calls,
            agentic_earnings_search,
            get_company_earnings_summary,
        ],
        before_model_callback=dynamic_callback,
    )
    
    return vector_store_agent


if __name__ == "__main__":
    config = _get_vector_search_config()
    
    print("\n=== Vector Store Agent ===")
    print("This agent searches S&P 500 earnings call transcripts using Vertex AI Vector Search.")
    print("\nTo create the agent:")
    print("  from finance_agent.workflow_1.agents.vector_store_agent import create_vector_store_agent")
    print("  vector_agent = create_vector_store_agent()")
    print("\nAvailable Tools:")
    print("  * search_earnings_calls - Direct hybrid search (fast)")
    print("  * agentic_earnings_search - LLM-optimized search with query rewriting (optimized)")
    print("  * get_company_earnings_summary - Get available data for a ticker")
    print("\nVector Search Configuration:")
    print(f"  Project: {config['project_id']}")
    print(f"  Location: {config['location']}")
    print(f"  Index Endpoint: {config['index_endpoint_id'] or '[NOT SET - WILL FAIL]'}")
    print(f"  Deployed Index: {config['deployed_index_id']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  Embedding Dimensions: {config['embedding_dimensions']}")
    
    # Test company summary (uses local chunk content)
    print("\n--- Testing Company Summary ---")
    result = get_company_earnings_summary("AAPL")
    print(result)

# Terraform Infrastructure

Infrastructure as Code for the Real-Time Finance AI POC using **Vertex AI Vector Search Native Hybrid Search**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Vertex AI Vector Search                              │
│                      (Native Hybrid Search - GA Dec 2024)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐   │
│   │    Dense      │         │    Sparse     │         │   HybridQuery │   │
│   │  Embeddings   │         │  Embeddings   │         │     API       │   │
│   │  (Gemini)     │         │   (BM25)      │         │               │   │
│   └───────┬───────┘         └───────┬───────┘         └───────┬───────┘   │
│           │                         │                         │           │
│           │    ┌────────────────────┴─────────────────────────┘           │
│           │    │                                                          │
│           ▼    ▼                                                          │
│   ┌─────────────────────┐                                                 │
│   │  Reciprocal Rank    │  ← Alpha controls weighting:                    │
│   │  Fusion (RRF)       │    1.0 = Dense only (semantic)                  │
│   └──────────┬──────────┘    0.0 = Sparse only (keyword)                  │
│              │               0.5 = Balanced hybrid                        │
│              ▼                                                            │
│   ┌─────────────────────┐                                                 │
│   │   Merged Results    │                                                 │
│   │   with Citations    │                                                 │
│   └─────────────────────┘                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Hybrid Search?

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Dense (Semantic)** | Understands meaning, synonyms, context | May miss exact keywords, entity names |
| **Sparse (BM25)** | Exact keyword matching, entities, acronyms | No semantic understanding |
| **Hybrid (Both)** | Best of both worlds | Slightly more complex |

### Data Flow

1. **Query** → Generate dense embedding (Gemini) + sparse embedding (BM25)
2. **HybridQuery** → Search both dense and sparse indexes simultaneously
3. **RRF Fusion** → Merge ranked results using Reciprocal Rank Fusion
4. **Results** → Return top-k documents with scores and citations

## Resources Created

| Resource | Purpose | File |
|----------|---------|------|
| **Vertex AI Vector Search Index** | Stores dense + sparse embeddings | `vector_search.tf` |
| **Vertex AI Index Endpoint** | Serves HybridQuery API | `vector_search.tf` |
| **VPC Network** | Private network for Vector Search | `vector_search.tf` |
| **Cloud Storage Bucket** | Embeddings JSONL storage | `storage.tf` |
| **Cloud SQL (PostgreSQL)** | Agent memory persistence | `cloudsql.tf` |
| **Service Account** | Application identity | `iam.tf` |

## Prerequisites

1. **GCP Project** with billing enabled
2. **Terraform** >= 1.0
3. **Python** 3.10+ with `rank_bm25` library for sparse embeddings

## Setup

### 1. Initialize Terraform
```bash
cd terraform
terraform init
```

### 2. Plan and Apply
```bash
# Review what will be created
terraform plan

# Apply (creates all resources)
terraform apply
```

### 3. Generate and Upload Embeddings

After infrastructure is created, generate embeddings and upload to GCS:

```python
# Generate embeddings (see finance_agent/workflow_1/embeddings/)
from embeddings.hybrid_generator import HybridEmbeddingGenerator

generator = HybridEmbeddingGenerator()

# For each document chunk:
result = generator.generate_hybrid_embedding(
    text="Apple reported strong Q3 earnings...",
    doc_id="AAPL_Q3_2025_1",
    metadata={"ticker": "AAPL", "quarter": "Q3", "year": 2025}
)
```

```bash
# Upload to GCS
gsutil cp embeddings.jsonl gs://PROJECT-finance-data-bucket/embeddings/
```

### JSONL Format for Hybrid Search

Each document requires both dense and sparse embeddings:

```json
{
  "id": "AAPL_Q3_2025_1",
  "embedding": [0.1, 0.2, ...],
  "sparse_embedding": {
    "values": [0.5, 0.3, 0.8, ...],
    "dimensions": [42, 156, 891, ...]
  },
  "restricts": [
    {"namespace": "ticker", "allow_list": ["AAPL"]},
    {"namespace": "quarter", "allow_list": ["Q3"]}
  ]
}
```

| Field | Description |
|-------|-------------|
| `id` | Unique document identifier |
| `embedding` | Dense vector (3072-dim for gemini-embedding-001) |
| `sparse_embedding.values` | Non-zero BM25 scores |
| `sparse_embedding.dimensions` | Indices of non-zero values |
| `restricts` | Metadata filters (ticker, quarter, etc.) |

## Querying with HybridQuery API

```python
from google.cloud import aiplatform

# Initialize client
client = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name="projects/PROJECT/locations/REGION/indexEndpoints/ENDPOINT_ID"
)

# Hybrid search
response = client.find_neighbors(
    deployed_index_id="earnings-deployed-index",
    queries=[{
        "datapoint": {
            "datapoint_id": "query",
            "feature_vector": dense_embedding,  # From Gemini
            "sparse_embedding": {
                "values": sparse_values,        # From BM25
                "dimensions": sparse_dims
            }
        },
        "neighbor_count": 10,
        "rrf_ranking_alpha": 0.5,  # Hybrid weighting
    }],
    return_full_datapoint=True
)
```

### Alpha Parameter Guide

| Alpha Value | Search Behavior | Best For |
|-------------|-----------------|----------|
| `1.0` | Dense only (semantic) | "What's Apple's AI strategy?" |
| `0.0` | Sparse only (BM25) | "AAPL Q3 2025 revenue" |
| `0.5` | Balanced hybrid | Most queries (recommended) |
| `0.7` | Semantic-heavy hybrid | Conceptual questions |
| `0.3` | Keyword-heavy hybrid | Entity-focused queries |

## Outputs

After `terraform apply`, these values are available:

```bash
# Vector Search
terraform output vector_index_id
terraform output vector_endpoint_id
terraform output deployed_index_id

# Storage
terraform output gcs_bucket_name
terraform output embeddings_gcs_uri

# Database
terraform output cloudsql_connection_name
terraform output session_db_url

# Service Account
terraform output service_account_email
```

## Cost Estimates

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| Vector Search (small index, 1 replica) | ~$50-100 |
| Cloud SQL (db-f1-micro) | ~$10 |
| Cloud Storage | ~$1-5 |
| Embedding API calls | ~$5-20 |
| **Total** | **~$65-135/month** |

**Note:** Hybrid search with Vertex AI Vector Search is more cost-effective than running a separate Elastic Cloud instance (~$95/month saved).

## Cleanup

```bash
# Destroy all resources (WARNING: irreversible)
terraform destroy
```

## File Structure

```
terraform/
├── provider.tf          # GCP provider configuration
├── variables.tf         # Input variables (embedding dims, hybrid config)
├── terraform.tfvars     # Variable values (gitignored)
├── apis.tf              # Enable required GCP APIs
├── vector_search.tf     # Vertex AI Vector Search (hybrid-enabled)
├── storage.tf           # GCS bucket for embeddings
├── cloudsql.tf          # PostgreSQL for agent memories
├── iam.tf               # Service account and permissions
└── README.md            # This file
```

## References

- [Vertex AI Vector Search Hybrid Search (GA Dec 2024)](https://cloud.google.com/vertex-ai/docs/vector-search/hybrid-search)
- [HybridQuery API Documentation](https://cloud.google.com/vertex-ai/docs/vector-search/query-index#hybrid-query)
- [rank_bm25 Python Library](https://pypi.org/project/rank-bm25/)

# =============================================================================
# Vertex AI Vector Search Infrastructure - Hybrid Search
# =============================================================================
# Native Hybrid Search Architecture (GA Dec 2024):
#   - Dense embeddings: Gemini gemini-embedding-001 (3072-dim, pre-normalized) for semantic search
#   - Sparse embeddings: BM25 vectors for keyword matching
#   - HybridQuery API: Combines both with Reciprocal Rank Fusion (RRF)
#
# Data Flow:
#   Query → Generate Dense + Sparse Embeddings → HybridQuery → RRF Fusion → Results
#
# Key Parameters:
#   - alpha: Weighting between dense (1.0) and sparse (0.0) search
#   - rrf_ranking_alpha: Controls RRF fusion behavior
#
# NOTE: Using PUBLIC endpoint to avoid VPC peering permission issues.
#       For production, consider switching to private endpoint with proper IAM.
# =============================================================================

# -----------------------------------------------------------------------------
# Vertex AI Vector Search Index
# -----------------------------------------------------------------------------
# The index stores embeddings and uses ScaNN-based ANN for fast retrieval.
# Embeddings are loaded from GCS bucket in JSONL format.

resource "google_vertex_ai_index" "earnings_embeddings" {
  region       = var.region
  project      = var.project_id
  display_name = var.vector_index_display_name
  description  = "Earnings call transcript embeddings for hybrid search (dense + sparse/BM25)"

  # BATCH_UPDATE: Update index by uploading new embeddings to GCS
  # STREAM_UPDATE: Real-time updates via API (more expensive)
  index_update_method = "BATCH_UPDATE"

  labels = {
    environment = "production"
    purpose     = "earnings-rag"
    search_type = "hybrid"
  }

  metadata {
    # GCS path where embeddings JSONL files are stored
    # Files must include both 'embedding' (dense) and 'sparse_embedding' (BM25) fields
    contents_delta_uri = "gs://${google_storage_bucket.finance_data.name}/embeddings/"

    config {
      # Dense embedding dimensions (3072 for gemini-embedding-001, pre-normalized for DOT_PRODUCT)
      dimensions = var.embedding_dimensions

      # ANN configuration
      approximate_neighbors_count = var.approximate_neighbors_count
      shard_size                  = var.shard_size

      # Distance measure - DOT_PRODUCT for normalized embeddings
      distance_measure_type = "DOT_PRODUCT_DISTANCE"

      # Tree-AH algorithm configuration (Google's ScaNN variant)
      algorithm_config {
        tree_ah_config {
          # More leaf nodes = higher recall, slower indexing
          leaf_node_embedding_count = 500

          # Higher % = more accurate, slower queries
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  depends_on = [
    google_project_service.aiplatform,
    google_storage_bucket.finance_data
  ]
}

# -----------------------------------------------------------------------------
# Vertex AI Vector Search Index Endpoint (PUBLIC)
# -----------------------------------------------------------------------------
# Using public endpoint to avoid VPC peering permission requirements.
# For production with sensitive data, use private endpoint with proper IAM.

resource "google_vertex_ai_index_endpoint" "earnings_endpoint" {
  display_name = "earnings-search-endpoint"
  description  = "Endpoint for earnings call hybrid search (HybridQuery API)"
  region       = var.region
  project      = var.project_id

  labels = {
    environment = "production"
  }

  # PUBLIC endpoint - no VPC peering required
  # Omitting 'network' parameter creates a public endpoint
  public_endpoint_enabled = true

  depends_on = [
    google_project_service.aiplatform
  ]
}

# -----------------------------------------------------------------------------
# Deploy Index to Endpoint
# -----------------------------------------------------------------------------
# This deploys the index to the endpoint, making it queryable.
# Note: This can take 20-30 minutes to complete.

resource "google_vertex_ai_index_endpoint_deployed_index" "earnings_deployed" {
  deployed_index_id = "earnings_hybrid_3072"
  display_name      = "Earnings Embeddings Deployed"

  index          = google_vertex_ai_index.earnings_embeddings.id
  index_endpoint = google_vertex_ai_index_endpoint.earnings_endpoint.id

  # Use automatic scaling (cheaper for low traffic)
  automatic_resources {
    min_replica_count = 1
    max_replica_count = 2
  }

  depends_on = [
    google_vertex_ai_index.earnings_embeddings,
    google_vertex_ai_index_endpoint.earnings_endpoint
  ]
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "vector_index_id" {
  description = "Vertex AI Vector Search Index ID"
  value       = google_vertex_ai_index.earnings_embeddings.id
}

output "vector_index_name" {
  description = "Vertex AI Vector Search Index resource name"
  value       = google_vertex_ai_index.earnings_embeddings.name
}

output "vector_endpoint_id" {
  description = "Vertex AI Vector Search Endpoint ID"
  value       = google_vertex_ai_index_endpoint.earnings_endpoint.id
}

output "vector_endpoint_name" {
  description = "Vertex AI Vector Search Endpoint resource name"
  value       = google_vertex_ai_index_endpoint.earnings_endpoint.name
}

output "deployed_index_id" {
  description = "Deployed Index ID for queries"
  value       = google_vertex_ai_index_endpoint_deployed_index.earnings_deployed.deployed_index_id
}

output "public_endpoint_domain" {
  description = "Public endpoint domain for Vector Search queries"
  value       = google_vertex_ai_index_endpoint.earnings_endpoint.public_endpoint_domain_name
}

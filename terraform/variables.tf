variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "service_account_name" {
  description = "Name for the service account"
  type        = string
  default     = "finance-chatbot-sa"
}

variable "db_password" {
  description = "Password for the Cloud SQL database user"
  type        = string
  sensitive   = true
}

# =============================================================================
# Vertex AI Vector Search Configuration - Hybrid Search
# =============================================================================

variable "vector_index_display_name" {
  description = "Display name for the Vector Search index"
  type        = string
  default     = "earnings-call-embeddings-hybrid"
}

variable "embedding_dimensions" {
  description = "Dimensionality of dense embeddings (3072 for gemini-embedding-001, pre-normalized)"
  type        = number
  default     = 3072
}

variable "approximate_neighbors_count" {
  description = "Number of neighbors to find in ANN search"
  type        = number
  default     = 150
}

variable "shard_size" {
  description = "Shard size for the index (SHARD_SIZE_SMALL, SHARD_SIZE_MEDIUM, SHARD_SIZE_LARGE)"
  type        = string
  default     = "SHARD_SIZE_SMALL"
}

# =============================================================================
# Hybrid Search Configuration
# =============================================================================

variable "sparse_embedding_dimensions" {
  description = "Dimensionality of sparse embeddings (BM25 vocabulary size)"
  type        = number
  default     = 50000 # Typical BM25 vocabulary size
}

variable "default_hybrid_alpha" {
  description = "Default alpha for hybrid search (1.0=dense only, 0.0=sparse only, 0.5=balanced)"
  type        = number
  default     = 0.5
}

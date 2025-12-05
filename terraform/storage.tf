# =============================================================================
# Cloud Storage for Embeddings and Data
# =============================================================================

# GCS bucket for embeddings JSONL and data staging
resource "google_storage_bucket" "finance_data" {
  name          = "${var.project_id}-finance-data-bucket"
  location      = var.region
  project       = var.project_id
  force_destroy = false # Prevent accidental deletion

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  # Lifecycle rules for different content types
  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["staging/"]
      with_state     = "ANY"
    }
    action {
      type = "Delete"
    }
  }

  # Keep embeddings longer (they're expensive to regenerate)
  lifecycle_rule {
    condition {
      age                = 365
      matches_prefix     = ["embeddings/"]
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }
}

# -----------------------------------------------------------------------------
# Folder structure (created as empty objects)
# -----------------------------------------------------------------------------

# Embeddings folder - Vector Search index source
resource "google_storage_bucket_object" "embeddings_folder" {
  name    = "embeddings/"
  content = " "
  bucket  = google_storage_bucket.finance_data.name
}

# Staging folder - temporary uploads
resource "google_storage_bucket_object" "staging_folder" {
  name    = "staging/"
  content = " "
  bucket  = google_storage_bucket.finance_data.name
}

# Raw data folder - original transcript files
resource "google_storage_bucket_object" "raw_data_folder" {
  name    = "raw-data/"
  content = " "
  bucket  = google_storage_bucket.finance_data.name
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "gcs_bucket_name" {
  description = "GCS bucket name for data storage"
  value       = google_storage_bucket.finance_data.name
}

output "embeddings_gcs_uri" {
  description = "GCS URI for embeddings (Vector Search index source)"
  value       = "gs://${google_storage_bucket.finance_data.name}/embeddings/"
}


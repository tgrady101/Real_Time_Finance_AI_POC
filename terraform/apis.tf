# =============================================================================
# Enable Required Google Cloud APIs
# =============================================================================

# Vertex AI API (for Vector Search and Embeddings)
resource "google_project_service" "aiplatform" {
  project = var.project_id
  service = "aiplatform.googleapis.com"

  disable_on_destroy = false
}

# Compute Engine API (required for VPC/networking)
resource "google_project_service" "compute" {
  project = var.project_id
  service = "compute.googleapis.com"

  disable_on_destroy = false
}

# Service Networking API (for private VPC peering)
resource "google_project_service" "servicenetworking" {
  project = var.project_id
  service = "servicenetworking.googleapis.com"

  depends_on         = [google_project_service.compute]
  disable_on_destroy = false
}

# Cloud SQL Admin API
resource "google_project_service" "sqladmin" {
  project = var.project_id
  service = "sqladmin.googleapis.com"

  disable_on_destroy = false
}

# Cloud Run API
resource "google_project_service" "run" {
  project = var.project_id
  service = "run.googleapis.com"

  disable_on_destroy = false
}

# Secret Manager API (for storing Elastic credentials)
resource "google_project_service" "secretmanager" {
  project = var.project_id
  service = "secretmanager.googleapis.com"

  disable_on_destroy = false
}

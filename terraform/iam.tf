# =============================================================================
# IAM Configuration
# =============================================================================

# Service account for the application
resource "google_service_account" "finance_chatbot" {
  account_id   = var.service_account_name
  display_name = "Finance Chatbot Service Account"
  description  = "Service account for Real-Time Finance AI POC chatbot"
  project      = var.project_id
}

# -----------------------------------------------------------------------------
# Storage Permissions
# -----------------------------------------------------------------------------

# Storage Object Admin (for GCS embeddings read/write)
resource "google_storage_bucket_iam_member" "storage_admin" {
  bucket = google_storage_bucket.finance_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.finance_chatbot.email}"
}

# -----------------------------------------------------------------------------
# Vertex AI Permissions
# -----------------------------------------------------------------------------
# NOTE: These IAM bindings require project-level IAM Admin permissions.
# If Terraform cannot apply them, grant manually via GCP Console or gcloud:
#
#   gcloud projects add-iam-policy-binding PROJECT_ID \
#     --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
#     --role="roles/aiplatform.user"
#
# Required roles for the finance-chatbot-sa:
#   - roles/aiplatform.user (Vector Search queries, embeddings API)
#   - roles/cloudsql.client (Cloud SQL connections)
# -----------------------------------------------------------------------------

# Vertex AI User - grant via console if this fails
# resource "google_project_iam_member" "vertex_ai_user" {
#   project = var.project_id
#   role    = "roles/aiplatform.user"
#   member  = "serviceAccount:${google_service_account.finance_chatbot.email}"
# }

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "service_account_email" {
  description = "Service account email for authentication"
  value       = google_service_account.finance_chatbot.email
}


# Service account for the application
resource "google_service_account" "finance_chatbot" {
  account_id   = var.service_account_name
  display_name = "Finance Chatbot Service Account"
  description  = "Service account for Real-Time Finance AI POC chatbot"
}

# IAM binding: Discovery Engine Editor (for Data Store operations)
resource "google_project_iam_member" "discovery_engine_editor" {
  project = var.project_id
  role    = "roles/discoveryengine.editor"
  member  = "serviceAccount:${google_service_account.finance_chatbot.email}"
}

# IAM binding: Storage Object Admin (for GCS operations)
resource "google_storage_bucket_iam_member" "storage_admin" {
  bucket = google_storage_bucket.finance_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.finance_chatbot.email}"
}

# IAM binding: Vertex AI User (for Discovery Engine RAG and Model Garden)
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.finance_chatbot.email}"
}

# Output service account email
output "service_account_email" {
  description = "Service account email for authentication"
  value       = google_service_account.finance_chatbot.email
}

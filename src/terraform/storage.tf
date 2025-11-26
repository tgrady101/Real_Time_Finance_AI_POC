# GCS bucket for data staging and file uploads
resource "google_storage_bucket" "finance_data" {
  name          = "${var.project_id}-finance-data-bucket"
  location      = var.region
  force_destroy = false  # Prevent accidental deletion

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90  # Delete files older than 90 days
    }
    action {
      type = "Delete"
    }
  }
}

# Output bucket name
output "gcs_bucket_name" {
  description = "GCS bucket name for data storage"
  value       = google_storage_bucket.finance_data.name
}

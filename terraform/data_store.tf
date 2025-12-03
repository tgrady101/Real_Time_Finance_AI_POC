# Vertex AI Data Store for earnings calls and SEC filings
resource "google_discovery_engine_data_store" "financial_filings" {
  location                    = var.data_store_location
  data_store_id              = var.data_store_id
  display_name               = "Financial Filings Data Store"
  industry_vertical          = "GENERIC"
  content_config             = "CONTENT_REQUIRED"  # Required for RAG
  solution_types             = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  # Note: Ensure Discovery Engine API is enabled in your project
}

# Output the Data Store resource name for use in application
output "data_store_name" {
  description = "Full resource name of the Data Store"
  value       = google_discovery_engine_data_store.financial_filings.name
}

output "data_store_id_output" {
  description = "Data Store ID for application configuration"
  value       = google_discovery_engine_data_store.financial_filings.data_store_id
}

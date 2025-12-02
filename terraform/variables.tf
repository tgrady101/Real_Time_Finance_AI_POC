variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "data_store_location" {
  description = "Location for Vertex AI Data Store"
  type        = string
  default     = "global"
}

variable "data_store_id" {
  description = "ID for the Vertex AI Data Store"
  type        = string
  default     = "financial-filings-datastore"
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

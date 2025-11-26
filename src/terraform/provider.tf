terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  
  # Impersonate service account (no key file needed)
  impersonate_service_account = "terraform-admin@project-4b3d3288-7603-4755-899.iam.gserviceaccount.com"
}

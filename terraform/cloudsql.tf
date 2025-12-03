# Cloud SQL PostgreSQL instance for persistent memory storage
# This provides persistent memory storage for cross-session agent recall
# 
# Note: Sessions are ephemeral (InMemorySessionService) - Arize handles logging.
# Only agent memories are persisted here for the load_memory tool.

resource "google_sql_database_instance" "sessions" {
  name             = "finance-agent-sessions"
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  settings {
    # db-f1-micro is the smallest/cheapest tier (~$10/month)
    tier = "db-f1-micro"

    # Disk settings
    disk_size       = 10 # GB, minimum
    disk_type       = "PD_SSD"
    disk_autoresize = false

    # IP configuration for Cloud Run connection
    ip_configuration {
      ipv4_enabled = true
      # For production, consider using private IP with VPC connector
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0" # For development - restrict in production
      }
    }

    # Backup configuration (optional but recommended)
    backup_configuration {
      enabled            = true
      start_time         = "03:00" # 3 AM UTC
      binary_log_enabled = false   # Not needed for PostgreSQL
    }

    # Maintenance window
    maintenance_window {
      day  = 7 # Sunday
      hour = 3 # 3 AM
    }
  }

  # Prevent accidental deletion
  deletion_protection = false # Set to true for production
}

# Database for memories (table auto-created by PostgresMemoryService)
resource "google_sql_database" "sessions_db" {
  name     = "sessions" # Database name kept for compatibility
  instance = google_sql_database_instance.sessions.name
  project  = var.project_id
}

# Database user
resource "google_sql_user" "sessions_user" {
  name     = "finance_agent"
  instance = google_sql_database_instance.sessions.name
  password = var.db_password
  project  = var.project_id
}

# Output the connection string components
output "cloudsql_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.sessions.name
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name for Cloud Run"
  value       = google_sql_database_instance.sessions.connection_name
}

output "cloudsql_public_ip" {
  description = "Cloud SQL public IP address"
  value       = google_sql_database_instance.sessions.public_ip_address
}

output "session_db_url" {
  description = "Database URL for SESSION_DB_URL env var"
  value       = "postgresql://finance_agent:${var.db_password}@${google_sql_database_instance.sessions.public_ip_address}:5432/sessions"
  sensitive   = true
}

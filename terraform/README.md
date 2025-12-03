# Terraform Infrastructure for Real-Time Finance AI POC

This directory contains Terraform configurations to provision all required GCP infrastructure.

## Resources Provisioned

- **Vertex AI Data Store** - For RAG over earnings calls and SEC filings
- **GCS Bucket** - For data staging and file uploads
- **Service Account** - With least-privilege IAM permissions
- **IAM Bindings** - Discovery Engine Editor, Storage Object Admin, Vertex AI User

## Prerequisites

1. **GCP Project** with billing enabled
2. **APIs Enabled**:
   ```bash
   gcloud services enable discoveryengine.googleapis.com
   gcloud services enable storage.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable iam.googleapis.com
   ```

3. **Terraform State Bucket** (create manually):
   ```bash
   gsutil mb gs://YOUR_PROJECT_ID-terraform-state
   ```

## Setup Instructions

### 1. Configure Variables

```bash
# Copy example file
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

### 2. Update Backend Configuration

Edit `backend.tf` and replace `REPLACE_WITH_YOUR_STATE_BUCKET` with your actual state bucket name.

### 3. Initialize Terraform

```bash
terraform init
```

### 4. Review Plan

```bash
terraform plan
```

### 5. Apply Configuration

```bash
terraform apply
```

## Outputs

After successful apply, Terraform will output:

- `data_store_id_output` - Use this in your `.env` file
- `gcs_bucket_name` - GCS bucket for data uploads
- `service_account_email` - Service account for authentication

## Important Notes

- **State Storage**: Terraform state is stored in GCS for team collaboration
- **Permissions**: The service account has minimal required permissions
- **Cleanup**: Run `terraform destroy` to tear down all resources (be careful!)

## Updating Infrastructure

```bash
# After making changes to .tf files
terraform plan
terraform apply
```

## Troubleshooting

**API not enabled errors**: Run the `gcloud services enable` commands above

**Permission denied**: Ensure you have Owner or Editor role on the GCP project

**State bucket errors**: Verify the bucket exists and is accessible

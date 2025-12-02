"""
Deploy Finance Agent to Google Cloud Run (FastAPI)

This script deploys the Finance Agent FastAPI application to Google Cloud Run
using gcloud CLI with source-based deployment.

Prerequisites:
1. GCP Project with Cloud Run API enabled
2. gcloud CLI authenticated
3. Docker installed (for local testing)
4. Required env vars in .env:
   - GOOGLE_CLOUD_PROJECT
   - GOOGLE_CLOUD_LOCATION (default: us-central1)
   - CLOUD_RUN_SERVICE_NAME (default: finance-agent)

Usage:
    python deploy_cloud_run.py              # Show deployment options
    python deploy_cloud_run.py --deploy     # Deploy to Cloud Run
    python deploy_cloud_run.py --local      # Run locally with uvicorn
    python deploy_cloud_run.py --docker     # Build and run with Docker locally
    python deploy_cloud_run.py --status     # Check deployment status
    python deploy_cloud_run.py --logs       # View recent logs
    python deploy_cloud_run.py --delete     # Delete the service
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
SERVICE_NAME = os.getenv("CLOUD_RUN_SERVICE_NAME", "finance-agent")
AGENT_PATH = project_root / "finance_agent"

# Environment variables to pass to Cloud Run
ENV_VARS = [
    f"GOOGLE_CLOUD_PROJECT={PROJECT_ID}",
    f"GOOGLE_CLOUD_LOCATION={os.getenv('GOOGLE_CLOUD_LOCATION', 'global')}",
    "GOOGLE_GENAI_USE_VERTEXAI=TRUE",
    "MEMORY_STORAGE=auto",  # Auto-detect: Postgres on Cloud Run, InMemory locally
]

# Add SESSION_DB_URL for PostgresMemoryService if configured
SESSION_DB_URL = os.getenv("SESSION_DB_URL")
if SESSION_DB_URL:
    ENV_VARS.append(f"SESSION_DB_URL={SESSION_DB_URL}")

# Add Arize env vars if configured
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "finance-chatbot")
if ARIZE_API_KEY and ARIZE_SPACE_ID:
    ENV_VARS.extend([
        f"ARIZE_API_KEY={ARIZE_API_KEY}",
        f"ARIZE_SPACE_ID={ARIZE_SPACE_ID}",
        f"ARIZE_PROJECT_NAME={ARIZE_PROJECT_NAME}",
        "ARIZE_ENABLED=true",
    ])


def validate_config() -> bool:
    """Validate required configuration."""
    errors = []
    
    if not PROJECT_ID:
        errors.append("GOOGLE_CLOUD_PROJECT not set in .env")
    
    if not AGENT_PATH.exists():
        errors.append(f"Agent directory not found: {AGENT_PATH}")
    
    if not (AGENT_PATH / "Dockerfile").exists():
        errors.append(f"Dockerfile not found in {AGENT_PATH}")
    
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    print("\n✅ Configuration:")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Location: {LOCATION}")
    print(f"   Service: {SERVICE_NAME}")
    print(f"   Source: {AGENT_PATH}")
    return True


def check_service_exists() -> bool:
    """Check if Cloud Run service exists."""
    try:
        result = subprocess.run(
            ["gcloud", "run", "services", "describe", SERVICE_NAME,
             f"--region={LOCATION}", f"--project={PROJECT_ID}", "--format=value(name)"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_service_url() -> str | None:
    """Get the URL of the deployed service."""
    try:
        result = subprocess.run(
            ["gcloud", "run", "services", "describe", SERVICE_NAME,
             f"--region={LOCATION}", f"--project={PROJECT_ID}", "--format=value(status.url)"],
            capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def show_instructions():
    """Show deployment options and instructions."""
    print("\n" + "="*60)
    print("FINANCE AGENT - DEPLOYMENT OPTIONS")
    print("="*60)
    
    print("\n📋 Available Commands:\n")
    print("  python deploy_cloud_run.py --deploy    Deploy to Cloud Run")
    print("  python deploy_cloud_run.py --local     Run locally (uvicorn)")
    print("  python deploy_cloud_run.py --docker    Run locally (Docker)")
    print("  python deploy_cloud_run.py --status    Check deployment status")
    print("  python deploy_cloud_run.py --logs      View service logs")
    print("  python deploy_cloud_run.py --delete    Delete the service")
    
    print("\n" + "-"*60)
    print("Manual gcloud command:")
    print("-"*60)
    print(f"""
gcloud run deploy {SERVICE_NAME} \\
    --source {AGENT_PATH} \\
    --region {LOCATION} \\
    --project {PROJECT_ID} \\
    --allow-unauthenticated \\
    --set-env-vars="{','.join(ENV_VARS)}"
""")
    
    print("-"*60)
    print("API Endpoints (after deployment):")
    print("-"*60)
    print("""
    GET  /           - API info
    GET  /health     - Health check
    GET  /docs       - OpenAPI documentation
    GET  /agent/info - Agent configuration
    POST /chat       - Chat with agent (full response)
    POST /chat/stream - Chat with streaming (SSE)
""")


def deploy():
    """Deploy to Google Cloud Run."""
    print("\n" + "="*60)
    print("DEPLOYING TO GOOGLE CLOUD RUN")
    print("="*60)
    
    existing = check_service_exists()
    if existing:
        print(f"\n📝 Updating existing service: {SERVICE_NAME}")
    else:
        print(f"\n🆕 Creating new service: {SERVICE_NAME}")
    
    # Build gcloud command
    cmd = [
        "gcloud", "run", "deploy", SERVICE_NAME,
        f"--source={AGENT_PATH}",
        f"--region={LOCATION}",
        f"--project={PROJECT_ID}",
        "--allow-unauthenticated",
        f"--set-env-vars={','.join(ENV_VARS)}",
        "--memory=2Gi",
        "--cpu=2",
        "--timeout=300",
        "--concurrency=80",
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd[:5])}...")
    print("-"*60 + "\n")
    
    try:
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print("\n❌ Deployment failed!")
            return False
        
        print("\n" + "="*60)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        
        url = get_service_url()
        if url:
            print(f"\n🌐 Service URL: {url}")
            print(f"\n📚 API Docs: {url}/docs")
            print(f"\n🧪 Test with:")
            print(f'   curl -X POST {url}/chat \\')
            print('       -H "Content-Type: application/json" \\')
            print('       -d \'{"message": "What is AAPL stock price?"}\'')
        
        return True
        
    except FileNotFoundError:
        print("\n❌ gcloud CLI not found. Please install Google Cloud SDK.")
        return False


def run_local():
    """Run the FastAPI server locally with uvicorn."""
    print("\n" + "="*60)
    print("RUNNING LOCALLY WITH UVICORN")
    print("="*60)
    
    print("\n🌐 Starting server at http://localhost:8080")
    print("📚 API docs at http://localhost:8080/docs")
    print("\nPress Ctrl+C to stop\n")
    
    os.chdir(AGENT_PATH)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--reload"
    ])


def run_docker():
    """Build and run with Docker locally."""
    print("\n" + "="*60)
    print("RUNNING LOCALLY WITH DOCKER")
    print("="*60)
    
    image_name = f"{SERVICE_NAME}:local"
    
    # Build
    print(f"\n🔨 Building Docker image: {image_name}")
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(AGENT_PATH)],
    )
    
    if result.returncode != 0:
        print("\n❌ Docker build failed!")
        return False
    
    # Run
    print(f"\n🚀 Starting container on http://localhost:8080")
    print("Press Ctrl+C to stop\n")
    
    # Prepare env vars for docker
    env_args = []
    for var in ENV_VARS:
        env_args.extend(["-e", var])
    
    # Add additional env vars from .env
    # Note: NEWS_API_KEY is deprecated - headlines_agent uses Google Search
    for key in ["FRED_API_KEY", "ARIZE_API_KEY", "ARIZE_SPACE_ID"]:
        value = os.getenv(key)
        if value:
            env_args.extend(["-e", f"{key}={value}"])
    
    subprocess.run([
        "docker", "run", "--rm", "-it",
        "-p", "8080:8080",
        *env_args,
        image_name
    ])
    return True


def show_status():
    """Show deployment status."""
    print("\n" + "="*60)
    print("CLOUD RUN SERVICE STATUS")
    print("="*60)
    
    if not check_service_exists():
        print(f"\n❌ Service '{SERVICE_NAME}' not found.")
        print("   Run: python deploy_cloud_run.py --deploy")
        return
    
    print(f"\n📊 Service: {SERVICE_NAME}")
    print(f"   Region: {LOCATION}")
    
    url = get_service_url()
    if url:
        print(f"   URL: {url}")
    
    # List revisions
    print("\n📋 Revisions:")
    print("-"*60)
    
    subprocess.run([
        "gcloud", "run", "revisions", "list",
        f"--service={SERVICE_NAME}",
        f"--region={LOCATION}",
        f"--project={PROJECT_ID}",
        "--format=table(name,active.yesno(yes='✓',no=''),createTime.date(),status.conditions[0].type)"
    ])


def show_logs():
    """Show service logs."""
    print("\n" + "="*60)
    print("CLOUD RUN LOGS")
    print("="*60)
    
    subprocess.run([
        "gcloud", "run", "services", "logs", "read", SERVICE_NAME,
        f"--region={LOCATION}",
        f"--project={PROJECT_ID}",
        "--limit=50"
    ])


def delete_service():
    """Delete the Cloud Run service."""
    print("\n" + "="*60)
    print("DELETE CLOUD RUN SERVICE")
    print("="*60)
    
    if not check_service_exists():
        print(f"\n❌ Service '{SERVICE_NAME}' not found.")
        return
    
    print(f"\n⚠️  This will delete: {SERVICE_NAME}")
    response = input("Type 'yes' to confirm: ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    subprocess.run([
        "gcloud", "run", "services", "delete", SERVICE_NAME,
        f"--region={LOCATION}",
        f"--project={PROJECT_ID}",
        "--quiet"
    ])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy Finance Agent to Cloud Run")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Cloud Run")
    parser.add_argument("--local", action="store_true", help="Run locally with uvicorn")
    parser.add_argument("--docker", action="store_true", help="Run locally with Docker")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--logs", action="store_true", help="View service logs")
    parser.add_argument("--delete", action="store_true", help="Delete the service")
    
    args = parser.parse_args()
    
    if not validate_config():
        sys.exit(1)
    
    if args.status:
        show_status()
    elif args.logs:
        show_logs()
    elif args.delete:
        delete_service()
    elif args.local:
        run_local()
    elif args.docker:
        run_docker()
    elif args.deploy:
        response = input("\n⚠️  Deploy to Cloud Run? (y/n): ")
        if response.lower() == 'y':
            success = deploy()
            sys.exit(0 if success else 1)
        else:
            print("Cancelled.")
    else:
        show_instructions()


if __name__ == "__main__":
    main()

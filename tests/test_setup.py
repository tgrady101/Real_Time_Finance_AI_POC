"""Quick setup validation script.

Run this to verify your environment is configured correctly.
"""

import os
import sys
import io
from typing import List, Tuple

# Force UTF-8 encoding for stdout (fix for Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_env_vars() -> List[Tuple[str, bool, str]]:
    """Check if required environment variables are set."""
    checks = []
    
    # Required
    checks.append(("NINJA_API_KEY", bool(os.getenv("NINJA_API_KEY")), "Required for stock/crypto data"))
    checks.append(("GOOGLE_CLOUD_PROJECT", bool(os.getenv("GOOGLE_CLOUD_PROJECT")), "Required for GCP services"))
    
    # Recommended
    checks.append(("FRED_API_KEY", bool(os.getenv("FRED_API_KEY")), "Recommended for economic data"))
    checks.append(("DATA_STORE_ID", bool(os.getenv("DATA_STORE_ID")), "Set by Terraform"))
    
    # Optional
    checks.append(("NEWS_API_KEY", bool(os.getenv("NEWS_API_KEY")), "Optional for news (100/day limit)"))
    checks.append(("ARIZE_API_KEY", bool(os.getenv("ARIZE_API_KEY")), "Optional for observability"))
    
    return checks

def check_imports() -> List[Tuple[str, bool, str]]:
    """Check if required packages are installed."""
    checks = []
    
    packages = [
        ("requests", "HTTP client"),
        ("dotenv", "Environment variables"),
        ("google.genai", "Google Generative AI"),
        ("google.cloud.storage", "GCS access"),
        ("google.cloud.discoveryengine_v1", "Data Store access"),
    ]
    
    for package, description in packages:
        try:
            __import__(package.replace(".", "_") if "." in package else package)
            checks.append((package, True, description))
        except ImportError:
            checks.append((package, False, description))
    
    return checks

def check_api_clients() -> List[Tuple[str, bool, str]]:
    """Check if API clients can be imported."""
    checks = []
    
    try:
        from src.workflow_1.api_clients.ninja_api import NinjaAPIClient
        checks.append(("NinjaAPIClient", True, "Ninja API wrapper"))
    except ImportError as e:
        checks.append(("NinjaAPIClient", False, str(e)))
    
    try:
        from src.workflow_1.api_clients.fred_api import FREDAPIClient
        checks.append(("FREDAPIClient", True, "FRED API wrapper"))
    except ImportError as e:
        checks.append(("FREDAPIClient", False, str(e)))
    
    try:
        from src.workflow_1.api_clients.news_api import NewsAPIClient
        checks.append(("NewsAPIClient", True, "News API wrapper"))
    except ImportError as e:
        checks.append(("NewsAPIClient", False, str(e)))
    
    try:
        from src.workflow_1.utils.sp500_validator import SP500Validator
        checks.append(("SP500Validator", True, "S&P 500 validation"))
    except ImportError as e:
        checks.append(("SP500Validator", False, str(e)))
    
    return checks

def print_results(title: str, checks: List[Tuple[str, bool, str]]):
    """Print check results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    for name, passed, description in checks:
        status = "✓" if passed else "✗"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {name:30} - {description}")

def main():
    """Run all validation checks."""
    print("\n🔍 Real-Time Finance AI - Setup Validation")
    
    # Load .env from project root
    from pathlib import Path
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Run checks
    env_checks = check_env_vars()
    import_checks = check_imports()
    client_checks = check_api_clients()
    
    # Print results
    print_results("Environment Variables", env_checks)
    print_results("Python Packages", import_checks)
    print_results("API Clients", client_checks)
    
    # Summary
    total_checks = len(env_checks) + len(import_checks) + len(client_checks)
    passed_checks = sum(1 for checks in [env_checks, import_checks, client_checks] for _, passed, _ in checks if passed)
    
    print(f"\n{'='*60}")
    print(f" Summary: {passed_checks}/{total_checks} checks passed")
    print(f"{'='*60}\n")
    
    # Required checks
    required_env = ["NINJA_API_KEY", "GOOGLE_CLOUD_PROJECT"]
    missing_required = [name for name, passed, _ in env_checks if name in required_env and not passed]
    
    if missing_required:
        print("⚠️  Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        print("\nPlease set these in your .env file before continuing.\n")
        return 1
    
    print("✅ Setup validation complete! You're ready to test the API clients.\n")
    print("Next steps:")
    print("  1. Run: python tests/test_api_clients.py")
    print("  2. Or test individual clients:")
    print("     python src/workflow_1/api_clients/ninja_api.py")
    print("     python src/workflow_1/api_clients/fred_api.py")
    print("     python src/workflow_1/utils/sp500_validator.py\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

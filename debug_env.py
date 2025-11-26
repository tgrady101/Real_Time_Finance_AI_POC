import os
from pathlib import Path
from dotenv import load_dotenv

# Calculate path
project_root = Path(__file__).parent
env_path = project_root / '.env'

print(f"Project Root: {project_root}")
print(f"Env Path: {env_path}")
print(f"Env Exists: {env_path.exists()}")

if env_path.exists():
    print("\n--- .env Content ---")
    print(env_path.read_text())
    print("--------------------\n")

# Load with override
print("Loading .env with override=True...")
load_dotenv(dotenv_path=env_path, override=True)

# Check variable
fred_key = os.getenv('FRED_API_KEY')
print(f"FRED_API_KEY: {fred_key}")

if fred_key == 'xyz789yourkeyhere':
    print("❌ STILL PLACEHOLDER!")
else:
    print("✅ CORRECT KEY!")

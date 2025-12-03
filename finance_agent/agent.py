"""Finance Agent - ADK Entry Point.

This module defines the root_agent that ADK discovers and uses
for `adk web`, `adk run`, and `adk deploy` commands.

The agent provides S&P 500 market data analysis through a multi-agent
architecture with specialized sub-agents for different data sources.
"""

import sys
from pathlib import Path

# Add finance_agent directory to path for local imports
agent_dir = Path(__file__).parent
sys.path.insert(0, str(agent_dir))

# Also add project root for backwards compatibility (local dev)
project_root = agent_dir.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv

# Try multiple .env locations (Cloud Run vs local dev)
env_paths = [
    agent_dir / '.env',           # Cloud Run: .env in finance_agent/
    project_root / '.env',        # Local dev: .env in project root
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        break

# NOTE: Arize tracing is NOT initialized here for Cloud Run deployments.
# Cloud Run has its own Cloud Trace infrastructure.
# For local development with `adk web`, tracing can be initialized manually.

# Import the agent creation function
# Try bundled version first (Cloud Run), fall back to src (local dev)
try:
    from workflow_1.agents.root_agent import create_root_agent
except ImportError:
    from src.workflow_1.agents.root_agent import create_root_agent

# Create the root agent - this is what ADK discovers
root_agent = create_root_agent()

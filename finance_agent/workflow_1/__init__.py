"""Workflow 1 - Finance Agent Core Module.

This is the primary application module for the S&P 500 Finance Chatbot.
Designed for deployment to Google Cloud Run via FastAPI.

Key Components:
- agents/: Root orchestrator and specialized sub-agents
- utils/: S&P 500 validation, query classification, dynamic model routing
- mcp_clients/: Yahoo Finance MCP toolset wrapper
- api_clients/: FRED and NewsAPI clients
- arize_observability/: Tracing and evaluation

Usage:
    from finance_agent.workflow_1.agents import create_root_agent
    from finance_agent.workflow_1.config import Config
"""

from .config import Config

__all__ = ["Config"]

"""
Backend API Gateway Package

Unified FastAPI gateway that connects:
- LangGraph pipeline (graph/)
- Voice services (voice/)
- MCP tools (mcp_server/)
"""

from backend.api_gateway import app

__all__ = ["app"]


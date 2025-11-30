from typing import Any, Dict

from mcp_server.tools import web_search_sync, rag_search_sync


def call_mcp_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small helper used by LangGraph nodes to call MCP tools.

    For now we call the synchronous Python shims inside the same process
    (web_search_sync / rag_search_sync). This keeps the graph simple
    while still matching the MCP tool contract.

    Args:
        tool_name: "web.search" or "rag.search"
        args: Dict of keyword arguments for the tool.

    Returns:
        Dict with the tool's normalized output (usually {"results": [...]}).
    """

    if tool_name == "web.search":
        return web_search_sync(**args)

    if tool_name == "rag.search":
        return rag_search_sync(**args)

    # Fallback: unknown tool
    return {"error": f"Unknown MCP tool: {tool_name}", "results": []}

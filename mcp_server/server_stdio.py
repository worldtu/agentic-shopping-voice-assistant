import sys
import json
import asyncio
from typing import Any, Dict

from mcp_server.tools import web_search, rag_search
from mcp_server.util import log_request_response


async def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a single MCP-style request.

    Expected input schema (one JSON per line):

        {
          "id": "optional-correlation-id",
          "tool": "web.search" | "rag.search" | "tools.list",
          "args": { ... }
        }
    """

    tool = req.get("tool")
    args = req.get("args", {}) or {}
    request_id = req.get("id")

    if tool == "web.search":
        result = await web_search(**args)
    elif tool == "rag.search":
        result = await rag_search(**args)
    elif tool == "tools.list":
        # Simple discovery endpoint
        result = {
            "tools": [
                {
                    "name": "web.search",
                    "description": "Live web search over product pages",
                },
                {
                    "name": "rag.search",
                    "description": "Private Amazon 2020 vector search",
                },
            ]
        }
    else:
        result = {"error": f"Unknown tool: {tool}"}

    # Also log at server level (in addition to per-tool logging)
    log_request_response(tool or "unknown", args, result)

    return {"id": request_id, "tool": tool, "result": result}


def main() -> None:
    """
    Very small stdio loop:

    - Read one JSON object per line from stdin
    - Process it with handle_request(...)
    - Print result JSON on its own line to stdout
    """

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(
                json.dumps({"error": "invalid_json", "raw": line}) + "\n"
            )
            sys.stdout.flush()
            continue

        result = loop.run_until_complete(handle_request(req))
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()


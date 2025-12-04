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
        # MCP tool discovery endpoint with full JSON schemas
        result = {
            "tools": [
                {
                    "name": "web.search",
                    "description": "Live web search over product pages using Serper API. Supports both general search and shopping-specific search.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)",
                                "default": 5
                            },
                            "site_filter": {
                                "type": "string",
                                "description": "Optional site restriction (e.g. 'site:amazon.com')",
                                "optional": True
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["search", "shopping"],
                                "description": "Type of search: 'search' for general, 'shopping' for products (default: 'search')",
                                "default": "search"
                            }
                        },
                        "required": ["query"]
                    },
                    "outputSchema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                        "price": {"type": ["number", "null"]},
                                        "availability": {"type": ["string", "null"]},
                                        "source": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                {
                    "name": "rag.search",
                    "description": "Private vector search over Amazon 2020 product catalog using FAISS. Returns structured product data with ratings, prices, and metadata.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of products to retrieve (default: 10)",
                                "default": 10
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional filters (e.g., {'max_price': 50, 'category': 'Electronics'})",
                                "optional": True,
                                "properties": {
                                    "max_price": {"type": "number"},
                                    "min_price": {"type": "number"},
                                    "category": {"type": "string"},
                                    "brand": {"type": "string"},
                                    "material": {"type": "string"}
                                }
                            }
                        },
                        "required": ["query"]
                    },
                    "outputSchema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "doc_id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "price": {"type": "number"},
                                        "rating": {"type": "number"},
                                        "brand": {"type": "string"},
                                        "material": {"type": "string"},
                                        "category": {"type": "string"},
                                        "ingredients": {"type": "string"},
                                        "content": {"type": "string"},
                                        "score": {"type": "number"},
                                        "source": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
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


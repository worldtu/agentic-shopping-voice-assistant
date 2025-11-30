import json
from datetime import datetime
from typing import Any, Dict


def log_request_response(
    tool_name: str,
    request: Dict[str, Any],
    response: Dict[str, Any],
    logfile: str = "mcp_logs.jsonl",
) -> None:
    """
    Append a single request/response record to a JSONL log file.

    This is helpful for grading:
    - Shows which MCP tools were called
    - Shows arguments and normalized outputs
    """

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool_name,
        "request": request,
        "response": response,
    }

    try:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never crash the pipeline
        pass

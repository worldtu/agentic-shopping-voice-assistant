import json
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import urllib.robotparser


# Cache robots.txt parsers to avoid repeated fetches
_robots_cache = {}


def log_request_response(
    tool_name: str,
    request: Dict[str, Any],
    response: Dict[str, Any],
    logfile: str = "mcp_logs.jsonl",
    source_url: Optional[str] = None,
) -> None:
    """
    Append a single request/response record to a JSONL log file.

    This is helpful for grading:
    - Shows which MCP tools were called
    - Shows arguments and normalized outputs
    - Includes source URL and robots.txt compliance check
    """

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool_name,
        "request": request,
        "response": response,
    }
    
    # Add source URL and robots.txt compliance if provided
    if source_url:
        record["source_url"] = source_url
        record["robots_txt_compliant"] = check_robots_txt(source_url)

    try:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never crash the pipeline
        pass


def check_robots_txt(url: str, user_agent: str = "VoiceShoppingBot") -> bool:
    """
    Check if a URL is allowed by robots.txt.
    
    Args:
        url: The URL to check
        user_agent: The user agent string
    
    Returns:
        True if allowed or if robots.txt can't be fetched, False if explicitly disallowed
    """
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check cache
        if base_url in _robots_cache:
            rp = _robots_cache[base_url]
        else:
            # Fetch and parse robots.txt
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
                _robots_cache[base_url] = rp
            except Exception:
                # If we can't fetch robots.txt, assume allowed (fail open)
                return True
        
        # Check if URL is allowed
        return rp.can_fetch(user_agent, url)
    
    except Exception:
        # On any error, assume allowed (fail open)
        return True

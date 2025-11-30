import time
from typing import Any, Dict, Optional


class SimpleTTLCache:
    """
    Very small in-memory TTL cache.

    This is only meant for demo / homework:
    - Not thread-safe
    - Lives in process memory only
    """

    def __init__(self, ttl_seconds: int = 180):
        self.ttl = ttl_seconds
        self.store: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if not expired, else None."""
        if key not in self.store:
            return None

        value, expire_ts = self.store[key]
        if time.time() < expire_ts:
            return value

        # Expired → remove from cache
        del self.store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value with TTL."""
        expire_ts = time.time() + self.ttl
        self.store[key] = (value, expire_ts)


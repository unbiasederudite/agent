"""A trivial, deterministic ITool implementation, used to prove the tool-calling plumbing."""

from datetime import UTC, datetime
from typing import Any


class GetCurrentTimeTool:
    """Returns the current UTC time. No parameters, no network, no config."""

    name = "get_current_time"
    description = "Get the current date and time in UTC."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now(UTC).isoformat()

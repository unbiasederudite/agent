"""A tool proving the tool-calling plumbing, with one real validated parameter."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MIN_OFFSET_MINUTES = -1439
_MAX_OFFSET_MINUTES = 1439


class GetCurrentTimeParams(BaseModel):
    """Arguments for retrieving the current time."""

    model_config = ConfigDict(extra="forbid")

    utc_offset_minutes: int | None = Field(
        default=None,
        description="Offset from UTC in minutes, e.g. 330 for +05:30. Omit for UTC.",
    )

    @field_validator("utc_offset_minutes")
    @classmethod
    def _valid_offset(cls, v: int | None) -> int | None:
        """Reject an out-of-range offset as a normal validation error, not a runtime crash.

        Args:
            v: The offset to validate, in minutes.

        Returns:
            int | None: `v`, unchanged.

        Raises:
            ValueError: if the offset is out of range.
        """
        if v is not None and not (_MIN_OFFSET_MINUTES <= v <= _MAX_OFFSET_MINUTES):
            raise ValueError(
                f"utc_offset_minutes must be between {_MIN_OFFSET_MINUTES} and "
                f"{_MAX_OFFSET_MINUTES}, got {v}"
            )
        return v


class GetCurrentTimeTool:
    """Returns the current time, optionally at a UTC offset. No network, no config."""

    name = "get_current_time"
    description = "Get the current date and time, optionally at a UTC offset."
    parameters_model: type[BaseModel] = GetCurrentTimeParams

    async def execute(self, **kwargs: Any) -> str:
        """Return the current time as an ISO 8601 string, at `utc_offset_minutes` or UTC.

        Returns:
            str: the current time, in ISO 8601 format.
        """
        offset_minutes = kwargs.get("utc_offset_minutes")
        tz = UTC if offset_minutes is None else timezone(timedelta(minutes=offset_minutes))
        return datetime.now(tz).isoformat()

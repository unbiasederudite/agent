# tools

Concrete tool implementations, exposed to an LLM as function-calling schemas via
`ToolConfig`/`ToolRegistry`. `ITool.execute()` is invoked by whichever `IStrategy` a
request routes through (currently `ReactStrategy`, `core/strategies/`).

## Contents

- `get_current_time.py` -- `GetCurrentTimeTool`, returns the current time as an ISO 8601
  string, at a caller-given `utc_offset_minutes` or UTC. No network, no config. Stdlib
  `datetime`/`timedelta` only -- deliberately not `zoneinfo`/IANA zone names, which need a
  timezone database (`tzdata`) not bundled with Python on Windows. `GetCurrentTimeParams`
  (its `parameters_model`) validates the offset is within +/-1439 minutes, rejecting an
  out-of-range value as a normal validation error.

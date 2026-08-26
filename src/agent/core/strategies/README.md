# strategies

Reasoning-loop algorithm implementations of `IStrategy` (`core/protocols/istrategy.py`).
Selected per-agent via `AgentConfig.strategy`, or per-request via `AgentRunRequest.strategy`.

## Contents

- `react.py` — `ReactStrategy`, the ReAct loop: call the LLM, execute any requested tool
  calls concurrently, feed the results back as `role="tool"` messages, repeat up to
  `AgentConfig.max_tool_iterations`. On hitting that cap, forces one final call with tools
  omitted so the LLM answers instead of requesting another tool call.

# registries

Runtime name-to-instance maps, populated by `factories/` at startup and queried by `services/` at request time.

## Contents

- `base.py` — `_Registry`, generic name-to-instance map shared by the other registries.
- `llm.py` — `LLMRegistry`, maps a litellm-format model string to its `ILLM` instance
- `agent.py` — `AgentRegistry`, maps an agent name to its `AgentConfig`
- `tool.py` — `ToolRegistry`, maps a tool name to its `ITool` instance
- `strategy.py` — `StrategyRegistry`, maps a strategy name to its `IStrategy` instance
- `guardrail.py` — `GuardrailRegistry`, maps a guardrail name to its `IGuardrail` instance

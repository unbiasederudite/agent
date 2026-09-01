# strategies

Reasoning-loop algorithm implementations of `IStrategy` (`core/protocols/istrategy.py`).
Selected per-agent via `AgentConfig.strategy`, or per-request via `AgentRunRequest.strategy`.

## Contents

- `react.py` — `ReactStrategy`, the ReAct loop: call the LLM, execute any requested tool
  calls concurrently, feed the results back as `role="tool"` messages, repeat up to
  `AgentConfig.max_tool_iterations`. On hitting that cap, forces one final call with tools
  omitted so the LLM answers instead of requesting another tool call. That call declares no
  `tools`, and Bedrock's Converse API rejects toolUse/toolResult content blocks in a request
  with no `toolConfig`, even when they only replay earlier rounds -- but this file does
  nothing about that itself: folding such content out of a no-tools request is the `ILLM`
  implementation's contractual job, done automatically in `LiteLLMAdapter.complete()`. The
  strategy's own message list and the returned `Turn.messages` therefore keep the real
  tool-call/tool-result messages, which are valid again on any later call that does declare
  tools. The same protection covers the *main loop's* calls for free, with no special-casing
  anywhere here: when the resolved `tools` dict is empty (e.g. `AgentRunService.run()`'s
  documented `tools=[]` override) `tool_schemas` is `None` too, so a main-loop call replaying
  an earlier turn's tool exchange is flattened downstream exactly like the forced final one --
  a gap that existed for as long as this flattening was applied by hand at individual call
  sites.

  The forced final call also appends one scoped instruction message after `messages`, for
  that outbound request only -- never stored in `messages`, so never part
  of the returned `Turn` or persisted session history. Needed, not just polish: if the cap is
  hit mid-tool-use, `messages` ends on unflushed tool content, and flattening folds that into a
  trailing synthetic assistant message with nothing after it -- which Anthropic treats as a
  prefill to continue, not a request for a fresh reply. The trailing instruction keeps the real
  outbound request ending on `role="user"` regardless, sidestepping that.

  If `AgentConfig.max_tool_result_chars` is set, a tool's result content (success or a
  caught tool-raised error) longer than that cap is truncated with a trailing marker before
  being fed back into the message list; the strategy's own short, fixed error strings
  (unoffered tool, bad JSON, non-object arguments) are never truncated. A schema-validation
  failure (the LLM's own arguments don't match the tool's `parameters_model`) is also never
  truncated by this cap, but isn't "short and fixed" the same way -- it echoes back
  LLM-supplied field names and values, so `_format_validation_error` bounds it internally
  (capped error count, each field/message truncated) instead of relying on
  `max_tool_result_chars`.

  If `AgentConfig.max_tool_calls_per_round` is set, only that many of one LLM response's tool
  calls actually execute -- the excess are skipped in the order the response listed them, each
  replaced with a short, never-truncated error result naming the limit, so the LLM sees they
  didn't run and can adjust next round instead of the loop silently dropping them.

  If `AgentConfig.max_tool_results_total_chars` is set, it caps the *combined* result content
  across the whole run (every round together), independent of the per-call
  `max_tool_result_chars` -- many individually-small results still add up. It is applied to a
  round's results once they are known (they execute concurrently, so their sizes can't be
  predicted beforehand): results are walked in order, and once the running total has reached
  the budget every further result -- including a skipped call's own marker, which is a result
  like any other -- is replaced with a short omission marker instead of its real content. The
  running total carries across rounds and counts the already-truncated length, so
  `max_tool_result_chars` applies first and only what actually enters the message list counts.
  Not a hard ceiling by itself: the one result that crosses the threshold is still admitted in
  full before the marker applies to anything after it -- pair it with `max_tool_result_chars`
  for an actual per-result size guarantee too.

  Known limitation, not fixed: if a mid-run context-window overflow triggers
  `AgentRunService`'s compact-and-retry-once fallback, the retry reruns this whole loop
  from the start -- any tool calls already executed in the failed attempt (including ones with
  real side effects) can execute again. A cache keyed on `(tool name, arguments)` was tried and
  then deliberately removed: it would have made an *exact* duplicate call reuse the earlier
  attempt's result instead of re-executing, but that's only safe for tools whose result doesn't
  depend on *when* they run -- `GetCurrentTimeTool`, the one tool that exists today, is exactly
  the counterexample (a cached "current time" served on a later retry would be stale and
  wrong), so the cache traded a hypothetical future safety benefit for a real, immediate
  correctness bug. Safe today only because that one tool is read-only; a side-effecting tool
  would need real idempotency keys or a checkpoint-and-resume mechanism (neither built, and
  naive result-caching is not a substitute for either) before this stops being safe by
  accident.

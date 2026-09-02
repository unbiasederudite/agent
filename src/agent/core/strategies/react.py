"""ReAct (reason, act, observe) tool-calling loop -- the first IStrategy implementation."""

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from agent.core.models.message import Message, ToolCall
from agent.core.models.turn import Turn
from agent.core.models.usage import Usage, sum_usage
from agent.core.protocols.illm import ILLM
from agent.core.protocols.itool import ITool

logger = logging.getLogger(__name__)

_OMITTED_MARKER = "Error: tool result omitted -- aggregate tool-output budget exhausted"
_MAX_VALIDATION_ERRORS_SHOWN = 5
_MAX_VALIDATION_FIELD_CHARS = 200


def _tool_schema(tool: ITool) -> dict[str, Any]:
    """Build an OpenAI-format function schema for `tool`."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_model.model_json_schema(),
        },
    }


def _truncate(content: str, max_chars: int | None) -> str:
    """Cap `content` at `max_chars`, appending a marker noting what was cut.

    `None` means uncapped -- returns `content` unchanged.
    """
    if max_chars is None or len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n...[truncated, {len(content) - max_chars} more characters]"


def _apply_aggregate_budget(
    results: list[Message], running_total: int, max_total_chars: int | None
) -> tuple[list[Message], int]:
    """Cap the combined tool-result content across a whole run, not just one call.

    Applied once a round's results are known (not predicted beforehand -- calls in a round
    execute concurrently, so individual result sizes aren't known until they return).
    Walks `results` in order: once `running_total` has already reached `max_total_chars`,
    every further result in this round is replaced with a short omission marker instead of
    its real content. Earlier rounds' already-appended results are never touched -- this
    only affects what's about to be added. `None` means uncapped; returns `results`
    unchanged along with the updated running total.
    """
    if max_total_chars is None:
        return results, running_total + sum(len(message.content or "") for message in results)
    trimmed: list[Message] = []
    total = running_total
    for message in results:
        if total >= max_total_chars:
            trimmed.append(
                Message(
                    role="tool",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    content=_OMITTED_MARKER,
                )
            )
        else:
            trimmed.append(message)
            total += len(message.content or "")
    return trimmed, total


def _tool_result_message(call: ToolCall, content: str) -> Message:
    """A role="tool" result message for `call`, the shared shape every result uses."""
    return Message(role="tool", tool_call_id=call.id, name=call.function.name, content=content)


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a ValidationError into a short "field: message" list for the LLM to read.

    Pydantic's own str(exc) includes a per-error documentation URL, which is noise for an
    error whose only audience is the LLM being asked to correct its own tool call. Bounded
    internally (error count, and each field/message) rather than relying on
    `max_tool_result_chars` -- unlike this function's sibling fixed error strings, this
    content echoes back LLM-supplied field names and values, so it isn't actually short and
    fixed without a cap of its own.
    """
    errors = exc.errors()
    parts = []
    for err in errors[:_MAX_VALIDATION_ERRORS_SHOWN]:
        field = ".".join(str(loc) for loc in err["loc"]) or "(root)"
        parts.append(
            f"{_truncate(field, _MAX_VALIDATION_FIELD_CHARS)}: "
            f"{_truncate(err['msg'], _MAX_VALIDATION_FIELD_CHARS)}"
        )
    message = "; ".join(parts)
    omitted = len(errors) - _MAX_VALIDATION_ERRORS_SHOWN
    if omitted > 0:
        message += f"; ...and {omitted} more error(s)"
    return message


def _skipped_call_message(call: ToolCall, max_tool_calls_per_round: int) -> Message:
    """A fixed, non-truncated result for a tool call skipped by `max_tool_calls_per_round`.

    Lets the LLM see the call didn't run (and why) instead of silently vanishing -- same
    convention as `_execute_call`'s other fixed, short error strings, never truncated.
    """
    return _tool_result_message(
        call,
        f"Error: skipped -- this round requested more than the "
        f"{max_tool_calls_per_round} tool calls allowed at once",
    )


async def _execute_call(
    tools: dict[str, ITool], call: ToolCall, max_tool_result_chars: int | None
) -> Message:
    """Run one tool call and return its result as a role="tool" message.

    Never raises -- any failure (a tool not offered for this call, bad JSON, arguments that
    fail schema validation, the tool itself raising, or a tool returning something that
    isn't a string) becomes the message's error content instead, so the loop can keep going
    and the LLM can see and react to the failure. `max_tool_result_chars` caps the
    tool-controlled content only (the success result and the caught-exception message) --
    our own short, fixed error strings are never truncated.
    """
    name = call.function.name
    tool = tools.get(name)
    if tool is None:
        logger.warning("tool call named '%s', which was not offered for this call", name)
        return _tool_result_message(call, f"Error: tool '{name}' was not offered for this call")
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        logger.warning("tool '%s' call had malformed JSON arguments: %s", name, exc)
        return _tool_result_message(call, f"Error: invalid arguments: {exc}")
    if not isinstance(arguments, dict):
        logger.warning("tool '%s' call arguments were not a JSON object", name)
        return _tool_result_message(call, "Error: arguments must be a JSON object")
    try:
        validated = tool.parameters_model.model_validate(arguments)
    except ValidationError as exc:
        formatted = _format_validation_error(exc)
        logger.warning("tool '%s' call failed argument validation: %s", name, formatted)
        return _tool_result_message(call, f"Error: invalid arguments: {formatted}")
    except Exception as exc:  # noqa: BLE001 -- a tool's own validator can raise anything
        logger.warning(
            "tool '%s' argument validation raised: %s",
            name,
            exc,
            extra={"exception_type": type(exc).__name__},
        )
        return _tool_result_message(call, f"Error: invalid arguments for tool '{name}'")
    logger.info("executing tool '%s'", name)
    start = time.monotonic()
    try:
        result = await tool.execute(**validated.model_dump())
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "tool '%s' completed in %.1fms, result length %d", name, duration_ms, len(result)
        )
        return _tool_result_message(call, _truncate(result, max_tool_result_chars))
    except Exception as exc:  # noqa: BLE001 -- tool can raise anything, or violate -> str; never crash
        logger.warning(
            "tool '%s' raised: %s", name, exc, extra={"exception_type": type(exc).__name__}
        )
        return _tool_result_message(
            call, _truncate(f"Error: tool '{name}' failed: {exc}", max_tool_result_chars)
        )


class ReactStrategy:
    """The ReAct loop: call the LLM, execute any requested tool calls, repeat.

    Terminates when the LLM responds without `tool_calls`, or after `max_iterations`
    rounds -- whichever comes first. On hitting the cap, one final call is made with
    `tools` omitted so the LLM is forced to answer instead of requesting another tool
    call; that call's own `finish_reason` is returned as-is, nothing invented. `ILLM`
    implementations flatten tool-shaped content out of any request that declares no
    `tools`, so what's sent is safe to transmit and what the returned `Turn` carries
    stays the real exchange. The main loop's own calls get the same protection for free
    on any run where `tools` is empty (`tool_schemas` is then `None` too) -- no
    special-casing here for either call.

    The forced-final call additionally appends one scoped, call-only instruction message
    after `messages`, never stored in `messages` itself and so never part of the returned
    `Turn` or persisted session history. This is needed, not just polish: if the cap was
    hit mid-tool-use, `messages` ends on unflushed tool content, and flattening folds
    that into a trailing synthetic assistant message with nothing after it -- which at
    least one provider (Anthropic) treats as a prefill to continue, not a request for a
    fresh reply. Appending a trailing user-role instruction keeps the actual outbound
    request properly ending on `role="user"`, sidestepping that and making the ask
    explicit either way.

    `max_tool_calls_per_round` caps how many of one response's tool calls actually execute
    (the excess are skipped with a short error result each), and
    `max_tool_results_total_chars` caps the combined result content across the whole run,
    replacing every result past that budget with a short omission marker.
    """

    async def run(
        self,
        messages: list[Message],
        llm: ILLM,
        tools: dict[str, ITool],
        max_iterations: int,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        max_tool_calls_per_round: int | None = None,
        max_tool_results_total_chars: int | None = None,
    ) -> Turn:
        """Run the ReAct loop and return the final Turn."""
        tool_schemas = [_tool_schema(tool) for tool in tools.values()] or None

        messages = list(messages)
        input_length = len(messages)
        total_usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        tool_results_total_chars = 0

        for iteration in range(1, max_iterations + 1):
            logger.debug("calling the LLM, iteration %d/%d", iteration, max_iterations)
            start = time.monotonic()
            completion = await llm.complete(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=tool_schemas,
            )
            duration_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "LLM responded, iteration %d/%d: finish_reason=%s prompt=%d completion=%d "
                "total=%d, %.1fms",
                iteration,
                max_iterations,
                completion.finish_reason,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                completion.usage.total_tokens,
                duration_ms,
            )
            total_usage = sum_usage(total_usage, completion.usage)
            if not completion.message.tool_calls:
                messages.append(completion.message)
                return Turn(
                    messages=messages[input_length:],
                    usage=total_usage,
                    finish_reason=completion.finish_reason,
                    final_total_tokens=completion.usage.total_tokens,
                )
            messages.append(completion.message)
            tool_calls = completion.message.tool_calls
            skipped: list[Message] = []
            if max_tool_calls_per_round is not None and len(tool_calls) > max_tool_calls_per_round:
                logger.info(
                    "round requested %d tool calls, more than max_tool_calls_per_round=%d, "
                    "skipping the rest",
                    len(tool_calls),
                    max_tool_calls_per_round,
                )
                skipped = [
                    _skipped_call_message(call, max_tool_calls_per_round)
                    for call in tool_calls[max_tool_calls_per_round:]
                ]
                tool_calls = tool_calls[:max_tool_calls_per_round]
            executed = await asyncio.gather(
                *(_execute_call(tools, call, max_tool_result_chars) for call in tool_calls)
            )
            results, tool_results_total_chars = _apply_aggregate_budget(
                [*executed, *skipped], tool_results_total_chars, max_tool_results_total_chars
            )
            if any(message.content == _OMITTED_MARKER for message in results):
                logger.info(
                    "aggregate tool-output budget (%s chars) reached this round, omitting the rest",
                    max_tool_results_total_chars,
                )
            messages.extend(results)

        logger.warning(
            "max_iterations (%d) exhausted, forcing a final call without tools", max_iterations
        )
        logger.debug("calling the LLM for the forced final answer")
        start = time.monotonic()
        final = await llm.complete(
            [
                *messages,
                Message(
                    role="user",
                    content="No further tool calls are available. Provide your final answer now.",
                ),
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=None,
        )
        duration_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "LLM responded (forced final): finish_reason=%s prompt=%d completion=%d total=%d, "
            "%.1fms",
            final.finish_reason,
            final.usage.prompt_tokens,
            final.usage.completion_tokens,
            final.usage.total_tokens,
            duration_ms,
        )
        total_usage = sum_usage(total_usage, final.usage)
        messages.append(final.message)
        return Turn(
            messages=messages[input_length:],
            usage=total_usage,
            finish_reason=final.finish_reason,
            final_total_tokens=final.usage.total_tokens,
        )

"""ReAct (reason, act, observe) tool-calling loop."""

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

_OMITTED_MARKER = "Error: tool result omitted — aggregate tool-output budget exhausted"
_MAX_VALIDATION_ERRORS_SHOWN = 5
_MAX_VALIDATION_FIELD_CHARS = 200


def _tool_schema(tool: ITool) -> dict[str, Any]:
    """Build an OpenAI-format function schema for `tool`.

    Args:
        tool: Tool to build a schema for.

    Returns:
        dict[str, Any]: the OpenAI-format function schema.
    """
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

    Args:
        content: Text to cap.
        max_chars: Cap in characters. `None` means uncapped.

    Returns:
        str: `content`, truncated if it exceeded `max_chars`.
    """
    if max_chars is None or len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n...[truncated, {len(content) - max_chars} more characters]"


def _apply_aggregate_budget(
    results: list[Message], running_total: int, max_total_chars: int | None
) -> tuple[list[Message], int]:
    """Cap the combined tool-result content across a whole run, not just one call.

    Args:
        results: This round's tool-result messages, in order.
        running_total: Combined character count already consumed by earlier rounds.
        max_total_chars: Cap in characters. `None` means uncapped.

    Returns:
        The updated results and running total.
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
    """Build a role="tool" result message for `call`.

    Args:
        call: The tool call this result answers.
        content: The result text.

    Returns:
        Message: the role="tool" result message.
    """
    return Message(role="tool", tool_call_id=call.id, name=call.function.name, content=content)


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a ValidationError into a short "field: message" list for the LLM to read.

    Args:
        exc: The validation error to format.

    Returns:
        A short, bounded error message.
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
    """Build a fixed, non-truncated result for a tool call skipped this round.

    Args:
        call: The skipped tool call.
        max_tool_calls_per_round: The limit that caused the skip.

    Returns:
        Message: the role="tool" result message noting the skip.
    """
    return _tool_result_message(
        call,
        f"Error: skipped — this round requested more than the "
        f"{max_tool_calls_per_round} tool calls allowed at once",
    )


async def _execute_call(
    tools: dict[str, ITool], call: ToolCall, max_tool_result_chars: int | None
) -> Message:
    """Run one tool call and return its result as a role="tool" message. Never raises.

    Args:
        tools: Tools available for this call, by name.
        call: The tool call to execute.
        max_tool_result_chars: Cap on the result content, in characters. `None` means
            uncapped.

    Returns:
        A role="tool" result message.
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
    except Exception as exc:  # noqa: BLE001 — a tool's own validator can raise anything
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
    except Exception as exc:  # noqa: BLE001 — tool can raise anything, or violate -> str; never crash
        logger.warning(
            "tool '%s' raised: %s", name, exc, extra={"exception_type": type(exc).__name__}
        )
        return _tool_result_message(
            call, _truncate(f"Error: tool '{name}' failed: {exc}", max_tool_result_chars)
        )


class ReactStrategy:
    """The ReAct loop: call the LLM, execute any requested tool calls, repeat."""

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
        """Run the ReAct loop and return the final Turn.

        Args:
            messages: The initial message list.
            llm: The LLM to call.
            tools: Tools available to invoke, by name.
            max_iterations: Cap on iterations.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max output tokens.
            max_tool_result_chars: Cap on a single tool result's length.
            max_tool_calls_per_round: Cap on tool calls executed per round.
            max_tool_results_total_chars: Cap on combined tool-result length for the run.

        Returns:
            Turn: the run's aggregate result.
        """
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

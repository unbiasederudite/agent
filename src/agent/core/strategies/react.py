"""ReAct (reason, act, observe) tool-calling loop -- the first IStrategy implementation."""

import asyncio
import json
from typing import Any

from agent.core.models.completion import Completion
from agent.core.models.message import Message, ToolCall
from agent.core.models.usage import Usage
from agent.core.protocols.illm import ILLM
from agent.core.protocols.itool import ITool


def _tool_schema(tool: ITool) -> dict[str, Any]:
    """Build an OpenAI-format function schema for `tool`."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _sum_usage(a: Usage, b: Usage) -> Usage:
    """Add two Usage totals together, field by field."""
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


async def _execute_call(tools: dict[str, ITool], call: ToolCall) -> Message:
    """Run one tool call and return its result as a role="tool" message.

    Never raises -- any failure (a tool not offered for this call, bad JSON, the tool
    itself raising, or a tool returning something that isn't a string) becomes the
    message's error content instead, so the loop can keep going and the LLM can see and
    react to the failure.
    """
    name = call.function.name
    tool = tools.get(name)
    if tool is None:
        return Message(
            role="tool",
            tool_call_id=call.id,
            name=name,
            content=f"Error: tool '{name}' was not offered for this call",
        )
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        return Message(
            role="tool",
            tool_call_id=call.id,
            name=name,
            content=f"Error: invalid arguments: {exc}",
        )
    if not isinstance(arguments, dict):
        return Message(
            role="tool",
            tool_call_id=call.id,
            name=name,
            content="Error: arguments must be a JSON object",
        )
    try:
        result = await tool.execute(**arguments)
        return Message(role="tool", tool_call_id=call.id, name=name, content=result)
    except Exception as exc:  # noqa: BLE001 -- tool can raise anything, or violate -> str; never crash
        return Message(role="tool", tool_call_id=call.id, name=name, content=f"Error: {exc}")


class ReactStrategy:
    """The ReAct loop: call the LLM, execute any requested tool calls, repeat.

    Terminates when the LLM responds without `tool_calls`, or after `max_iterations`
    rounds -- whichever comes first. On hitting the cap, one final call is made with
    `tools` omitted so the LLM is forced to answer instead of requesting another tool
    call; that call's own `finish_reason` is returned as-is, nothing invented.
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
    ) -> Completion:
        """Run the ReAct loop and return the final Completion."""
        tool_schemas = [_tool_schema(tool) for tool in tools.values()] or None

        messages = list(messages)
        total_usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        for _ in range(max_iterations):
            completion = await llm.complete(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=tool_schemas,
            )
            total_usage = _sum_usage(total_usage, completion.usage)
            if not completion.message.tool_calls:
                return Completion(
                    message=completion.message,
                    usage=total_usage,
                    finish_reason=completion.finish_reason,
                )
            messages.append(completion.message)
            results = await asyncio.gather(
                *(_execute_call(tools, call) for call in completion.message.tool_calls)
            )
            messages.extend(results)

        final = await llm.complete(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=None,
        )
        total_usage = _sum_usage(total_usage, final.usage)
        return Completion(
            message=final.message, usage=total_usage, finish_reason=final.finish_reason
        )

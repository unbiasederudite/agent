"""Tests for ReactStrategy -- the ReAct tool-calling loop."""

import asyncio
from typing import Any

from agent.core.models.completion import Completion
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.usage import Usage
from agent.core.protocols.itool import ITool
from agent.core.strategies.react import ReactStrategy


class _FakeLLM:
    """Returns queued completions in order, one per call; records every call's args."""

    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "tools": tools,
            }
        )
        return self._completions[len(self.calls) - 1]


class _EchoTool:
    name = "echo"
    description = "Echoes its input."
    parameters: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs["value"])


class _RaisingTool:
    name = "boom"
    description = "Always raises."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("tool exploded")


class _ConcurrentTool:
    """Records the max number of overlapping in-flight executions."""

    name = "slow"
    description = "Sleeps briefly to prove concurrent execution."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, counter: list[int], max_seen: list[int]) -> None:
        self._counter = counter
        self._max_seen = max_seen

    async def execute(self, **kwargs: Any) -> str:
        self._counter[0] += 1
        self._max_seen[0] = max(self._max_seen[0], self._counter[0])
        await asyncio.sleep(0.01)
        self._counter[0] -= 1
        return "done"


def _usage(n: int = 1) -> Usage:
    return Usage(prompt_tokens=n, completion_tokens=n, total_tokens=2 * n)


def _final_completion(content: str = "final answer") -> Completion:
    return Completion(
        message=Message(role="assistant", content=content), usage=_usage(), finish_reason="stop"
    )


def _tool_call_completion(calls: list[ToolCall]) -> Completion:
    return Completion(
        message=Message(role="assistant", content=None, tool_calls=calls),
        usage=_usage(),
        finish_reason="tool_calls",
    )


def _call(id_: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(id=id_, function=ToolCallFunction(name=name, arguments=arguments))


async def test_run_given_no_tool_calls_returns_after_one_llm_call():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert len(llm.calls) == 1
    assert turn.message.content == "final answer"
    assert turn.finish_reason == "stop"


async def test_run_given_no_tool_calls_turn_messages_is_just_the_final_answer():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert turn.messages == [Message(role="assistant", content="final answer")]


async def test_run_given_one_tool_call_executes_it_and_calls_llm_again():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]),
            _final_completion("you said hi"),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run(
        [Message(role="user", content="echo hi")], llm, tools, max_iterations=10
    )

    assert len(llm.calls) == 2
    assert turn.message.content == "you said hi"
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1] == Message(
        role="tool", tool_call_id="call_1", name="echo", content="hi"
    )
    assert second_call_messages[-2].tool_calls == [_call("call_1", "echo", '{"value": "hi"}')]


async def test_run_given_one_tool_call_turn_messages_is_the_full_generated_delta():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "echo", '{"value": "hi"}')]),
            _final_completion("you said hi"),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run(
        [Message(role="user", content="echo hi")], llm, tools, max_iterations=10
    )

    assert turn.messages == [
        Message(
            role="assistant",
            content=None,
            tool_calls=[_call("call_1", "echo", '{"value": "hi"}')],
        ),
        Message(role="tool", tool_call_id="call_1", name="echo", content="hi"),
        Message(role="assistant", content="you said hi"),
    ]


async def test_run_given_multiple_tool_calls_executes_them_concurrently():
    counter = [0]
    max_seen = [0]
    tools: dict[str, ITool] = {"slow": _ConcurrentTool(counter, max_seen)}
    llm = _FakeLLM(
        [
            _tool_call_completion([_call("call_1", "slow", "{}"), _call("call_2", "slow", "{}")]),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert max_seen[0] == 2


async def test_run_given_multiple_tool_calls_returns_one_result_message_per_call_in_order():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            _tool_call_completion(
                [
                    _call("call_1", "echo", '{"value": "a"}'),
                    _call("call_2", "echo", '{"value": "b"}'),
                ]
            ),
            _final_completion(),
        ]
    )
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_messages = llm.calls[1]["messages"][-2:]
    assert result_messages[0] == Message(
        role="tool", tool_call_id="call_1", name="echo", content="a"
    )
    assert result_messages[1] == Message(
        role="tool", tool_call_id="call_2", name="echo", content="b"
    )


async def test_run_given_tool_call_names_an_unoffered_tool_returns_error_content_and_continues():
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "missing", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, {}, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.tool_call_id == "call_1"
    assert result_message.content == "Error: tool 'missing' was not offered for this call"
    assert turn.message.content == "final answer"


async def test_run_given_tool_raises_returns_error_content_and_continues():
    tools: dict[str, ITool] = {"boom": _RaisingTool()}
    llm = _FakeLLM([_tool_call_completion([_call("call_1", "boom", "{}")]), _final_completion()])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: tool exploded"
    assert turn.message.content == "final answer"


async def test_run_given_malformed_json_arguments_returns_error_content_without_invoking_tool():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "{not json")]), _final_completion()]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content is not None
    assert result_message.content.startswith("Error:")
    assert turn.message.content == "final answer"


async def test_run_given_non_object_json_arguments_returns_error_content_without_invoking_tool():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", "[1, 2]")]), _final_completion()]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    result_message = llm.calls[1]["messages"][-1]
    assert result_message.content == "Error: arguments must be a JSON object"
    assert turn.message.content == "final answer"


async def test_run_given_max_iterations_exhausted_forces_one_final_call_without_tools():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    assert len(llm.calls) == 3
    assert llm.calls[2]["tools"] is None
    assert turn.message.content == "gave up"
    assert turn.finish_reason == "stop"


async def test_run_given_max_iterations_exhausted_turn_messages_includes_every_round():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    always_calls_tool = _tool_call_completion([_call("call_1", "echo", '{"value": "x"}')])
    llm = _FakeLLM([always_calls_tool, always_calls_tool, _final_completion("gave up")])
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=2)

    # 2 rounds x (assistant tool-call + tool result) + 1 forced final answer.
    assert len(turn.messages) == 5
    assert turn.messages[-1] == turn.message


async def test_run_sums_usage_across_every_llm_call():
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    llm = _FakeLLM(
        [
            Completion(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[_call("call_1", "echo", '{"value": "x"}')],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason="tool_calls",
            ),
            Completion(
                message=Message(role="assistant", content="done"),
                usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                finish_reason="stop",
            ),
        ]
    )
    strategy = ReactStrategy()

    turn = await strategy.run([Message(role="user", content="go")], llm, tools, max_iterations=10)

    assert turn.usage == Usage(prompt_tokens=30, completion_tokens=8, total_tokens=38)


async def test_run_forwards_sampling_params_to_every_llm_call():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run(
        [Message(role="user", content="hi")],
        llm,
        {},
        max_iterations=10,
        temperature=0.3,
        top_p=0.8,
        max_tokens=100,
    )

    assert llm.calls[0]["temperature"] == 0.3
    assert llm.calls[0]["top_p"] == 0.8
    assert llm.calls[0]["max_tokens"] == 100


async def test_run_given_no_tools_never_offers_tools_to_llm():
    llm = _FakeLLM([_final_completion()])
    strategy = ReactStrategy()

    await strategy.run([Message(role="user", content="hi")], llm, {}, max_iterations=10)

    assert llm.calls[0]["tools"] is None


async def test_run_given_original_messages_list_is_not_mutated():
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "x"}')]), _final_completion()]
    )
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    strategy = ReactStrategy()
    original = [Message(role="user", content="go")]

    await strategy.run(original, llm, tools, max_iterations=10)

    assert original == [Message(role="user", content="go")]


async def test_run_returned_turn_messages_excludes_the_input_messages():
    llm = _FakeLLM(
        [_tool_call_completion([_call("call_1", "echo", '{"value": "x"}')]), _final_completion()]
    )
    tools: dict[str, ITool] = {"echo": _EchoTool()}
    strategy = ReactStrategy()
    original = [Message(role="user", content="go")]

    turn = await strategy.run(original, llm, tools, max_iterations=10)

    assert original[0] not in turn.messages

"""Compaction: keeps a session's stored history under a configured token budget."""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from agent.core.exceptions import LLMContextWindowExceededError, LLMError
from agent.core.models.config import CompactionConfig
from agent.core.models.message import Message
from agent.core.protocols.illm import ILLM
from agent.core.protocols.isession_store import ISessionStore
from agent.core.registries.llm import LLMRegistry

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "[Summary of earlier conversation]\n"
_ACKNOWLEDGMENT = "Understood — I have that context."


def _user_message_indices(messages: list[Message]) -> list[int]:
    """Indices of every role="user" message in `messages` -- each marks one turn's start."""
    return [i for i, message in enumerate(messages) if message.role == "user"]


def _split_at_turn_boundary(history: list[Message], keep_recent_turns: int) -> int:
    """Split `history` into an old (summarizable) part and a recent (kept verbatim) part.

    The split always lands on a turn boundary (each turn starts at one role="user"
    message), so a tool-call/tool-result pair is never separated. Returns 0 if there are
    `keep_recent_turns` turns or fewer in `history` -- nothing older than the keep-window,
    a no-op signal callers rely on. `keep_recent_turns=0` returns `len(history)` for a
    non-empty `history` (summarize everything, keep nothing) -- the one case with no real
    user-message index to anchor on, since "the 0th-from-the-end turn" doesn't exist.
    """
    user_indices = _user_message_indices(history)
    turn_count = len(user_indices)
    if keep_recent_turns >= turn_count:
        return 0
    if keep_recent_turns == 0:
        return len(history)
    return user_indices[turn_count - keep_recent_turns]


def _chunk_by_turns(messages: list[Message], turns_per_chunk: int) -> list[list[Message]]:
    """Split `messages` into consecutive groups of up to `turns_per_chunk` turns each.

    Each turn starts at one role="user" message, same convention as
    `_split_at_turn_boundary`. Used only by chunked summarization's map step -- unrelated to
    the old/recent split. Empty input returns an empty list. Known gap, not reachable today:
    any messages before the first role="user" message are silently dropped from the first
    chunk -- `messages` is always a prefix of stored history here, which always starts on a
    role="user" message, so this can't currently happen.
    """
    user_indices = _user_message_indices(messages)
    if not user_indices:
        return [messages] if messages else []
    boundaries = user_indices[::turns_per_chunk]
    ends = [*boundaries[1:], len(messages)]
    return [messages[start:end] for start, end in zip(boundaries, ends, strict=True)]


def _is_previous_summary_turn(old: list[Message]) -> bool:
    """True if `old` is exactly the synthetic summary pair a previous `compact()` produced.

    Recognized structurally against our own fixed prefix/acknowledgment text -- there is
    nothing new here to fold in, only our own prior output, so re-running the summarizer
    over it in isolation is a non-retriable case (nothing changed since it was written, so
    there's no real signal a repeat pass would behave differently) rather than a worthwhile
    speculative attempt. Real compaction should update the existing summary with genuinely
    new content once there's some to fold in (a later call whose `old` includes this pair
    *plus* newer turns no longer matches this shape, and proceeds normally) -- not
    recursively compress the summary alone on the hope that it happens to shrink.
    """
    if len(old) != 2:
        return False
    first, second = old
    return (
        first.role == "user"
        and (first.content or "").startswith(_SUMMARY_PREFIX)
        and second.role == "assistant"
        and second.content == _ACKNOWLEDGMENT
    )


@dataclass(frozen=True)
class _SummaryOutcome:
    """The result of one `_try_summarize` attempt.

    `status`: "ok" (`text` holds the usable summary), "llm_error" (the call itself failed --
    not retried again by `_summarize_with_retry`, since `LiteLLMAdapter` already retried
    whatever was retriable one layer down), or "unusable" (the call succeeded but produced
    content not worth keeping -- empty, whitespace-only, or truncated -- still retried once,
    since this is summarization-specific business logic the adapter has no way to know about.
    """

    status: Literal["ok", "llm_error", "unusable"]
    text: str | None = None


class CompactionService:
    """Keeps a session's stored history under its configured token budget.

    Owns the lagged usage estimate as its own private, non-durable state -- not part of
    `ISessionStore` (that's message data only), not a separate cache object either. Lost on
    restart, same posture `InMemorySessionStore` already has.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        session_store: ISessionStore,
        config: CompactionConfig,
        max_sessions: int | None = None,
    ) -> None:
        """Initialize CompactionService.

        Args:
            llm_registry: Registry of available LLM implementations -- used both to resolve
                the agent's own model (for the budget check) and the configured summarizer.
            session_store: Per-conversation message history storage.
            config: Compaction settings (summarizer model, budget, keep-window, prompt).
            max_sessions: Caps how many (agent, session_id) usage estimates are kept at
                once, independently of ISessionStore's own cap on the same value -- see
                `record_usage`. `None` means unbounded.
        """
        self._llm_registry = llm_registry
        self._session_store = session_store
        self._config = config
        self._last_total_tokens: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._max_sessions = max_sessions

    def record_usage(self, agent: str, session_id: str, total_tokens: int) -> None:
        """Record this turn's ending size as the estimate for the next turn's check.

        Bounded by `max_sessions` via LRU eviction, independently of `ISessionStore`'s own
        cap on the same value -- no cross-object coordination needed: a missing estimate
        for a session that still exists just makes `maybe_compact` skip its proactive
        check for one turn (already-existing, best-effort behavior for a session's very
        first turn), and the reactive overflow catch in `AgentRunService.run()` remains
        the real safety net regardless.
        """
        key = (agent, session_id)
        self._last_total_tokens[key] = total_tokens
        self._last_total_tokens.move_to_end(key)
        if self._max_sessions is not None and len(self._last_total_tokens) > self._max_sessions:
            self._last_total_tokens.popitem(last=False)

    def forget(self, agent: str, session_id: str) -> None:
        """Discard the recorded usage estimate for `(agent, session_id)`, if any.

        Called when a session is deleted -- a no-op, not an error, if there was never a
        recorded estimate for it (e.g. a session deleted before its first turn ended).
        """
        self._last_total_tokens.pop((agent, session_id), None)

    async def maybe_compact(self, agent: str, session_id: str, model: str) -> None:
        """Compact now if the last-recorded size is over budget for `model`.

        No-op if there's no recorded usage yet for `(agent, session_id)`, or if `model`'s
        context window isn't known (e.g. a self-hosted or fine-tuned model litellm's static
        data doesn't recognize) -- the reactive fallback in AgentRunService still catches a
        real overflow when one actually happens; this proactive check is best-effort only.
        """
        last = self._last_total_tokens.get((agent, session_id))
        if last is None:
            return
        try:
            max_input_tokens = self._llm_registry.get(model).max_input_tokens()
        except LLMError:
            logger.debug(
                "skipping proactive compaction check for (%s, %s): context window unknown "
                "for model %s",
                agent,
                session_id,
                model,
            )
            return
        budget = max_input_tokens * self._config.token_budget_pct
        if last > budget:
            logger.info(
                "proactive compaction triggered for (%s, %s): %d tokens over budget %.0f",
                agent,
                session_id,
                last,
                budget,
            )
            await self.compact(agent, session_id)
        else:
            logger.debug(
                "checked (%s, %s): %d tokens, still under budget %.0f",
                agent,
                session_id,
                last,
                budget,
            )

    async def compact(self, agent: str, session_id: str) -> bool:
        """Summarize the old portion of `(agent, session_id)`'s history, if there is one.

        Called both proactively (via `maybe_compact`, once the lagged usage estimate is over
        budget) and reactively (by `AgentRunService`, after an `LLMContextWindowExceededError`
        forces compaction regardless of budget) -- same behavior either way. Always uses
        `CompactionConfig.keep_recent_turns` -- the most recent turns it protects are never
        summarized away by any caller, under any circumstance, including on a reactive retry;
        if that isn't enough to fit, the caller's retry fails rather than this function ever
        being asked to discard more than configured.

        The summary is stored as its own synthetic turn -- a `role="user"` message holding
        the summary text, followed by a fixed `role="assistant"` acknowledgment -- prepended
        before whatever turns are kept, regardless of `keep_recent_turns`. Anthropic's own
        docs endorse synthetic assistant messages for exactly this purpose. This keeps the
        stored history starting on `role="user"` (the shape Anthropic requires) and never
        produces two consecutive same-role messages, since the acknowledgment always sits
        between the summary and whatever comes next -- one code path for every
        `keep_recent_turns` value, including 0.

        Every summarizer call -- the single pass, and each chunked-summarization map/reduce
        call -- is retried exactly once with the same input if it fails transiently (any
        `LLMError` other than an overflow, a truncated response, or empty content). A
        summarizer overflow is never retried that way, since the same oversized input would
        fail identically -- the single pass falls back to chunked summarization instead; an
        overflow on a chunk or the reduce call fails that chunked attempt outright (no
        further fallback within a fallback).

        Never lets a failed or truncated summary overwrite good history -- returns False
        and leaves the stored history untouched in every failure case, including any
        summarizer failure (context-window overflow, rate limit, timeout, or any other
        `LLMError`), a failed chunked fallback, an empty summary, or if the resulting summary
        wouldn't actually be smaller than what it's replacing. When `old` is recognizably
        just a previous compaction's own summary turn, with nothing new to fold in, that's
        refused for free before any summarizer call is made at all -- re-running the
        summarizer over our own prior output alone is a non-retriable case (nothing changed
        since it was written), not a worthwhile speculative attempt. This covers both
        compaction re-triggering with no new turns since the last one, and a session whose
        summary alone is still oversized with nothing else left to fall back on -- either
        way, real compaction resumes automatically once genuinely new content is part of
        `old` again, since that no longer matches this shape.

        Holds `self._session_store.lock(agent, session_id)` across the entire method -- the
        get, every summarizer call, and the final replace -- so a concurrent
        `AgentRunService.run()`'s `append()` on the same session can't land in the window
        between this method's read and its write and be silently lost.
        """
        async with self._session_store.lock(agent, session_id):
            history = await self._session_store.get(agent, session_id)
            split = _split_at_turn_boundary(history, self._config.keep_recent_turns)
            if split == 0:
                logger.debug(
                    "nothing old enough to summarize for (%s, %s) given keep_recent_turns=%d",
                    agent,
                    session_id,
                    self._config.keep_recent_turns,
                )
                return False
            old, recent = history[:split], history[split:]
            if _is_previous_summary_turn(old):
                logger.debug(
                    "(%s, %s): old portion is already a summary turn, nothing new to fold in",
                    agent,
                    session_id,
                )
                return False
            summarizer = self._llm_registry.get(self._config.model)
            try:
                summary_content = await self._summarize_with_retry(summarizer, old)
            except LLMContextWindowExceededError:
                logger.warning(
                    "single-pass summary overflowed the summarizer for (%s, %s), "
                    "falling back to chunked map-reduce",
                    agent,
                    session_id,
                )
                summary_content = await self._summarize_chunked(summarizer, old)
            if summary_content is None:
                return False
            summary_text = f"{_SUMMARY_PREFIX}{summary_content}"
            new_history = [
                Message(role="user", content=summary_text),
                Message(role="assistant", content=_ACKNOWLEDGMENT),
                *recent,
            ]
            # Message *count* would hide this: both sides can be 2 messages. Content length
            # is what actually answers "did this shrink anything".
            old_content_chars = sum(len(message.content or "") for message in old)
            new_content_chars = sum(len(message.content or "") for message in new_history[:2])
            if new_content_chars >= old_content_chars:
                logger.warning(
                    "summary for (%s, %s) was not smaller than the original content "
                    "(%d >= %d chars), discarding",
                    agent,
                    session_id,
                    new_content_chars,
                    old_content_chars,
                )
                return False
            # Commits immediately on success -- see RequestTimeoutError's docstring for the
            # case where a timeout in the caller's subsequent LLM call leaves this rewrite
            # committed with no signal back to the caller that it happened.
            await self._session_store.replace(agent, session_id, new_history)
            self._last_total_tokens.pop((agent, session_id), None)
            logger.info(
                "compacted (%s, %s): %d -> %d chars, %d turn(s) kept verbatim",
                agent,
                session_id,
                old_content_chars,
                new_content_chars,
                len(_user_message_indices(recent)),
            )
            return True

    async def _try_summarize(
        self, summarizer: ILLM, messages: list[Message], *, is_retry: bool = False
    ) -> _SummaryOutcome:
        """One summarizer call attempt over `messages` plus the configured prompt.

        The prompt is appended as a new trailing `role="user"` message -- unless `messages`
        already ends on `role="user"` (the chunked reduce step's `combined` input does),
        in which case it's merged into that last message instead, so this never produces
        two consecutive `role="user"` messages regardless of what shape `messages` arrives
        in.

        `messages` is passed through with its tool exchanges intact, even though this call
        never declares `tools` and Bedrock's Converse API rejects toolUse/toolResult blocks in
        a request with no `toolConfig`: `ILLM` implementations are required to fold that
        content out themselves whenever a call declares no tools, so nothing is needed here.

        `is_retry` gates the truncation ERROR log so it fires at most once per
        `_summarize_with_retry` call -- on the first (non-retry) attempt, a truncation is not
        yet a real problem (the retry may well succeed), so it's logged only if it's still
        happening on the retry, the point at which it's an actual, actionable failure.

        Returns a `_SummaryOutcome`: "ok" with the summary text, "llm_error" if the call
        raised any `LLMError` other than a context-window overflow, or "unusable" if the call
        succeeded but produced a truncated response or empty/whitespace content. A
        context-window overflow is not retryable by calling again with the same input -- it
        propagates uncaught so the caller can fall back to chunked summarization instead.
        """
        if messages and messages[-1].role == "user":
            request = [
                *messages[:-1],
                Message(role="user", content=f"{messages[-1].content}\n\n{self._config.prompt}"),
            ]
        else:
            request = [*messages, Message(role="user", content=self._config.prompt)]
        try:
            completion = await summarizer.complete(request)
        except LLMContextWindowExceededError:
            raise
        except LLMError as exc:
            logger.warning(
                "summarizer call failed, giving up (adapter already retried what was "
                "retriable): %s",
                exc,
                extra={"exception_type": type(exc).__name__},
            )
            return _SummaryOutcome(status="llm_error")
        if completion.finish_reason == "length":
            if is_retry:
                logger.error(
                    "summary was truncated (finish_reason=length) for model %s -- "
                    "its own max_tokens is likely too low",
                    self._config.model,
                )
            return _SummaryOutcome(status="unusable")
        content = (completion.message.content or "").strip()
        if not content:
            return _SummaryOutcome(status="unusable")
        return _SummaryOutcome(status="ok", text=completion.message.content)

    async def _summarize_with_retry(self, summarizer: ILLM, messages: list[Message]) -> str | None:
        """`_try_summarize`, retried once if the result was unusable -- not if it raised.

        A raised `LLMError` gives up immediately: `LiteLLMAdapter` already retried whatever
        was retriable one layer down before ever raising, so retrying again here would be
        redundant for a transient failure and pure waste for a permanent one. An *unusable*
        result (empty, whitespace-only, or truncated content) is still retried once -- that's
        summarization-specific business logic the adapter has no way to know about.
        """
        outcome = await self._try_summarize(summarizer, messages)
        if outcome.status == "unusable":
            outcome = await self._try_summarize(summarizer, messages, is_retry=True)
        return outcome.text if outcome.status == "ok" else None

    async def _summarize_chunked(self, summarizer: ILLM, messages: list[Message]) -> str | None:
        """Summarize `messages` in chunks, for when a single pass overflows the summarizer.

        Split into turn groups of `CompactionConfig.chunk_turns` turns each, summarize each
        independently (map), then
        summarize the concatenation of those summaries into one final summary (reduce) --
        the standard map-reduce pattern for long-document summarization. `None` if any
        chunk's summary fails or overflows too, or if `messages` is already too small to
        split further (a single turn alone overflowing the summarizer isn't recoverable by
        chunking).
        """
        chunks = _chunk_by_turns(messages, self._config.chunk_turns)
        if len(chunks) <= 1:
            return None
        chunk_summaries: list[str] = []
        for chunk in chunks:
            try:
                summary = await self._summarize_with_retry(summarizer, chunk)
            except LLMContextWindowExceededError:
                logger.warning(
                    "a chunk's own content still overflowed the summarizer even after chunking"
                )
                return None
            if summary is None:
                return None
            chunk_summaries.append(summary)
        combined_text = "\n\n".join(
            f"Summary of an earlier part of the conversation:\n{s}" for s in chunk_summaries
        )
        combined = [Message(role="user", content=combined_text)]
        try:
            return await self._summarize_with_retry(summarizer, combined)
        except LLMContextWindowExceededError:
            logger.warning("the reduce step (combining chunk summaries) overflowed the summarizer")
            return None

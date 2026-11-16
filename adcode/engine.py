"""The conversation engine, with the behaviour a long session needs.

The engine itself (turn queue, status machine, compaction, sessions,
collaboration) lives in the sdk: ``harness.agent.ConversationEngine``.  This
subclass adds the three things that decide whether a long coding session
survives its interruptions:

- **the reasoning of an unfinished round stays in the history.**  The sdk
  throws the round away when the user interrupts, so the next turn starts
  from a blank slate: the model re-reads the same files and re-plans the same
  work.  The transcript shows the reasoning, the model does not see it.
- **an interrupt reaches the provider.**  The stream stops at the next chunk
  instead of running to the end of the response.
- **``/hold`` keeps a broken request alive.**  A provider error, a dropped
  connection or a response that ends without a finish signal are retried
  until the request works, the user interrupts, or the retries run out.
"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.agent.engine import ConversationEngine

#: What the model sees in front of the reasoning of a round that never
#: finished.  It has to be in the history, not only in the transcript: the
#: model has to know that it was cut off and what it had worked out.
UNFINISHED_NOTE = ('[this turn did not finish - it was interrupted or the '
                   'response was cut off. Continue from here instead of '
                   'starting over.]')


class AdCodeEngine(ConversationEngine):
    """The sdk engine plus the interruption and hold behaviour."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``/hold``: retry a request whose stream broke.
        self.hold_enabled = False
        self.hold_max_retries = 20
        self.hold_delay = 2.0
        # The reasoning of the round that did not finish, if any.
        self._unfinished_round: tuple[str, str] | None = None
        # The reasoning of the round in flight.  The sdk flushes its buffer
        # into the transcript as soon as the text starts, so the reasoning
        # that came first is gone from the request it returns -- and with it
        # from the history.
        self._round_thinking: list[str] = []

    # ── hold ────────────────────────────────────────────────
    def configure_hold(self, enabled: bool, max_retries: int,
                       delay: float) -> None:
        """Apply the ``/hold`` settings."""
        self.hold_enabled = bool(enabled)
        self.hold_max_retries = max(0, int(max_retries))
        self.hold_delay = max(0.0, float(delay))

    def _hold_active(self) -> bool:
        return bool(self.hold_enabled and self.hold_max_retries > 0)

    def _interrupted(self) -> bool:
        return bool(self._interrupt.is_set())

    # ── the request ─────────────────────────────────────────
    async def _stream_request(self, budget: Any
                              ) -> tuple[str, str, list[dict[str, Any]], str]:
        """One model round, with the hold retry loop around it.

        The retry lives here and not in the client: a round that broke may
        already have streamed text, and only the engine knows which parts
        belong to the failed attempt.
        """
        attempt = 0
        while True:
            self._round_thinking = []
            try:
                result = await super()._stream_request(budget)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - the hold mode retries
                if not self._hold_active() or self._interrupted():
                    raise
                attempt += 1
                if attempt > self.hold_max_retries:
                    raise
                self._retry_notice(attempt, str(error))
                await asyncio.sleep(self.hold_delay)
                continue
            thinking, full, tool_calls, finish = result
            thinking = thinking or ''.join(self._round_thinking)
            if self._interrupted():
                # The user stopped the turn: remember what this round had
                # produced.  It is the difference between continuing the work
                # and starting it over.
                if thinking or full:
                    self._unfinished_round = (thinking, full)
                return result
            if finish != 'incomplete':
                # The round finished properly: there is nothing to restore.
                self._unfinished_round = None
                return result
            if thinking:
                self._unfinished_round = (thinking, full)
            # The provider closed the stream without a finish signal: the
            # answer is truncated, not final.
            if not self._hold_active() or self._interrupted():
                return result
            attempt += 1
            if attempt > self.hold_max_retries:
                self.add_notice(
                    f'Gave up after {self.hold_max_retries} '
                    f'retr{"y" if self.hold_max_retries == 1 else "ies"}: '
                    f'the response kept ending early.', 'warning')
                return result
            self._retry_notice(attempt, 'the response ended without a finish '
                                         'signal')
            await asyncio.sleep(self.hold_delay)

    def _thinking_done(self, text: str, started: float) -> None:
        """Remember the reasoning the sdk is about to flush.

        The sdk moves the reasoning into the transcript when the answer text
        starts and empties its buffer, so the request it returns no longer
        carries it.  The history needs it.
        """
        if text:
            self._round_thinking.append(text)
        super()._thinking_done(text, started)

    def _retry_notice(self, attempt: int, reason: str) -> None:
        self._set_status('connecting', retrying=True)
        self.add_notice(f'Retrying ({attempt}/{self.hold_max_retries}): '
                        f'{reason[:160]}', 'warning')

    # ── the unfinished round ────────────────────────────────
    async def _run_chat(self, text: str) -> None:
        """Run a turn, keeping what an unfinished round had produced."""
        self._unfinished_round = None
        await super()._run_chat(text)
        self._fill_unfinished_reasoning()

    def _commit_partial(self) -> bool:
        """Commit the part of a turn that is worth keeping.

        The sdk keeps the tool rounds that finished and drops everything
        else -- including the reasoning the model had just streamed.  That
        reasoning is the model's own plan; dropping it makes the next turn
        start over, which is how a session ends up in circles.
        """
        committed = super()._commit_partial()
        unfinished = self._unfinished_round
        if unfinished is not None:
            thinking, full = unfinished
            if thinking or full:
                self.conversation.stage(self._unfinished_message(thinking, full))
                self.conversation.commit()
                committed = True
        if not committed and self._interrupted():
            # Nothing was produced, but the request itself is part of the
            # context: the next turn has to answer it.
            draft = self.conversation.draft
            if draft and draft[0].get('role') == 'user':
                self.conversation.commit()
                committed = True
        return committed

    @staticmethod
    def _unfinished_message(thinking: str, full: str) -> dict[str, Any]:
        """The assistant turn that carries an unfinished round's reasoning.

        The reasoning goes into the text as well: the Responses and the
        Anthropic wire have no field for it, and a turn that says nothing
        is a turn the model cannot continue.
        """
        parts = [UNFINISHED_NOTE]
        if full:
            parts.append(full)
        if thinking:
            parts.append(thinking)
        return {'role': 'assistant',
                'content': '\n'.join(parts),
                'reasoning_content': thinking}

    def _fill_unfinished_reasoning(self) -> None:
        """Give a truncated turn its reasoning back.

        The sdk commits an empty assistant message when the response ends
        without a finish signal.  The model then sees a turn that said
        nothing, forgets the plan it had just made and starts over.
        """
        unfinished = self._unfinished_round
        self._unfinished_round = None
        if not unfinished:
            return
        thinking, full = unfinished
        if not thinking or full:
            return
        committed = self.conversation.committed
        if not committed:
            return
        last = committed[-1]
        if last.get('role') != 'assistant' or last.get('content'):
            return
        # The reasoning is only in ``reasoning_content`` here, and the
        # Responses wire has no field for it: as text it reaches every wire.
        last['content'] = f'{UNFINISHED_NOTE}\n{thinking}'
        self.conversation.invalidate()

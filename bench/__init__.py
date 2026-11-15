"""Headless simulation harness for AdCode.

Public surface:

* :class:`~bench.app.AppHarness` -- the real app on a thread, with fake
  terminal screen/input and a temporary config/workspace.
* :class:`~bench.gateway.FakeLLMGateway` -- a real local HTTP provider
  speaking the OpenAI Chat Completions, OpenAI Responses and Anthropic Messages
  wire protocols.
* :class:`~bench.script.Turn` / ``ToolCall`` / responders -- the model script.
* :class:`~bench.artifacts.ArtifactWriter` -- screenshots and run data.

Run ``python -m bench list`` or ``python -m bench run chat`` for the
manual, screenshot-producing scenarios.
"""

from bench.app import AppHarness, HarnessError
from bench.artifacts import ArtifactWriter
from bench.gateway import FakeGateway, FakeLLMGateway
from bench.screen import FakeScreen, FrameSnapshot
from bench.script import (
    CallbackResponder, EchoResponder, ModelRequest, ScriptedResponder,
    ToolCall, Turn, TurnUsage,
)

__all__ = [
    'AppHarness', 'HarnessError', 'ArtifactWriter', 'FakeGateway',
    'FakeLLMGateway', 'FakeScreen', 'FrameSnapshot', 'CallbackResponder',
    'EchoResponder', 'ModelRequest', 'ScriptedResponder', 'ToolCall', 'Turn',
    'TurnUsage',
]

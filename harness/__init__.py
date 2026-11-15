"""Headless simulation harness for AdCode.

Public surface:

* :class:`~harness.app.AppHarness` -- the real app on a thread, with fake
  terminal screen/input and a temporary config/workspace.
* :class:`~harness.gateway.FakeLLMGateway` -- a real local HTTP provider
  speaking the OpenAI Chat Completions, OpenAI Responses and Anthropic Messages
  wire protocols.
* :class:`~harness.script.Turn` / ``ToolCall`` / responders -- the model script.
* :class:`~harness.artifacts.ArtifactWriter` -- screenshots and run data.

Run ``python -m harness list`` or ``python -m harness run chat`` for the
manual, screenshot-producing scenarios.
"""

from harness.app import AppHarness, HarnessError
from harness.artifacts import ArtifactWriter
from harness.gateway import FakeGateway, FakeLLMGateway
from harness.screen import FakeScreen, FrameSnapshot
from harness.script import (
    CallbackResponder, EchoResponder, ModelRequest, ScriptedResponder,
    ToolCall, Turn, TurnUsage,
)

__all__ = [
    'AppHarness', 'HarnessError', 'ArtifactWriter', 'FakeGateway',
    'FakeLLMGateway', 'FakeScreen', 'FrameSnapshot', 'CallbackResponder',
    'EchoResponder', 'ModelRequest', 'ScriptedResponder', 'ToolCall', 'Turn',
    'TurnUsage',
]

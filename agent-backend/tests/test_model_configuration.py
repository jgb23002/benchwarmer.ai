"""Ensure model configuration reaches every agent without using live API keys."""

from unittest.mock import MagicMock

import pytest

from benchwarmer.agents.backends import (
    ClaudeBackend, DEFAULT_CLAUDE_MODEL, DEFAULT_IMPLEMENTATION_MODEL,
)
from benchwarmer.agents.implementation import ImplementationAgent
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.plot import PlotAgent


@pytest.fixture(autouse=True)
def isolated_models(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_IMPLEMENTATION_MODEL", raising=False)
    monkeypatch.setattr("anthropic.Anthropic", MagicMock())


def selected_model(agent):
    return agent.backend.model if isinstance(agent, IntakeAgent) else agent.model


@pytest.mark.parametrize("factory", [ClaudeBackend, IntakeAgent, PlotAgent])
def test_chat_model_environment_setting_reaches_each_agent(factory, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "configured-chat-model")
    assert selected_model(factory()) == "configured-chat-model"


@pytest.mark.parametrize("factory", [ClaudeBackend, IntakeAgent, PlotAgent, ImplementationAgent])
def test_explicit_model_takes_precedence_over_environment(factory, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "configured-chat-model")
    monkeypatch.setenv("ANTHROPIC_IMPLEMENTATION_MODEL", "configured-code-model")
    assert selected_model(factory(model="explicit-model")) == "explicit-model"


def test_defaults_use_the_supported_model_generation():
    for factory in [ClaudeBackend, IntakeAgent, PlotAgent]:
        assert selected_model(factory()) == DEFAULT_CLAUDE_MODEL == "claude-sonnet-4-6"
    assert selected_model(ImplementationAgent()) == DEFAULT_IMPLEMENTATION_MODEL


def test_blank_environment_settings_keep_working_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_IMPLEMENTATION_MODEL", "")
    assert selected_model(IntakeAgent()) == DEFAULT_CLAUDE_MODEL
    assert selected_model(PlotAgent()) == DEFAULT_CLAUDE_MODEL
    assert selected_model(ImplementationAgent()) == DEFAULT_IMPLEMENTATION_MODEL


def test_code_model_is_configured_independently(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "configured-chat-model")
    monkeypatch.setenv("ANTHROPIC_IMPLEMENTATION_MODEL", "configured-code-model")
    assert selected_model(ImplementationAgent()) == "configured-code-model"
    assert selected_model(IntakeAgent()) == "configured-chat-model"

from __future__ import annotations

import pytest


def test_openai_compatible_provider_uses_local_endpoint_and_key(monkeypatch):
    from src.support_agent import agent

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools):
            captured["tools"] = tools
            return self

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "local-test-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "example/tool-model")

    model = agent.build_chat_model()

    assert isinstance(model, FakeChatOpenAI)
    assert captured["api_key"] == "local-test-key"
    assert captured["base_url"] == "https://example.invalid/v1"
    assert captured["model"] == "example/tool-model"
    assert "temperature" not in captured
    assert captured["tools"] == agent.tools


def test_openai_compatible_provider_requires_key_without_making_request(monkeypatch):
    from src.support_agent import agent

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://example.invalid/v1")

    with pytest.raises(RuntimeError, match="OPENAI_COMPATIBLE_API_KEY"):
        agent.build_chat_model()


def test_ollama_remains_selectable(monkeypatch):
    from src.support_agent import agent

    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools):
            captured["tools"] = tools
            return self

    monkeypatch.setattr(agent, "ChatOllama", FakeChatOllama)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("MODEL_NAME", "llama3.1:latest")

    model = agent.build_chat_model()

    assert isinstance(model, FakeChatOllama)
    assert captured["base_url"] == "http://localhost:11434"
    assert captured["model"] == "llama3.1:latest"
    assert captured["tools"] == agent.tools

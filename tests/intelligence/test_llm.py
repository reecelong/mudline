"""Unit tests for LLM provider abstraction.

Tests verify:
- Message, ToolDef, ToolCall, LLMResponse dataclasses
- AnthropicProvider async complete() with mocked API
- OllamaProvider async complete() with mocked HTTP
- Error handling and edge cases
- create_provider factory function
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types.content_block import TextBlock, ToolUseBlock

from mudline.exceptions import LLMError
from mudline.intelligence.llm import (
    AnthropicProvider,
    LLMResponse,
    Message,
    OllamaProvider,
    ToolCall,
    ToolDef,
    VertexAIProvider,
    create_provider,
)


class TestDataclasses:
    """Test message, tool, and response dataclasses."""

    def test_message_creation(self) -> None:
        """Test Message dataclass."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_frozen(self) -> None:
        """Test that Message is immutable."""
        msg = Message(role="user", content="Hello")
        with pytest.raises(AttributeError):
            msg.content = "Goodbye"  # type: ignore

    def test_tool_def_creation(self) -> None:
        """Test ToolDef dataclass."""
        tool = ToolDef(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "properties" in tool.parameters

    def test_tool_call_creation(self) -> None:
        """Test ToolCall dataclass."""
        call = ToolCall(id="123", name="search", arguments={"query": "test"})
        assert call.id == "123"
        assert call.name == "search"
        assert call.arguments == {"query": "test"}

    def test_llm_response_creation(self) -> None:
        """Test LLMResponse dataclass."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            tool_calls=[],
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.tool_calls == []
        assert response.usage["input_tokens"] == 10

    def test_llm_response_default_fields(self) -> None:
        """Test LLMResponse with default fields."""
        response = LLMResponse(content="Test", model="test-model")
        assert response.content == "Test"
        assert response.tool_calls == []
        assert response.usage == {}


class TestAnthropicProvider:
    """Test AnthropicProvider with mocked Anthropic SDK."""

    def test_init_with_explicit_api_key(self) -> None:
        """Test initialization with explicit API key."""
        provider = AnthropicProvider(api_key="test-key-123", model="claude-opus")
        assert provider.model == "claude-opus"

    def test_init_with_env_api_key(self) -> None:
        """Test initialization reading API key from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-456"}):
            provider = AnthropicProvider()
            assert provider.model == "claude-sonnet-4-20250514"

    def test_init_missing_api_key(self) -> None:
        """Test initialization fails when API key is missing."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(LLMError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicProvider()

    @pytest.mark.asyncio
    async def test_complete_text_only(self) -> None:
        """Test completion with text-only response."""
        provider = AnthropicProvider(api_key="test-key")

        # Mock the Anthropic API
        mock_response = MagicMock()
        mock_response.content = [TextBlock(type="text", text="Hello!")]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hello!"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.tool_calls == []
        assert response.usage["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self) -> None:
        """Test completion with tool calls."""
        provider = AnthropicProvider(api_key="test-key")

        # Mock response with text + tool use
        text_block = TextBlock(type="text", text="I'll search for that.")
        tool_block = ToolUseBlock(
            type="tool_use",
            id="call-1",
            name="search",
            input={"query": "test query"},
        )

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 15

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        tools = [
            ToolDef(
                name="search",
                description="Search",
                parameters={"type": "object"},
            )
        ]
        messages = [Message(role="user", content="Search for info")]
        response = await provider.complete(messages, tools)

        assert response.content == "I'll search for that."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "test query"}

    @pytest.mark.asyncio
    async def test_complete_api_error(self) -> None:
        """Test handling of API errors."""
        provider = AnthropicProvider(api_key="test-key")

        # Mock an API error
        provider.client.messages.create = MagicMock(side_effect=Exception("API rate limited"))

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(LLMError, match="Anthropic API failed"):
            await provider.complete(messages)


class TestOllamaProvider:
    """Test OllamaProvider with mocked HTTP client."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434/v1"
        assert provider.model == "llama3.2"

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        provider = OllamaProvider(
            base_url="http://example.com/api",
            model="mistral",
        )
        assert provider.base_url == "http://example.com/api"
        assert provider.model == "mistral"

    @pytest.mark.asyncio
    async def test_complete_text_only(self) -> None:
        """Test completion with text-only response."""
        provider = OllamaProvider()

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from Ollama!",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        provider.client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hello from Ollama!"
        assert response.model == "llama3.2"
        assert response.tool_calls == []
        assert response.usage["input_tokens"] == 10

        # Verify the HTTP call
        provider.client.post.assert_called_once()
        call_args = provider.client.post.call_args
        assert "chat/completions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self) -> None:
        """Test completion with tool calls in response."""
        provider = OllamaProvider()

        # Mock HTTP response with tool calls
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll search for that.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "test query"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }

        provider.client.post = AsyncMock(return_value=mock_response)

        tools = [
            ToolDef(
                name="search",
                description="Search",
                parameters={"type": "object"},
            )
        ]
        messages = [Message(role="user", content="Search for info")]
        response = await provider.complete(messages, tools)

        assert response.content == "I'll search for that."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_complete_empty_response(self) -> None:
        """Test handling of empty response."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}

        provider.client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        response = await provider.complete(messages)

        assert response.content == ""
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_complete_http_error(self) -> None:
        """Test handling of HTTP errors."""
        provider = OllamaProvider()

        import httpx

        provider.client.post = AsyncMock(side_effect=httpx.RequestError("Connection refused"))

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(LLMError, match="Ollama service unavailable"):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_complete_invalid_json(self) -> None:
        """Test handling of invalid JSON in response."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        provider.client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(LLMError, match="Ollama response parsing failed"):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test context manager for async client cleanup."""
        provider = OllamaProvider()
        provider.client.aclose = AsyncMock()

        async with provider:
            pass

        provider.client.aclose.assert_called_once()


class TestVertexAIProvider:
    """Test the Vertex AI (Gemini) provider."""

    @staticmethod
    def _make_provider() -> VertexAIProvider:
        """Construct a provider with the genai client patched out (no ADC needed)."""
        with patch("google.genai.Client", return_value=MagicMock()):
            return VertexAIProvider(project="test-proj", location="us-central1")

    def test_init_uses_env_and_vertex_mode(self) -> None:
        """Project/location fall back to env; the client runs in Vertex mode."""
        with (
            patch("google.genai.Client") as mock_client_cls,
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-proj"}, clear=True),
        ):
            VertexAIProvider()
            _, kwargs = mock_client_cls.call_args
            assert kwargs["vertexai"] is True
            assert kwargs["project"] == "env-proj"
            assert kwargs["location"] == "us-central1"

    def test_missing_extra_raises_llm_error(self) -> None:
        """A clear LLMError is raised when the google-genai SDK is unavailable."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            fromlist = args[2] if len(args) > 2 else (kwargs.get("fromlist") or ())
            if name == "google.genai" or (name == "google" and "genai" in fromlist):
                raise ImportError("simulated missing google-genai")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=fake_import),
            pytest.raises(LLMError, match="vertex"),
        ):
            VertexAIProvider()

    @pytest.mark.asyncio
    async def test_complete_extracts_text_and_tool_calls(self) -> None:
        """Text + function calls are extracted; system msg -> system_instruction."""
        provider = self._make_provider()

        text_part = MagicMock()
        text_part.text = "Here is the answer."
        text_part.function_call = None

        func_call = MagicMock()
        func_call.id = None
        func_call.name = "search_messages"
        func_call.args = {"query": "plumber"}
        call_part = MagicMock()
        call_part.text = None
        call_part.function_call = func_call

        candidate = MagicMock()
        candidate.content.parts = [text_part, call_part]

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata.prompt_token_count = 12
        response.usage_metadata.candidates_token_count = 8

        provider.client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await provider.complete(
            [
                Message(role="system", content="be helpful"),
                Message(role="user", content="who fixed the sink?"),
            ],
            tools=[
                ToolDef(
                    name="search_messages",
                    description="search messages",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ],
        )

        assert result.content == "Here is the answer."
        assert result.model == "gemini-2.5-pro"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_messages"
        assert result.tool_calls[0].arguments == {"query": "plumber"}
        assert result.tool_calls[0].id == "vertex-0"
        assert result.usage == {"input_tokens": 12, "output_tokens": 8}

        _, kwargs = provider.client.aio.models.generate_content.call_args
        assert kwargs["config"].system_instruction == "be helpful"
        # The system message is not part of the conversation contents.
        assert len(kwargs["contents"]) == 1
        assert kwargs["contents"][0].role == "user"

    @pytest.mark.asyncio
    async def test_complete_wraps_errors_in_llm_error(self) -> None:
        """Underlying SDK errors are surfaced as LLMError."""
        provider = self._make_provider()
        provider.client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(LLMError, match="Vertex AI failed"):
            await provider.complete([Message(role="user", content="hi")])


class TestFactoryFunction:
    """Test create_provider factory function."""

    def test_create_anthropic_provider(self) -> None:
        """Test creating Anthropic provider via factory."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_provider("anthropic", model="claude-opus")
            assert isinstance(provider, AnthropicProvider)
            assert provider.model == "claude-opus"

    def test_create_ollama_provider(self) -> None:
        """Test creating Ollama provider via factory."""
        provider = create_provider("ollama", base_url="http://custom:8000", model="custom-model")
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://custom:8000"
        assert provider.model == "custom-model"

    def test_create_vertex_provider(self) -> None:
        """Test creating the Vertex provider via factory, including the alias."""
        with patch("google.genai.Client", return_value=MagicMock()):
            provider = create_provider("vertex", project="p", location="us-central1")
            assert isinstance(provider, VertexAIProvider)
            assert provider.model == "gemini-2.5-pro"

            aliased = create_provider("gemini", model="gemini-2.5-flash")
            assert isinstance(aliased, VertexAIProvider)
            assert aliased.model == "gemini-2.5-flash"

    def test_create_unknown_provider(self) -> None:
        """Test factory fails for unknown provider."""
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            create_provider("unknown-provider")

    def test_create_with_defaults(self) -> None:
        """Test factory with default arguments."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            anthropic_provider = create_provider("anthropic")
            assert anthropic_provider.model == "claude-sonnet-4-20250514"

            ollama_provider = create_provider("ollama")
            assert ollama_provider.model == "llama3.2"
            assert ollama_provider.base_url == "http://localhost:11434/v1"


class TestProtocolCompliance:
    """Test that providers implement the LLMProvider protocol."""

    @pytest.mark.asyncio
    async def test_anthropic_protocol_compliance(self) -> None:
        """Test AnthropicProvider has required protocol methods."""
        provider = AnthropicProvider(api_key="test")
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

    @pytest.mark.asyncio
    async def test_ollama_protocol_compliance(self) -> None:
        """Test OllamaProvider has required protocol methods."""
        provider = OllamaProvider()
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

    @pytest.mark.asyncio
    async def test_vertex_protocol_compliance(self) -> None:
        """Test VertexAIProvider has required protocol methods."""
        with patch("google.genai.Client", return_value=MagicMock()):
            provider = VertexAIProvider(project="p")
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

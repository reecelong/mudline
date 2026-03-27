"""LLM provider abstraction — unified interface for multiple LLM backends.

Supports:
- Anthropic Claude via the anthropic SDK
- OpenAI-compatible services (Ollama) via httpx

This module provides a Protocol-based abstraction so the intelligence layer
can work with any LLM backend without coupling to a specific provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import anthropic
import httpx
from anthropic import AsyncAnthropic

from mudline.exceptions import LLMError

logger = logging.getLogger(__name__)


async def _retry_with_backoff(
    fn: Any,
    *,
    is_retryable: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    operation_name: str = "LLM call",
) -> Any:
    """Retry an async callable with exponential backoff on transient errors.

    Args:
        fn: Zero-argument async callable to invoke.
        is_retryable: Predicate that takes an exception and returns True if
            the call should be retried.
        max_retries: Maximum number of retry attempts (total calls = max_retries + 1).
        base_delay: Base delay in seconds; doubles each retry (1s, 2s, 4s, ...).
        operation_name: Label for log messages.

    Returns:
        The return value of *fn* on success.

    Raises:
        The original exception after all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                operation_name,
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    # Unreachable, but satisfies the type checker.
    raise last_exc  # type: ignore[misc]


def _is_anthropic_retryable(exc: Exception) -> bool:
    """Return True if an Anthropic SDK exception is transient and worth retrying."""
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _is_ollama_retryable(exc: Exception) -> bool:
    """Return True if an Ollama/httpx exception is transient and worth retrying."""
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


def _is_claude_code_retryable(exc: Exception) -> bool:
    """Return True if a ClaudeCode CLI error is transient and worth retrying."""
    if isinstance(exc, LLMError):
        msg = str(exc)
        if "not found" in msg:
            return False
        if re.search(r"exit \d+", msg):
            return True
    return False


@dataclass(frozen=True)
class Message:
    """A single message in a conversation.

    Args:
        role: "user", "assistant", or "system".
        content: The message text.
    """

    role: str
    content: str


@dataclass(frozen=True)
class ToolDef:
    """Definition of a tool the LLM can call.

    Args:
        name: The tool identifier.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation made by the LLM.

    Args:
        id: Unique identifier for this call (used in follow-up messages).
        name: The tool name.
        arguments: Parsed arguments dict (not a JSON string).
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Args:
        content: The text response from the model.
        tool_calls: List of tool invocations, if any.
        model: The model identifier that produced this response.
        usage: Token usage info (provider-specific).
    """

    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    All providers must implement async complete() for unified message handling.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: Conversation history.
            tools: Optional list of tools the LLM can call.

        Returns:
            LLMResponse with the model's output and any tool calls.

        Raises:
            LLMError: If the API call fails or times out.
        """
        ...


class AnthropicProvider:
    """Anthropic Claude provider via anthropic SDK.

    Uses claude-sonnet-4-20250514 by default.
    API key is read from ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514") -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: API key for Anthropic. If None, reads from ANTHROPIC_API_KEY.
            model: Model name. Defaults to claude-sonnet-4-20250514.
        """
        self.model = model
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY environment variable not set")
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Generate a completion using Claude.

        Args:
            messages: Conversation history.
            tools: Optional list of tools Claude can call.

        Returns:
            LLMResponse with Claude's output.

        Raises:
            LLMError: If the API call fails.
        """
        async def _call() -> LLMResponse:
            # Separate system messages from user/assistant messages
            system_parts = [msg.content for msg in messages if msg.role == "system"]
            system_text = "\n".join(system_parts) if system_parts else None
            api_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
                if msg.role != "system"
            ]

            # Convert ToolDef to Anthropic tool format
            api_tools = None
            if tools:
                api_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    }
                    for tool in tools
                ]

            # Build kwargs for Anthropic API call
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": api_messages,
            }
            if system_text:
                create_kwargs["system"] = system_text
            if api_tools:
                create_kwargs["tools"] = api_tools

            # Call Anthropic API
            response = await self.client.messages.create(**create_kwargs)

            # Extract content and tool calls from response
            content = ""
            tool_calls: list[ToolCall] = []

            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            # Extract usage info
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            logger.debug(
                f"Anthropic completion: model={self.model}, "
                f"input_tokens={usage['input_tokens']}, "
                f"output_tokens={usage['output_tokens']}"
            )

            return LLMResponse(
                content=content,
                model=self.model,
                tool_calls=tool_calls,
                usage=usage,
            )

        try:
            return await _retry_with_backoff(
                _call,
                is_retryable=_is_anthropic_retryable,
                operation_name="Anthropic API",
            )
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(f"Anthropic API failed: {e}") from e


class OllamaProvider:
    """OpenAI-compatible provider for Ollama or similar local LLM services.

    Communicates via HTTP using the OpenAI API format.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL for the OpenAI-compatible service.
            model: Model name to use. Defaults to llama3.2.
        """
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Generate a completion using Ollama (or compatible service).

        Args:
            messages: Conversation history.
            tools: Optional list of tools the service can call.

        Returns:
            LLMResponse with the service's output.

        Raises:
            LLMError: If the HTTP request fails.
        """
        async def _call() -> LLMResponse:
            # Convert Message dataclasses to dict format
            api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Build request body
            request_body: dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
            }

            # Add tools if provided (may not be supported by all services)
            if tools:
                request_body["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ]

            # Call the service
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
            )

            response.raise_for_status()
            data = response.json()

            # Extract content and tool calls from response
            content = ""
            tool_calls: list[ToolCall] = []

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    message = choice["message"]
                    if "content" in message and message["content"]:
                        content = message["content"]

                    # Handle tool calls if present
                    if "tool_calls" in message and message["tool_calls"]:
                        for tool_call in message["tool_calls"]:
                            if tool_call.get("type") == "function":
                                func = tool_call.get("function", {})
                                raw_args = func.get("arguments", {})
                                if isinstance(raw_args, str):
                                    try:
                                        raw_args = json.loads(raw_args)
                                    except (json.JSONDecodeError, TypeError):
                                        raw_args = {}
                                tool_calls.append(
                                    ToolCall(
                                        id=tool_call.get("id", ""),
                                        name=func.get("name", ""),
                                        arguments=raw_args,
                                    )
                                )

            # Extract usage info if available
            usage = {}
            if "usage" in data:
                usage = {
                    "input_tokens": data["usage"].get("prompt_tokens", 0),
                    "output_tokens": data["usage"].get("completion_tokens", 0),
                }

            logger.debug(
                f"Ollama completion: model={self.model}, "
                f"base_url={self.base_url}, "
                f"content_length={len(content)}"
            )

            return LLMResponse(
                content=content,
                model=self.model,
                tool_calls=tool_calls,
                usage=usage,
            )

        try:
            return await _retry_with_backoff(
                _call,
                is_retryable=_is_ollama_retryable,
                operation_name="Ollama API",
            )
        except httpx.RequestError as e:
            logger.error(f"Ollama HTTP request failed: {e}")
            raise LLMError(f"Ollama service unavailable: {e}") from e
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            logger.error(f"Ollama response parsing error: {e}")
            raise LLMError(f"Ollama response parsing failed: {e}") from e

    async def __aenter__(self) -> OllamaProvider:
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — clean up HTTP client."""
        await self.client.aclose()


class ClaudeCodeProvider:
    """LLM provider that uses the Claude Code CLI in headless mode.

    Requires `claude` to be installed and authenticated. Uses the user's
    existing Claude Code subscription — no separate API key needed.
    """

    def __init__(self, model: str = "sonnet") -> None:
        """Initialize the Claude Code provider.

        Args:
            model: Model hint for claude CLI ("sonnet", "opus", "haiku").
        """
        self.model = model

    @staticmethod
    def _extract_tool_calls(text: str) -> tuple[list[ToolCall], str]:
        """Extract tool call JSON objects from free-form response text.

        Scans for JSON objects matching {"tool": "...", "arguments": {...}}
        embedded in the response. This is best-effort — the model may
        format calls slightly differently, but the regex covers the
        common case from our prompt instructions.

        Args:
            text: Raw response text from the CLI.

        Returns:
            Tuple of (parsed ToolCall list, cleaned text with JSON removed).
        """
        # Match JSON-like blocks: outermost { ... } allowing nested braces
        json_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
        tool_calls: list[ToolCall] = []
        spans_to_remove: list[tuple[int, int]] = []

        for match in json_pattern.finditer(text):
            try:
                obj = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                continue

            if not isinstance(obj, dict):
                continue
            if "tool" not in obj or "arguments" not in obj:
                continue
            if not isinstance(obj["arguments"], dict):
                continue

            tool_calls.append(
                ToolCall(
                    id=f"cc-{len(tool_calls)}",
                    name=obj["tool"],
                    arguments=obj["arguments"],
                )
            )
            spans_to_remove.append(match.span())

        # Strip matched JSON blocks from the content (reverse order to keep indices valid)
        cleaned = text
        for start, end in reversed(spans_to_remove):
            cleaned = cleaned[:start] + cleaned[end:]
        cleaned = cleaned.strip()

        return tool_calls, cleaned

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> LLMResponse:
        """Generate a completion by shelling out to claude CLI.

        Tool descriptions are appended to the prompt so the model can respond
        with JSON tool calls. The response text is then scanned for those
        JSON objects and converted into proper ToolCall instances.

        Args:
            messages: Conversation history.
            tools: Optional list of tools the model can call via text.

        Returns:
            LLMResponse with the CLI output and any parsed tool calls.

        Raises:
            LLMError: If the CLI invocation fails.
        """
        import shutil

        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise LLMError("claude CLI not found — install Claude Code first")

        # Build a single prompt from all messages
        prompt_parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"[System]\n{msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"[User]\n{msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"[Assistant]\n{msg.content}")
        prompt = "\n\n".join(prompt_parts)

        # If tools are provided, append their descriptions so the model
        # knows what's available (it can describe calls in text)
        if tools:
            tool_desc = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            )
            prompt += (
                "\n\n[Available tools — respond with JSON tool calls "
                "in the format {\"tool\": \"name\", \"arguments\": {...}} "
                "if you need to use them]\n" + tool_desc
            )

        async def _call() -> LLMResponse:
            cmd = [
                claude_bin, "-p", prompt,
                "--output-format", "json",
                "--model", self.model,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120.0
            )

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else "unknown error"
                raise LLMError(f"claude CLI failed (exit {proc.returncode}): {err}")

            data = json.loads(stdout.decode())
            content = data.get("result", "")
            usage_data = data.get("usage", {})

            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
            }

            # Parse tool calls from the response text if tools were provided
            tool_calls: list[ToolCall] = []
            if tools and content:
                tool_calls, content = self._extract_tool_calls(content)

            logger.debug(
                "Claude Code completion: model=%s, input=%d, output=%d, tool_calls=%d",
                self.model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                len(tool_calls),
            )

            return LLMResponse(
                content=content,
                model=f"claude-code:{self.model}",
                tool_calls=tool_calls,
                usage=usage,
            )

        try:
            return await _retry_with_backoff(
                _call,
                is_retryable=_is_claude_code_retryable,
                operation_name="Claude Code CLI",
            )
        except LLMError:
            raise
        except TimeoutError as e:
            raise LLMError("claude CLI timed out after 120s") from e
        except Exception as e:
            logger.error("Claude Code CLI error: %s", e)
            raise LLMError(f"Claude Code CLI failed: {e}") from e


def create_provider(
    provider_type: str,
    **kwargs: Any,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_type: "anthropic", "ollama", or "claude-code".
        **kwargs: Provider-specific arguments.
                  - anthropic: api_key (optional), model (optional)
                  - ollama: base_url (optional), model (optional)
                  - claude-code: model (optional, default "sonnet")

    Returns:
        An LLMProvider instance.

    Raises:
        LLMError: If provider_type is unknown.
    """
    if provider_type == "anthropic":
        return AnthropicProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "claude-sonnet-4-20250514"),
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434/v1"),
            model=kwargs.get("model", "llama3.2"),
        )
    elif provider_type == "claude-code":
        return ClaudeCodeProvider(
            model=kwargs.get("model", "sonnet"),
        )
    else:
        raise LLMError(f"Unknown LLM provider: {provider_type}")

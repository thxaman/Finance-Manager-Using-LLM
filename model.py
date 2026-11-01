"""Groq-backed finance assistant.

The assistant uses Groq's official Python SDK directly. The model receives a
computed dataset brief and a bounded chat history, so CSV questions stay safe
(no model-generated Python is executed). Keeping the provider integration thin
also avoids private framework APIs that can disappear between releases.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq

from config import (
    DEFAULT_CURRENCY,
    build_system_prompt,
    get_settings,
)
from data import (
    FinancialSummary,
    build_dataset_brief,
    load_csv,
    summarize,
)

logger = logging.getLogger(__name__)
_MAX_MESSAGE_CHARS = 8_000
_MAX_OUTPUT_TOKENS = 4_096
_REQUEST_TIMEOUT = 60.0


class MissingAPIKeyError(RuntimeError):
    """Raised when the Groq API key is not configured."""


@dataclass(frozen=True)
class _ChatReply:
    """Small response object shared by the real client and test doubles."""

    content: str


def _content_to_text(content: Any) -> str:
    """Normalise message content (string or content blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text is None:
                    text = block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


class _GroqChat:
    """Minimal chat interface over the official Groq SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_retries: int = 2,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = Groq(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _create(self, messages: Sequence[dict[str, str]], *, stream: bool):
        return self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=_MAX_OUTPUT_TOKENS,
            stream=stream,
        )

    def invoke(self, messages: Sequence[dict[str, str]]) -> _ChatReply:
        response = self._create(messages, stream=False)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return _ChatReply("")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return _ChatReply(_content_to_text(content))

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[_ChatReply]:
        response = self._create(messages, stream=True)
        try:
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                content = getattr(getattr(choices[0], "delta", None), "content", None)
                if content:
                    yield _ChatReply(_content_to_text(content))
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


class FinanceAssistant:
    """Stateful finance assistant with optional CSV context."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        max_history_messages: int | None = None,
        currency_symbol: str | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model or settings.model
        self.temperature = settings.temperature if temperature is None else temperature
        self.currency_symbol = currency_symbol or settings.currency_symbol or DEFAULT_CURRENCY
        self.max_history_messages = max_history_messages or settings.max_history_messages
        self.max_context_chars = max_context_chars or settings.max_context_chars
        self._api_key = api_key or settings.api_key

        self._llm: _GroqChat | None = None
        self._history: list[dict[str, str]] = []
        self._dataframe = None
        self._summary: FinancialSummary | None = None
        self._brief: str | None = None
        self._dataset_name: str | None = None

    # ------------------------------------------------------------------ data
    @property
    def has_dataset(self) -> bool:
        return self._dataframe is not None

    @property
    def dataset_name(self) -> str | None:
        return self._dataset_name

    @property
    def summary(self) -> FinancialSummary | None:
        return self._summary

    @property
    def dataframe(self):
        return self._dataframe

    def read_csv(self, source, *, name: str | None = None):
        """Load a CSV from a path/file object, attach it, and return (df, warnings)."""
        source_name = name
        if source_name is None:
            if isinstance(source, (str, Path)):
                source_name = Path(source).name
            else:
                source_name = getattr(source, "name", None) or "uploaded CSV"
        df, warnings = load_csv(source)
        self.set_dataframe(df, name=source_name)
        return df, warnings

    def set_dataframe(
        self,
        df,
        *,
        name: str | None = None,
        summary: FinancialSummary | None = None,
    ) -> FinancialSummary:
        """Attach a dataframe and build the brief used for chat questions."""
        if df is None or len(df) == 0:
            raise ValueError("The dataset has no rows.")

        self._dataframe = df.copy()
        self._dataset_name = name or self._dataset_name or "uploaded CSV"
        self._summary = summary or summarize(self._dataframe)
        self._brief = build_dataset_brief(
            self._dataframe,
            self._summary,
            dataset_name=self._dataset_name,
            currency_symbol=self.currency_symbol,
            max_chars=self.max_context_chars,
        )
        # A new dataset starts a fresh conversation so old answers are not
        # accidentally applied to the new data.
        self._history.clear()
        return self._summary

    def clear_dataset(self, *, clear_history: bool = True) -> None:
        self._dataframe = None
        self._summary = None
        self._brief = None
        self._dataset_name = None
        if clear_history:
            self._history.clear()

    # ------------------------------------------------------------------ chat
    def reset(self) -> None:
        """Clear the conversation but keep the current dataset."""
        self._history.clear()

    def _ensure_llm(self) -> _GroqChat:
        if not self._api_key:
            # Allow a key added after app start (e.g. via Streamlit secrets).
            self._api_key = get_settings().api_key
        if not self._api_key:
            raise MissingAPIKeyError(
                "No Groq API key found. Add GROQ_API_KEY to your .env file "
                "(or to Streamlit secrets) and restart the app."
            )
        if self._llm is None:
            self._llm = _GroqChat(
                api_key=self._api_key,
                model=self.model_name,
                temperature=self.temperature,
            )
        return self._llm

    def _build_messages(self, message: str) -> list[dict[str, str]]:
        system_prompt = build_system_prompt(
            dataset_name=self._dataset_name,
            dataset_brief=self._brief,
            currency_symbol=self.currency_symbol,
        )
        history = [dict(item) for item in self._history[-self.max_history_messages :]]
        while history and history[0].get("role") != "user":
            history.pop(0)
        return [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ]

    def _remember(self, message: str, response: str) -> None:
        self._history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
        )
        if len(self._history) > self.max_history_messages:
            self._history = self._history[-self.max_history_messages :]

    def chat(self, message: str) -> str:
        """Send one message and return the complete assistant response."""
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        if len(message) > _MAX_MESSAGE_CHARS:
            raise ValueError(f"Please keep messages under {_MAX_MESSAGE_CHARS:,} characters.")

        llm = self._ensure_llm()
        response = llm.invoke(self._build_messages(message))
        text = _content_to_text(getattr(response, "content", response)).strip()
        if not text:
            raise RuntimeError("The model returned an empty response. Please try again.")
        self._remember(message, text)
        return text

    def chat_stream(self, message: str) -> Iterator[str]:
        """Stream the assistant response chunk by chunk."""
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        if len(message) > _MAX_MESSAGE_CHARS:
            raise ValueError(f"Please keep messages under {_MAX_MESSAGE_CHARS:,} characters.")

        llm = self._ensure_llm()
        chunks: list[str] = []
        try:
            for chunk in llm.stream(self._build_messages(message)):
                text = _content_to_text(getattr(chunk, "content", chunk))
                if text:
                    chunks.append(text)
                    yield text
        except Exception as exc:
            logger.warning("Streaming chat failed: %s", exc)
            raise

        response = "".join(chunks).strip()
        if not response:
            raise RuntimeError("The model returned an empty response. Please try again.")
        self._remember(message, response)


# Backwards-compatible name for code that imported the old class.
Model = FinanceAssistant

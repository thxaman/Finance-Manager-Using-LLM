"""Assistant tests that use a fake provider and never call the network."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import Settings
from model import FinanceAssistant, MissingAPIKeyError, _GroqChat


class FakeLLM:
    def __init__(self, *responses: str | list[str]) -> None:
        self.responses = list(responses)
        self.seen: list[list[dict[str, str]]] = []

    def invoke(self, messages):
        self.seen.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))

    def stream(self, messages):
        self.seen.append(messages)
        for part in self.responses.pop(0):
            yield SimpleNamespace(content=part)


class AssistantTests(unittest.TestCase):
    def _assistant(self, **kwargs) -> FinanceAssistant:
        return FinanceAssistant(api_key="test-key", max_history_messages=8, **kwargs)

    def test_chat_adds_system_history_and_user_message(self) -> None:
        assistant = self._assistant()
        fake = FakeLLM("A concise answer.")
        assistant._llm = fake

        self.assertEqual(assistant.chat("What is rent?"), "A concise answer.")
        self.assertEqual(
            assistant._history,
            [
                {"role": "user", "content": "What is rent?"},
                {"role": "assistant", "content": "A concise answer."},
            ],
        )

        messages = assistant._build_messages("And utilities?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], assistant._history[0])
        self.assertEqual(messages[2], assistant._history[1])
        self.assertEqual(messages[-1], {"role": "user", "content": "And utilities?"})

    def test_stream_joins_chunks_into_one_history_entry(self) -> None:
        assistant = self._assistant()
        assistant._llm = FakeLLM(["Rent is ", "your largest category."])

        self.assertEqual(
            list(assistant.chat_stream("Biggest category?")),
            ["Rent is ", "your largest category."],
        )
        self.assertEqual(
            assistant._history[-1],
            {"role": "assistant", "content": "Rent is your largest category."},
        )

    def test_groq_adapter_parses_invoke_and_stream_responses(self) -> None:
        with patch("model.Groq") as groq_cls:
            chat = _GroqChat(api_key="test-key", model="test-model", temperature=0.1)
            create = groq_cls.return_value.chat.completions.create

            create.return_value = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Complete response")
                    )
                ]
            )
            reply = chat.invoke([{"role": "user", "content": "Hi"}])
            self.assertEqual(reply.content, "Complete response")

            create.return_value = [
                SimpleNamespace(choices=[]),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Streamed"))]
                ),
            ]
            streamed = chat.stream([{"role": "user", "content": "Hi"}])
            self.assertEqual([reply.content for reply in streamed], ["Streamed"])

    def test_missing_api_key_is_reported_cleanly(self) -> None:
        assistant = self._assistant()
        assistant._api_key = None
        assistant._llm = None
        settings = Settings(
            api_key=None,
            model="test-model",
            temperature=0.2,
            currency_symbol="$",
            max_history_messages=8,
            max_context_chars=14_000,
            max_upload_mb=25,
        )
        with patch("model.get_settings", return_value=settings):
            with self.assertRaises(MissingAPIKeyError):
                assistant.chat("Hello")


if __name__ == "__main__":
    unittest.main()

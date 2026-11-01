"""Streamlit integration tests (no Groq API key required).

Run with:
    python -m unittest test_app.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

SAMPLE = Path(__file__).with_name("Sample.csv")


class AppIntegrationTests(unittest.TestCase):
    def _new_app(self) -> AppTest:
        app = AppTest.from_file("streamlit_app.py", default_timeout=30)
        app.run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        return app

    def _upload_sample(self, app: AppTest) -> None:
        uploader = app.get("file_uploader")[0]
        if not hasattr(uploader, "upload"):
            self.skipTest("This Streamlit version does not support uploader testing.")
        uploader.upload("Sample.csv", SAMPLE.read_bytes(), "text/csv").run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])

    def test_empty_state(self) -> None:
        app = self._new_app()
        self.assertIn("Finance Manager", app.title[0].value)
        self.assertEqual(len(app.chat_input), 1)

    def test_upload_renders_dashboard_and_remove_resets(self) -> None:
        app = self._new_app()
        self._upload_sample(app)

        metrics = [(metric.label, metric.value) for metric in app.metric]
        self.assertEqual(
            metrics,
            [
                ("Rows", "8"),
                ("Total spending", "$1,900"),
                ("Assets", "$150,000"),
                ("Liabilities", "$90,000"),
                ("Net worth", "$60,000"),
            ],
        )

        remove = next(
            button for button in app.sidebar.button if button.label == "Remove dataset"
        )
        remove.click().run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        self.assertIsNone(app.session_state.df)
        self.assertIsNone(app.get("file_uploader")[0].value)

    def test_chat_stream_updates_session(self) -> None:
        app = self._new_app()
        self._upload_sample(app)

        class FakeLLM:
            def stream(self, messages):  # noqa: ARG002 - matches the assistant interface
                yield SimpleNamespace(content="Rent is your ")
                yield SimpleNamespace(content="largest category at **$1,000**.")

        app.session_state.assistant._llm = FakeLLM()
        app.chat_input[0].set_value("What is my biggest expense?").run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        self.assertEqual(
            app.session_state.messages,
            [
                {"role": "user", "content": "What is my biggest expense?"},
                {
                    "role": "assistant",
                    "content": "Rent is your largest category at **$1,000**.",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()

"""Interactive command-line client for the finance assistant.

Usage:
    python test.py
    python test.py Sample.csv
    python test.py path/to/expenses.csv --model openai/gpt-oss-120b

Commands:
    /clear   clear the conversation
    /quit    exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import ASSISTANT_NAME
from model import FinanceAssistant, MissingAPIKeyError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the finance assistant in a terminal.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="Sample.csv",
        help="CSV file to analyse (default: Sample.csv)",
    )
    parser.add_argument(
        "--model",
        help="Groq model id to use instead of GROQ_MODEL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)

    assistant = FinanceAssistant(model=args.model)
    if csv_path.exists():
        try:
            df, warnings = assistant.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - CLI should show a clean error
            print(f"Could not load {csv_path}: {exc}", file=sys.stderr)
            return 1
        print(f"Loaded {len(df):,} rows from {csv_path.name}.")
        for warning in warnings:
            print(f"  note: {warning}")
    else:
        print(f"{csv_path} was not found; continuing without a dataset.")

    print(f"\nChatting with {ASSISTANT_NAME}. Type /clear to reset, /quit to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if message.lower() in {"/clear", "/reset"}:
            assistant.reset()
            print("Conversation cleared.\n")
            continue

        try:
            print(f"{ASSISTANT_NAME}: ", end="", flush=True)
            for chunk in assistant.chat_stream(message):
                print(chunk, end="", flush=True)
            print("\n")
        except MissingAPIKeyError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - keep the CLI alive after API errors
            print(f"\nAssistant error: {exc}\n", file=sys.stderr)

    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

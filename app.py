"""Streamlit UI for the Expense Tracker & Finance Manager."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import ASSISTANT_NAME, get_settings
from data import FinancialSummary, format_currency
from model import FinanceAssistant, MissingAPIKeyError

logger = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).with_name("Sample.csv")
AVATARS = {"user": "🧑", "assistant": "🤖"}


# --------------------------------------------------------------------------- state
def _init_state() -> None:
    defaults = {
        "messages": [],
        "dataset_id": None,
        "dataset_name": None,
        "df": None,
        "summary": None,
        "warnings": [],
        "pending_prompt": None,
        "uploader_nonce": 0,
        "failed_digest": None,
        "upload_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not isinstance(st.session_state.get("assistant"), FinanceAssistant):
        # Create the assistant once per session, and reset stale state after a
        # hot reload that changed the assistant class.
        st.session_state.assistant = FinanceAssistant()
        st.session_state.messages = []
        st.session_state.dataset_id = None
        st.session_state.dataset_name = None
        st.session_state.df = None
        st.session_state.summary = None
        st.session_state.warnings = []
        st.session_state.failed_digest = None
        st.session_state.upload_error = None


def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.assistant.reset()


def _clear_dataset(*, reset_uploader: bool = False) -> None:
    st.session_state.assistant.clear_dataset()
    st.session_state.messages = []
    for key in (
        "dataset_id",
        "dataset_name",
        "df",
        "summary",
        "warnings",
        "failed_digest",
        "upload_error",
    ):
        st.session_state[key] = [] if key == "warnings" else None
    if reset_uploader:
        # Changing the uploader key resets the widget so the removed file does
        # not immediately get loaded again on the next rerun.
        st.session_state.uploader_nonce += 1


def _load_dataset(raw: bytes, name: str) -> None:
    assistant: FinanceAssistant = st.session_state.assistant
    df, warnings = assistant.read_csv(raw, name=name)
    st.session_state.df = df
    st.session_state.summary = assistant.summary
    st.session_state.dataset_id = hashlib.sha256(raw).hexdigest()
    st.session_state.dataset_name = name
    st.session_state.warnings = warnings
    st.session_state.messages = []
    st.session_state.failed_digest = None
    st.session_state.upload_error = None


# --------------------------------------------------------------------------- sidebar
def _render_sidebar(settings) -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption(f"Model: `{settings.model}`")
        if settings.api_key:
            st.success("Groq API key detected", icon="✅")
        else:
            st.warning(
                "No Groq API key found. Add `GROQ_API_KEY` to `.env` or Streamlit secrets.",
                icon="🔑",
            )
            st.caption("Create a key at https://console.groq.com/keys")

        st.divider()
        st.subheader("💬 Conversation")
        if st.button(
            "Clear conversation",
            icon="🗑️",
            width="stretch",
            disabled=not st.session_state.messages,
        ):
            _reset_chat()
            st.rerun()

        st.divider()
        st.subheader("📂 Dataset")
        if st.session_state.df is not None:
            st.caption(
                f"{st.session_state.dataset_name or 'dataset'} · "
                f"{len(st.session_state.df):,} rows"
            )
            if st.button("Remove dataset", icon="❌", width="stretch"):
                _clear_dataset(reset_uploader=True)
                st.rerun()
        else:
            st.caption("No dataset loaded yet.")
        if SAMPLE_PATH.exists():
            st.download_button(
                "Download sample CSV",
                data=SAMPLE_PATH.read_bytes(),
                file_name="Sample.csv",
                mime="text/csv",
                icon="⬇️",
                width="stretch",
            )

        st.divider()
        with st.expander("How it works"):
            st.markdown(
                "Upload a CSV and ask questions about it, or ask general finance "
                "questions without a file. Only a compact brief of the data "
                "(totals, summaries and a few sample rows) is sent to Groq when "
                "you chat — the full CSV stays in this session."
            )
        st.caption(f"Assistant: {ASSISTANT_NAME}")


# --------------------------------------------------------------------------- dashboard
def _metric_cards(summary: FinancialSummary, currency: str) -> None:
    metrics: list[tuple[str, str, str]] = [
        ("Rows", f"{summary.rows:,}", "Rows in the dataset"),
    ]
    if summary.total_spending is not None:
        metrics.append(
            (
                "Total spending",
                format_currency(summary.total_spending, currency),
                "Sum of the amount column",
            )
        )
    if summary.total_assets is not None:
        metrics.append(
            ("Assets", format_currency(summary.total_assets, currency), "Sum of assets")
        )
    if summary.total_liabilities is not None:
        metrics.append(
            (
                "Liabilities",
                format_currency(summary.total_liabilities, currency),
                "Sum of liabilities",
            )
        )
    if summary.net_worth is not None:
        metrics.append(
            (
                "Net worth",
                format_currency(summary.net_worth, currency),
                "Assets minus liabilities",
            )
        )

    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        with column:
            st.metric(label, value, help=help_text, border=True)


def _spending_chart(summary: FinancialSummary) -> alt.Chart | None:
    data = summary.category_totals
    if data.empty:
        return None
    data = data.head(12).copy()
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce").fillna(0)

    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Amount:Q", title="Amount"),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.Color(
                "Category:N",
                legend=None,
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.2f"),
            ],
        )
        .properties(height=max(260, 30 * len(data)), title="Spending by category")
    )


def _balance_chart(summary: FinancialSummary) -> alt.Chart | None:
    rows = []
    if summary.total_assets is not None:
        rows.append({"Type": "Assets", "Value": summary.total_assets})
    if summary.total_liabilities is not None:
        rows.append({"Type": "Liabilities", "Value": summary.total_liabilities})
    if not rows:
        return None

    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=90)
        .encode(
            x=alt.X("Type:N", sort=["Assets", "Liabilities"], title=None),
            y=alt.Y("Value:Q", title="Amount"),
            color=alt.Color(
                "Type:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Assets", "Liabilities"],
                    range=["#1B998B", "#E76F51"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Type:N", title="Type"),
                alt.Tooltip("Value:Q", title="Amount", format=",.2f"),
            ],
        )
        .properties(height=300, title="Assets vs liabilities")
    )


def _monthly_chart(summary: FinancialSummary) -> alt.Chart | None:
    data = summary.monthly_totals
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2.5, color="#2E86AB")
        .encode(
            x=alt.X(
                "Month:T",
                title=None,
                axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount=min(8, len(data))),
            ),
            y=alt.Y("Amount:Q", title="Spending"),
            tooltip=[
                alt.Tooltip("Month:T", title="Month", format="%b %Y"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.2f"),
            ],
        )
        .properties(height=280, title="Monthly spending trend")
    )


def _render_dashboard(df: pd.DataFrame, summary: FinancialSummary, settings) -> None:
    st.subheader("📈 Overview")
    st.caption(
        f"{st.session_state.dataset_name or 'dataset'} · "
        f"{summary.rows:,} rows × {summary.columns} columns"
    )
    if st.session_state.warnings:
        with st.expander(f"Data cleaning notes ({len(st.session_state.warnings)})"):
            for note in st.session_state.warnings[:12]:
                st.markdown(f"- {note}")
            if len(st.session_state.warnings) > 12:
                st.caption(f"…and {len(st.session_state.warnings) - 12} more.")

    _metric_cards(summary, settings.currency_symbol)

    overview_tab, data_tab = st.tabs(["📊 Charts", "🧾 Data"])
    with overview_tab:
        left, right = st.columns(2)
        with left:
            chart = _spending_chart(summary)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                if len(summary.category_totals) > 12:
                    st.caption("Showing the top 12 categories.")
            else:
                st.info(
                    "Add numeric `Amount` and `Category` columns to see spending by category."
                )
        with right:
            chart = _balance_chart(summary)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            else:
                st.info("Add `Assets` and/or `Liabilities` columns for a balance chart.")

        monthly = _monthly_chart(summary)
        if monthly is not None:
            st.altair_chart(monthly, width="stretch")

    with data_tab:
        st.dataframe(df, width="stretch", hide_index=True)
        if summary.amount_col:
            st.caption(f"Detected amount column: `{summary.amount_col}`")
        if summary.category_col:
            st.caption(f"Detected category column: `{summary.category_col}`")
        if summary.date_col:
            st.caption(f"Detected date column: `{summary.date_col}`")

        column_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": [str(dtype) for dtype in df.dtypes],
                "Non-null": [int(df[column].notna().sum()) for column in df.columns],
                "Missing": [int(df[column].isna().sum()) for column in df.columns],
            }
        )
        with st.expander("Column details"):
            st.dataframe(column_info, width="stretch", hide_index=True)

        stem = Path(st.session_state.dataset_name or "dataset").stem
        st.download_button(
            "Download cleaned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_cleaned.csv",
            mime="text/csv",
            icon="⬇️",
        )


def _render_empty_state() -> None:
    st.info(
        "Upload a CSV to unlock the dashboard and data-aware chat. "
        "The assistant can still answer general finance questions without a file."
    )
    with st.expander("Expected CSV format"):
        st.code(
            "Category,Amount,Date,Assets,Liabilities\n"
            "Groceries,150,2025-01-05,150000,90000\n"
            "Rent,1000,2025-01-06,,",
            language="csv",
        )
        st.caption(
            "`Category` and `Amount` unlock the spending chart. `Date`, `Assets` "
            "and `Liabilities` are optional. Column names are matched "
            "case-insensitively, and currency-formatted values such as `$1,200.50` "
            "are cleaned automatically."
        )


# --------------------------------------------------------------------------- chat
def _render_suggestions(summary: FinancialSummary | None) -> None:
    if summary is not None and summary.amount_col:
        suggestions = [
            "Give me an overall analysis of my finances.",
            "What is my biggest expense category?",
            "How can I reduce my monthly spending?",
        ]
    else:
        suggestions = [
            "How do I build a monthly budget?",
            "How large should my emergency fund be?",
            "Explain needs vs wants spending.",
        ]

    columns = st.columns(len(suggestions))
    for index, (column, suggestion) in enumerate(zip(columns, suggestions)):
        with column:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                width="stretch",
            ):
                st.session_state.pending_prompt = suggestion


def _render_chat(summary: FinancialSummary | None) -> None:
    st.header("🤖 Chat with your Finance Assistant")
    if not st.session_state.messages:
        st.caption(
            "Ask about your uploaded data or general personal-finance questions. "
            f"{ASSISTANT_NAME} remembers the current conversation."
        )
        _render_suggestions(summary)

    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message(role, avatar=AVATARS.get(role)):
            st.markdown(message.get("content", ""))

    prompt = st.chat_input("Ask about your spending, budget, or uploaded CSV…")
    if not prompt:
        prompt = st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    assistant: FinanceAssistant = st.session_state.assistant
    response = None
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            response = st.write_stream(assistant.chat_stream(prompt))
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface a friendly message to the user
            logger.warning("Assistant request failed: %s", exc)
            st.error(f"Sorry, the assistant could not answer that: {exc}")
            st.caption("Check your API key, model name and Groq rate limits, then try again.")

    if response:
        text = response if isinstance(response, str) else "".join(map(str, response))
        if text.strip():
            st.session_state.messages.append({"role": "assistant", "content": text})


# --------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(
        page_title="Expense Tracker & Finance Manager",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    settings = get_settings()

    st.title("📊 Expense Tracker & Finance Manager")
    st.markdown(
        "Track spending, visualise your finances, and chat with an AI finance "
        "assistant about your data. 💸"
    )

    st.header("📂 Upload your data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Include Category and Amount columns; Date, Assets and Liabilities are optional.",
        key=f"csv_upload_{st.session_state.uploader_nonce}",
    )

    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(raw) > max_bytes:
            st.session_state.upload_error = (
                f"That file is {len(raw) / 1024 / 1024:.1f} MB. "
                f"The limit is {settings.max_upload_mb} MB."
            )
        else:
            digest = hashlib.sha256(raw).hexdigest()
            if digest != st.session_state.dataset_id and digest != st.session_state.failed_digest:
                st.session_state.upload_error = None
                try:
                    with st.spinner("Reading and cleaning your data…"):
                        _load_dataset(raw, uploaded_file.name)
                except Exception as exc:  # noqa: BLE001 - CSV errors are user-facing
                    logger.warning("Could not load uploaded CSV: %s", exc)
                    _clear_dataset()
                    st.session_state.failed_digest = digest
                    st.session_state.upload_error = f"Could not read that CSV: {exc}"
                else:
                    st.toast(f"Loaded {len(st.session_state.df):,} rows.", icon="✅")
    elif st.session_state.dataset_id is not None:
        _clear_dataset()
    else:
        st.session_state.failed_digest = None
        st.session_state.upload_error = None

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    _render_sidebar(settings)

    if st.session_state.df is not None and st.session_state.summary is not None:
        _render_dashboard(
            st.session_state.df,
            st.session_state.summary,
            settings,
        )
    else:
        _render_empty_state()

    st.divider()
    _render_chat(st.session_state.summary)
    st.caption(
        "Built with Streamlit, Pandas and Groq. Educational tool — not a substitute "
        "for professional financial advice."
    )


if __name__ == "__main__":
    main()

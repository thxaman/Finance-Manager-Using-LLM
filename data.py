"""CSV loading, cleaning and summarisation helpers.

The assistant is deliberately given a compact, computed brief instead of a
Python-executing CSV agent. This keeps the app useful for typical finance
questions while avoiding arbitrary code execution from model output.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import pandas as pd

_NUMERIC_HINTS = (
    "amount",
    "expense",
    "cost",
    "spend",
    "price",
    "total",
    "balance",
    "income",
    "salary",
    "asset",
    "liabilit",
    "debt",
    "payment",
    "saving",
    "budget",
)

_ID_HINTS = (
    "identifier",
    "zipcode",
    "postalcode",
    "phonenumber",
    "accountnumber",
    "cardnumber",
    "customerid",
    "userid",
    "transactionid",
    "invoiceid",
    "orderid",
    "employeeid",
    "accountid",
)
_CATEGORY_HINTS = ("category", "type", "group", "merchant", "vendor", "description")
_DATE_HINTS = ("date", "month", "period", "timestamp")

_ALIASES: dict[str, tuple[str, ...]] = {
    "amount": (
        "amount",
        "expense",
        "expenses",
        "expenseamount",
        "cost",
        "spend",
        "spending",
        "price",
        "payment",
        "value",
        "debit",
    ),
    "category": (
        "category",
        "categories",
        "expensecategory",
        "type",
        "group",
        "merchant",
        "vendor",
        "description",
    ),
    "assets": ("assets", "asset", "totalassets"),
    "liabilities": ("liabilities", "liability", "totalliabilities", "debt", "debts"),
    "date": ("date", "transactiondate", "month", "period", "timestamp"),
}

_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass(frozen=True)
class FinancialSummary:
    """Pre-computed facts used by both the UI and the assistant."""

    rows: int
    columns: int
    amount_col: str | None
    category_col: str | None
    date_col: str | None
    assets_col: str | None
    liabilities_col: str | None
    total_spending: float | None
    total_assets: float | None
    total_liabilities: float | None
    net_worth: float | None
    category_totals: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_totals: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric_stats: pd.DataFrame = field(default_factory=pd.DataFrame)


def normalise_name(name: Any) -> str:
    """Normalise a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, kind: str) -> str | None:
    """Find the most likely column for ``kind`` (amount, category, date, ...)."""
    aliases = _ALIASES.get(kind, ())
    if not aliases or df.empty:
        return None

    normalised = [(column, normalise_name(column)) for column in df.columns]

    # Prefer exact matches so e.g. "Amount" wins over "Amount paid".
    for alias in aliases:
        for column, name in normalised:
            if name == alias:
                return str(column)

    for alias in aliases:
        for column, name in normalised:
            if alias in name:
                return str(column)
    return None


def _to_number(series: pd.Series) -> pd.Series:
    """Parse common currency/number strings into floats."""
    text = series.astype("string").str.strip()
    text = text.str.replace(r"^(?:USD|EUR|GBP|INR|CAD|AUD|JPY)\s*", "", regex=True, case=False)
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    text = text.str.replace(r"[$€£₹¥,\s]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _looks_like_identifier(name: str) -> bool:
    """Avoid turning ID/code columns into measures without skipping words like 'paid'."""
    if name in {"id", "ids"}:
        return True
    return any(hint in name for hint in _ID_HINTS)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in df.columns:
        base = str(column).replace("\ufeff", "").strip() or "Column"
        seen[base] = seen.get(base, 0) + 1
        columns.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df.columns = columns

    # Drop completely empty index columns that spreadsheet exports often add.
    empty_unnamed = [
        column
        for column in df.columns
        if normalise_name(column).startswith("unnamed") and df[column].isna().all()
    ]
    if empty_unnamed:
        df = df.drop(columns=empty_unnamed)
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean a raw dataframe and return it together with human-readable notes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataframe expects a pandas DataFrame")

    warnings: list[str] = []
    df = _clean_columns(df)

    original_rows = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) < original_rows:
        warnings.append(f"Removed {original_rows - len(df)} fully blank row(s).")

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        df[column] = df[column].replace({"": pd.NA})

    for column in list(df.columns):
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        name = normalise_name(column)
        non_null = int(series.notna().sum())
        if non_null == 0:
            continue

        # Don't turn category labels such as numeric types into measures.
        if any(hint in name for hint in _CATEGORY_HINTS) and not any(
            hint in name for hint in _NUMERIC_HINTS
        ):
            continue
        if _looks_like_identifier(name):
            continue

        parsed = _to_number(series)
        parsed_count = int(parsed.notna().sum())
        if parsed_count == 0:
            continue

        hinted = any(hint in name for hint in _NUMERIC_HINTS)
        if hinted or parsed_count / non_null >= 0.9:
            df[column] = parsed
            warnings.append(f"Converted '{column}' to numeric values.")

    for column in list(df.columns):
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
            continue
        name = normalise_name(column)
        if not any(hint in name for hint in _DATE_HINTS):
            continue

        non_null = int(series.notna().sum())
        if non_null == 0:
            continue
        try:
            parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:  # pandas < 2.0
            parsed_dates = pd.to_datetime(series, errors="coerce")
        if parsed_dates.notna().sum() / non_null >= 0.8:
            df[column] = parsed_dates
            warnings.append(f"Parsed '{column}' as dates.")

    return df.reset_index(drop=True), warnings


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes, trying common encodings and delimiter sniffing."""
    last_error: Exception | None = None
    for encoding in _FALLBACK_ENCODINGS:
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
            # A semicolon/tab-delimited export can parse as one wide column.
            if frame.shape[1] == 1:
                header = str(frame.columns[0])
                if ";" in header or "\t" in header or "|" in header:
                    return pd.read_csv(
                        io.BytesIO(data),
                        encoding=encoding,
                        sep=None,
                        engine="python",
                    )
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            last_error = exc
            try:
                return pd.read_csv(
                    io.BytesIO(data),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                )
            except Exception as inner_exc:  # noqa: BLE001 - report the most useful error
                last_error = inner_exc
                continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The CSV file is empty.") from exc
    raise ValueError(f"Could not parse the CSV file: {last_error}") from last_error


def _read_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (bytes, bytearray, memoryview)):
        return _read_csv_bytes(bytes(source))

    if isinstance(source, (str, Path)):
        return _read_csv_bytes(Path(source).read_bytes())

    read = getattr(source, "read", None)
    if callable(read):
        seek = getattr(source, "seek", None)
        if callable(seek):
            try:
                seek(0)
            except (OSError, ValueError):
                pass
        data = read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("File-like objects must return bytes or text.")
        return _read_csv_bytes(bytes(data))

    raise TypeError("Unsupported CSV source. Pass a path, bytes, file object or DataFrame.")


def load_csv(source: str | Path | bytes | BinaryIO | TextIO | pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Load and clean a CSV from a path, bytes, file object or DataFrame."""
    raw = _read_source(source)
    if raw.empty and len(raw.columns) == 0:
        raise ValueError("The CSV file contains no columns.")

    df, warnings = clean_dataframe(raw)
    if df.empty:
        raise ValueError("The CSV file contains no data rows.")
    return df, warnings


def summarize(df: pd.DataFrame) -> FinancialSummary:
    """Compute the facts used by the dashboard and the assistant brief."""
    amount_col = find_column(df, "amount")
    category_col = find_column(df, "category")
    date_col = find_column(df, "date")
    assets_col = find_column(df, "assets")
    liabilities_col = find_column(df, "liabilities")

    # Only accept columns whose values are usable for the corresponding role.
    if amount_col and not pd.api.types.is_numeric_dtype(df[amount_col]):
        amount_col = None
    if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        date_col = None
    if assets_col and not pd.api.types.is_numeric_dtype(df[assets_col]):
        assets_col = None
    if liabilities_col and not pd.api.types.is_numeric_dtype(df[liabilities_col]):
        liabilities_col = None

    total_spending = _column_total(df, amount_col)
    total_assets = _column_total(df, assets_col)
    total_liabilities = _column_total(df, liabilities_col)

    net_worth: float | None = None
    if total_assets is not None and total_liabilities is not None:
        net_worth = total_assets - total_liabilities

    category_totals = pd.DataFrame(columns=["Category", "Amount"])
    if amount_col and category_col and category_col != amount_col:
        working = df[[category_col, amount_col]].dropna(subset=[amount_col]).copy()
        if not working.empty:
            working[category_col] = (
                working[category_col].fillna("(uncategorised)").astype("string").str.strip()
            )
            grouped = (
                working.groupby(category_col, dropna=False, observed=True)[amount_col]
                .sum()
                .sort_values(ascending=False)
            )
            category_totals = grouped.rename_axis("Category").reset_index(name="Amount")
            category_totals = category_totals[category_totals["Amount"].fillna(0) != 0]

    monthly_totals = pd.DataFrame(columns=["Month", "Amount"])
    if amount_col and date_col and date_col != amount_col:
        working = df[[date_col, amount_col]].dropna().copy()
        if not working.empty and pd.api.types.is_datetime64_any_dtype(working[date_col]):
            months = working[date_col].dt.to_period("M").dt.to_timestamp()
            grouped = working.groupby(months)[amount_col].sum().sort_index()
            monthly_totals = grouped.rename_axis("Month").reset_index(name="Amount")
            monthly_totals = monthly_totals.tail(24).reset_index(drop=True)

    numeric_stats = pd.DataFrame(columns=["Column", "Count", "Sum", "Mean", "Min", "Max"])
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.agg(["count", "sum", "mean", "min", "max"]).T.reset_index()
        stats.columns = ["Column", "Count", "Sum", "Mean", "Min", "Max"]
        stats["Count"] = stats["Count"].astype(int)
        numeric_stats = stats.head(20)

    return FinancialSummary(
        rows=len(df),
        columns=len(df.columns),
        amount_col=amount_col,
        category_col=category_col,
        date_col=date_col,
        assets_col=assets_col,
        liabilities_col=liabilities_col,
        total_spending=total_spending,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        net_worth=net_worth,
        category_totals=category_totals,
        monthly_totals=monthly_totals,
        numeric_stats=numeric_stats,
    )


def _column_total(df: pd.DataFrame, column: str | None) -> float | None:
    if not column or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if not series.notna().any():
        return None
    return float(series.sum())


def _fmt(value: float | None, currency: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}{currency}{abs(value):,.2f}"


def format_currency(value: float | None, symbol: str = "$") -> str:
    """Format a value for display in the dashboard."""
    if value is None or pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 1_000:
        return f"{sign}{symbol}{absolute:,.0f}"
    return f"{sign}{symbol}{absolute:,.2f}"


def build_dataset_brief(
    df: pd.DataFrame,
    summary: FinancialSummary,
    *,
    dataset_name: str = "uploaded CSV",
    currency_symbol: str = "$",
    max_rows: int = 12,
    max_chars: int = 14_000,
) -> str:
    """Create a compact text brief that the LLM can answer questions from."""
    sections: list[str] = []
    sections.append(
        "\n".join(
            [
                f"DATASET: {dataset_name}",
                f"Rows: {summary.rows:,} | Columns: {summary.columns}",
                "Detected fields:",
                f"- spending amount: {summary.amount_col or 'not detected'}",
                f"- category: {summary.category_col or 'not detected'}",
                f"- date: {summary.date_col or 'not detected'}",
                f"- assets: {summary.assets_col or 'not detected'}",
                f"- liabilities: {summary.liabilities_col or 'not detected'}",
            ]
        )
    )

    totals = [
        f"- total spending: {_fmt(summary.total_spending, currency_symbol)}",
        f"- total assets: {_fmt(summary.total_assets, currency_symbol)}",
        f"- total liabilities: {_fmt(summary.total_liabilities, currency_symbol)}",
        f"- net worth (assets - liabilities): {_fmt(summary.net_worth, currency_symbol)}",
    ]
    sections.append("TOTALS (computed from the full dataset)\n" + "\n".join(totals))

    column_lines = [f"- {column} ({dtype})" for column, dtype in df.dtypes.items()]
    sections.append("COLUMNS\n" + "\n".join(column_lines[:40]))

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    if not missing.empty:
        sections.append(
            "MISSING VALUES\n"
            + "\n".join(f"- {column}: {int(count):,}" for column, count in missing.items())
        )

    if not summary.category_totals.empty:
        lines = [
            f"- {row['Category']}: {_fmt(float(row['Amount']), currency_symbol)}"
            for _, row in summary.category_totals.head(25).iterrows()
        ]
        sections.append("SPENDING BY CATEGORY\n" + "\n".join(lines))

    if not summary.monthly_totals.empty:
        lines = [
            f"- {row['Month']:%Y-%m}: {_fmt(float(row['Amount']), currency_symbol)}"
            for _, row in summary.monthly_totals.tail(24).iterrows()
        ]
        sections.append("SPENDING BY MONTH\n" + "\n".join(lines))

    if (
        summary.amount_col
        and summary.category_col
        and summary.date_col
        and not summary.category_totals.empty
    ):
        working = df[[summary.date_col, summary.category_col, summary.amount_col]].dropna().copy()
        if not working.empty:
            working["_Month"] = working[summary.date_col].dt.to_period("M").astype(str)
            top_categories = (
                summary.category_totals.head(8)["Category"].astype("string").tolist()
            )
            working[summary.category_col] = working[summary.category_col].astype("string")
            working = working[working[summary.category_col].isin(top_categories)]
            if not working.empty:
                matrix = working.pivot_table(
                    index="_Month",
                    columns=summary.category_col,
                    values=summary.amount_col,
                    aggfunc="sum",
                    fill_value=0,
                ).sort_index().tail(12)
                matrix.index.name = "Month"
                matrix.columns = [str(column) for column in matrix.columns]
                sections.append(
                    "CATEGORY BY MONTH (recent 12 months, top categories; "
                    "values are sums)\n" + matrix.to_csv(float_format="%.2f")
                )

    if summary.amount_col:
        top_columns = list(
            dict.fromkeys(
                column
                for column in (summary.date_col, summary.category_col, summary.amount_col)
                if column
            )
        )
        working = df[top_columns].dropna(subset=[summary.amount_col]).copy()
        if not working.empty:
            working["_AbsoluteAmount"] = working[summary.amount_col].abs()
            top = working.nlargest(min(10, len(working)), "_AbsoluteAmount").drop(
                columns="_AbsoluteAmount"
            )
            for column in top.select_dtypes(include=["object", "string"]).columns:
                top[column] = top[column].astype("string").str.slice(0, 80)
            sections.append(
                "TOP TRANSACTIONS BY ABSOLUTE AMOUNT\n" + top.to_csv(index=False)
            )

    if not summary.numeric_stats.empty:
        lines = []
        for _, row in summary.numeric_stats.head(15).iterrows():
            lines.append(
                f"- {row['Column']}: count={int(row['Count'])}, "
                f"sum={_fmt(float(row['Sum']), currency_symbol)}, "
                f"mean={_fmt(float(row['Mean']), currency_symbol)}, "
                f"min={_fmt(float(row['Min']), currency_symbol)}, "
                f"max={_fmt(float(row['Max']), currency_symbol)}"
            )
        sections.append("NUMERIC COLUMN STATISTICS\n" + "\n".join(lines))

    value_counts_sections: list[str] = []
    for column in df.select_dtypes(include=["object", "string", "category"]).columns:
        unique = df[column].nunique(dropna=True)
        if 0 < unique <= 25:
            counts = df[column].fillna("(missing)").value_counts().head(10)
            lines = [f"- {value}: {count:,}" for value, count in counts.items()]
            value_counts_sections.append(f"{column}:\n" + "\n".join(lines))
    if value_counts_sections:
        sections.append("CATEGORICAL VALUE COUNTS\n" + "\n\n".join(value_counts_sections))

    sample = df.head(max_rows).copy()
    for column in sample.select_dtypes(include=["object", "string"]).columns:
        sample[column] = sample[column].astype("string").str.slice(0, 80)
    sections.append(f"SAMPLE ROWS (first {min(max_rows, len(df))})\n" + sample.to_csv(index=False))

    brief = "\n\n".join(sections)
    if len(brief) > max_chars:
        brief = (
            brief[: max_chars - 120].rstrip()
            + "\n\n[Dataset brief truncated to fit the model context window.]"
        )
    return brief

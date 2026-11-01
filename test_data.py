"""Data-layer tests that do not require an API key.

Run with:
    python -m unittest test_data.py -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data import find_column, format_currency, load_csv, summarize

SAMPLE = Path(__file__).with_name("Sample.csv")


class SampleDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df, self.warnings = load_csv(SAMPLE)
        self.summary = summarize(self.df)

    def test_columns_are_detected(self) -> None:
        self.assertEqual(self.summary.amount_col, "Amount")
        self.assertEqual(self.summary.category_col, "Category")
        self.assertEqual(self.summary.assets_col, "Assets")
        self.assertEqual(self.summary.liabilities_col, "Liabilities")

    def test_totals_are_computed(self) -> None:
        self.assertEqual(self.summary.rows, 8)
        self.assertAlmostEqual(self.summary.total_spending or 0, 1900)
        self.assertAlmostEqual(self.summary.total_assets or 0, 150_000)
        self.assertAlmostEqual(self.summary.total_liabilities or 0, 90_000)
        self.assertAlmostEqual(self.summary.net_worth or 0, 60_000)

    def test_category_totals_are_sorted(self) -> None:
        self.assertEqual(self.summary.category_totals.iloc[0]["Category"], "Rent")
        self.assertAlmostEqual(float(self.summary.category_totals.iloc[0]["Amount"]), 1000)

    def test_currency_formatting(self) -> None:
        self.assertEqual(format_currency(60_000), "$60,000")
        self.assertEqual(format_currency(-12.5), "-$12.50")
        self.assertEqual(format_currency(None), "—")


class CleaningTests(unittest.TestCase):
    def test_currency_dates_and_semicolon_delimiter(self) -> None:
        content = (
            "date;category;amount;paid\n"
            "2025-01-05;Groceries;\u20ac1,200.50;yes\n"
            "2025-02-03;Rent;(1,000.00);no\n"
            "2025-02-04;Utilities;\u20ac200;yes\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messy.csv"
            path.write_text(content, encoding="utf-8")
            df, warnings = load_csv(path)

        self.assertEqual(len(df), 3)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["amount"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertAlmostEqual(float(df["amount"].sum()), 400.5)
        self.assertTrue(any("numeric" in warning for warning in warnings))
        self.assertEqual(find_column(df, "category"), "category")

    def test_identifier_columns_are_not_converted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.csv"
            path.write_text("id,paid,amount\nA1,yes,10\nA2,no,20\n", encoding="utf-8")
            df, _ = load_csv(path)

        self.assertFalse(pd.api.types.is_numeric_dtype(df["id"]))
        self.assertFalse(pd.api.types.is_numeric_dtype(df["paid"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["amount"]))


if __name__ == "__main__":
    unittest.main()

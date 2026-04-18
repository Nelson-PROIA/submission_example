"""
End-to-end smoke test that does NOT require a GPU or the model.

Validates the parts of our submission that are deterministic and don't depend
on the LLM: the output parser, the `/app/data/<name>.csv` contract, and the
`result = ...` convention. For each synthetic question we hand-write an
"ideal" model response (clean, or dressed in noise the parser is supposed to
strip), run it through `strip_code_fence`, rewrite the `/app/data/` path to a
temp dir, exec the result, and diff against the expected DataFrame.

Run: python3 smoke_test.py
Deps: polars. Install with: pip install polars (or uv pip install polars).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import polars as pl

from parser import strip_code_fence


def _make_fixtures(data_dir: Path) -> None:
    pl.DataFrame(
        {
            "region": ["north", "south", "north", "east", "south"],
            "amount": [100.0, 50.0, 75.0, 200.0, 25.0],
            "year": [2024, 2023, 2024, 2024, 2023],
        }
    ).write_csv(data_dir / "sales.csv")

    pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["alice", "bob", "carol"],
        }
    ).write_csv(data_dir / "customers.csv")

    pl.DataFrame(
        {
            "customer_id": [1, 1, 2, 3, 3],
            "amount": [10.0, 20.0, 50.0, 5.0, 15.0],
        }
    ).write_csv(data_dir / "orders.csv")


CASES: list[dict] = [
    {
        "name": "aggregation · clean output",
        "raw_model_output": (
            'sales = pl.read_csv("/app/data/sales.csv", try_parse_dates=True)\n\n'
            "result = (\n"
            "    sales\n"
            '    .group_by("region")\n'
            '    .agg(pl.col("amount").sum().alias("revenue"))\n'
            ")"
        ),
        "expected": pl.DataFrame(
            {"region": ["north", "south", "east"], "revenue": [175.0, 75.0, 200.0]}
        ),
        "sort_key": "region",
    },
    {
        "name": "filter · markdown fence + prose prefix",
        "raw_model_output": (
            "Sure! Here's the code:\n"
            "```python\n"
            'sales = pl.read_csv("/app/data/sales.csv", try_parse_dates=True)\n\n'
            'result = sales.filter((pl.col("amount") > 60) & (pl.col("year") == 2024))\n'
            "```\n"
            "This computes the filter as requested."
        ),
        "expected": pl.DataFrame(
            {
                "region": ["north", "north", "east"],
                "amount": [100.0, 75.0, 200.0],
                "year": [2024, 2024, 2024],
            }
        ),
        "sort_key": "amount",
    },
    {
        "name": "join · chat-template tokens leaked",
        "raw_model_output": (
            '<s>[INST] customers = pl.read_csv("/app/data/customers.csv", try_parse_dates=True)\n'
            'orders = pl.read_csv("/app/data/orders.csv", try_parse_dates=True)\n\n'
            "totals = (\n"
            "    orders\n"
            '    .group_by("customer_id")\n'
            '    .agg(pl.col("amount").sum().alias("total"))\n'
            ")\n\n"
            'result = customers.join(totals, on="customer_id", how="inner")<|im_end|>'
        ),
        "expected": pl.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "name": ["alice", "bob", "carol"],
                "total": [30.0, 50.0, 20.0],
            }
        ),
        "sort_key": "customer_id",
    },
    {
        "name": "window · plain python fence",
        "raw_model_output": (
            "```py\n"
            'orders = pl.read_csv("/app/data/orders.csv", try_parse_dates=True)\n\n'
            "result = orders.with_columns(\n"
            '    pl.col("amount").cum_sum().over("customer_id").alias("running_total")\n'
            ")\n"
            "```"
        ),
        "expected": pl.DataFrame(
            {
                "customer_id": [1, 1, 2, 3, 3],
                "amount": [10.0, 20.0, 50.0, 5.0, 15.0],
                "running_total": [10.0, 30.0, 50.0, 5.0, 20.0],
            }
        ),
        "sort_key": None,
    },
    {
        "name": "parser · prefix + fence + trailing note",
        "raw_model_output": (
            "Certainly! Here is the code:\n"
            "```python\n"
            'sales = pl.read_csv("/app/data/sales.csv", try_parse_dates=True)\n'
            'result = sales.select(pl.col("amount").mean().alias("avg"))\n'
            "```\n"
            "# this computes the mean"
        ),
        "expected": pl.DataFrame({"avg": [90.0]}),
        "sort_key": None,
    },
]


def _run_case(case: dict, data_dir: Path) -> tuple[bool, str]:
    try:
        parsed = strip_code_fence(case["raw_model_output"])
        rewritten = parsed.replace("/app/data/", f"{data_dir}/")
        namespace: dict = {"pl": pl}
        exec(compile(rewritten, "<smoke>", "exec"), namespace)
        result = namespace.get("result")
        if result is None:
            return False, "no `result` variable set"
        if not isinstance(result, pl.DataFrame):
            return False, f"result is {type(result).__name__}, expected DataFrame"

        expected: pl.DataFrame = case["expected"]
        if case["sort_key"]:
            result = result.sort(case["sort_key"])
            expected = expected.sort(case["sort_key"])

        if result.shape != expected.shape:
            return False, f"shape {result.shape} != expected {expected.shape}"
        if result.columns != expected.columns:
            return False, f"columns {result.columns} != {expected.columns}"
        if not result.equals(expected):
            return False, f"data mismatch:\n  got:\n{result}\n  expected:\n{expected}"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="polarsbench_smoke_"))
    try:
        _make_fixtures(tmp)
        passed = 0
        failed = 0
        for case in CASES:
            ok, msg = _run_case(case, tmp)
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {case['name']}")
            if not ok:
                print(f"       {msg}")
                failed += 1
            else:
                passed += 1
        print(f"\n{passed}/{passed + failed} cases passed")
        return 0 if failed == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

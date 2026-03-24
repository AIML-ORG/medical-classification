"""Convert CSV file(s) to Parquet using Polars."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl


def convert_one(input_path: Path, output_path: Path) -> None:
    t0 = time.perf_counter()
    print(f"Reading CSV: {input_path}")
    df = pl.read_csv(input_path)
    read_s = time.perf_counter() - t0

    print(f"Rows: {df.height}, Columns: {df.width}")
    print(f"Schema: {df.schema}")
    print(f"Read time: {read_s:.2f}s")

    t1 = time.perf_counter()
    print(f"Writing Parquet: {output_path}")
    df.write_parquet(output_path, compression="zstd")
    write_s = time.perf_counter() - t1

    total_s = time.perf_counter() - t0
    print(f"Write time: {write_s:.2f}s")
    print(f"Total time: {total_s:.2f}s")
    print("Done.")


def resolve_input_paths(
    repo_root: Path,
    explicit: list[Path],
    positional: list[Path],
) -> list[Path]:
    raw: list[Path] = []
    raw.extend(explicit)
    raw.extend(positional)
    if not raw:
        raw = sorted(repo_root.glob("*.csv"))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in raw:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_single = repo_root / "141943_V4.0_full.csv"

    parser = argparse.ArgumentParser(
        description="Convert CSV to Parquet (Polars). "
        "With no inputs, converts all *.csv in the repo root.",
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="Input CSV (repeat for multiple). If omitted with no positional paths, "
        "all *.csv files under the repo root are converted.",
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Additional CSV paths (same as -i).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (only allowed with exactly one input CSV)",
    )
    parser.add_argument(
        "--default-single",
        action="store_true",
        help=f"If no paths given, convert only {default_single.name} (legacy default)",
    )
    args = parser.parse_args()

    explicit: list[Path] = list(args.input)
    positional: list[Path] = list(args.csv_paths)
    if args.default_single and not explicit and not positional:
        inputs = [default_single.resolve()]
    else:
        inputs = resolve_input_paths(repo_root, explicit, positional)

    if not inputs:
        print(f"No CSV files found under {repo_root}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None and len(inputs) != 1:
        print("--output requires exactly one input CSV", file=sys.stderr)
        sys.exit(2)

    for idx, input_path in enumerate(inputs):
        if not input_path.is_file():
            print(f"Skip (not a file): {input_path}", file=sys.stderr)
            continue
        if args.output is not None and len(inputs) == 1:
            output_path = args.output.resolve()
        else:
            output_path = input_path.with_suffix(".parquet")
        if len(inputs) > 1:
            print(f"\n--- [{idx + 1}/{len(inputs)}] {input_path.name} ---")
        convert_one(input_path, output_path)


if __name__ == "__main__":
    main()

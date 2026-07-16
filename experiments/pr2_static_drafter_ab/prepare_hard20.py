#!/usr/bin/env python3
"""Build the fixed 20-row GSM8K systems-regression dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_INDICES = (3331, 1202, 2345, 1647)
REPEATS = 5
ANSWER_INSTRUCTION = ' Answer with only "#### <number>" and no explanation.'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Preprocessed GSM8K train.parquet")
    parser.add_argument("output", type=Path, help="Destination train.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.source)
    if len(source) <= max(SOURCE_INDICES):
        raise ValueError(f"expected full GSM8K train split, got only {len(source)} rows")

    rows: list[dict] = []
    for repeat in range(REPEATS):
        for source_index in SOURCE_INDICES:
            row = source.iloc[source_index].to_dict()
            extra_info = dict(row["extra_info"])
            question = extra_info["question"]
            row["prompt"] = [{"role": "user", "content": question + ANSWER_INSTRUCTION}]
            extra_info.update(
                {
                    "index": len(rows),
                    "source_index": source_index,
                    "repeat": repeat,
                }
            )
            row["extra_info"] = extra_info
            rows.append(row)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"wrote {len(output)} rows ({len(SOURCE_INDICES)} sources x {REPEATS}) to {args.output}")


if __name__ == "__main__":
    main()

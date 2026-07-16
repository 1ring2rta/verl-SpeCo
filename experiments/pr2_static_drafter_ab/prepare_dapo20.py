#!/usr/bin/env python3
"""Select a deterministic, data-independent DAPO-Math-17k subset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_COUNT = 20
DEFAULT_SALT = "speco-pr2-dapo-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Official DAPO-Math-17k parquet")
    parser.add_argument("output", type=Path, help="Destination subset parquet")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser.parse_args()


def prompt_text(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError(f"unexpected prompt schema for {row['extra_info']['index']}")
    return prompt[0]["content"]


def rank_for(salt: str, text: str) -> int:
    digest = hashlib.sha256(salt.encode() + b"\0" + text.encode()).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    seen_prompts: set[str] = set()
    # Python's heap is a min-heap. Negative ranks keep the largest selected
    # rank at the root so it can be replaced by a smaller one.
    selected: list[tuple[int, str, dict[str, Any]]] = []
    source_rows = 0

    parquet = pq.ParquetFile(args.source)
    for batch in parquet.iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            source_rows += 1
            text = prompt_text(row)
            if text in seen_prompts:
                continue
            seen_prompts.add(text)

            prompt_id = str(row["extra_info"]["index"])
            rank = rank_for(args.salt, text)
            item = (-rank, prompt_id, row)
            if len(selected) < args.count:
                heapq.heappush(selected, item)
            elif rank < -selected[0][0]:
                heapq.heapreplace(selected, item)

    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} prompts, found only {len(selected)}")

    ranked = sorted((-neg_rank, prompt_id, row) for neg_rank, prompt_id, row in selected)
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for selection_rank, (rank, prompt_id, row) in enumerate(ranked):
        row.pop("__index_level_0__", None)
        extra_info = dict(row["extra_info"])
        extra_info.update(
            {
                "selection_rank": selection_rank,
                "selection_salt": args.salt,
            }
        )
        row["extra_info"] = extra_info
        rows.append(row)
        manifest_rows.append(
            {
                "selection_rank": selection_rank,
                "prompt_id": prompt_id,
                "sha256_rank": f"{rank:064x}",
                "prompt_sha256": hashlib.sha256(prompt_text(row).encode()).hexdigest(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "source_rows": source_rows,
        "unique_prompt_count": len(seen_prompts),
        "selection_salt": args.salt,
        "selected_count": len(rows),
        "selected": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"scanned {source_rows} rows / {len(seen_prompts)} unique prompts; "
        f"wrote {len(rows)} rows to {args.output} and {manifest_path}"
    )


if __name__ == "__main__":
    main()

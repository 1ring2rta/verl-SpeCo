#!/usr/bin/env python3
"""Summarize the console metrics emitted by the paired PR 2 regression."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from collections.abc import Iterable


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KEYS = (
    "training/global_step",
    "rollout/spec_accept_rate",
    "rollout/spec_accept_length",
    "timing_s/gen",
    "timing_per_token_ms/gen",
    "perf/time_per_step",
    "critic/score/mean",
    "critic/rewards/mean",
    "actor/grad_norm",
)


def metric_values(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def summarize(path: pathlib.Path) -> dict[str, object]:
    text = ANSI.sub("", path.read_text(errors="replace")).replace("\r", "")
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        row: dict[str, float] = {}
        for key in KEYS:
            match = re.search(rf"(?:^| - ){re.escape(key)}:({NUMBER})", line)
            if match:
                row[key] = float(match.group(1))
        rows.append(row)

    if not rows:
        raise ValueError(f"no training metric rows found in {path}")

    accept_lengths = metric_values(rows, "rollout/spec_accept_length")
    accept_rates = metric_values(rows, "rollout/spec_accept_rate")
    gen_seconds = metric_values(rows[1:], "timing_s/gen")
    gen_ms_per_token = metric_values(rows[1:], "timing_per_token_ms/gen")
    grad_norms = metric_values(rows, "actor/grad_norm")

    return {
        "log": str(path),
        "steps": len(rows),
        "last_step": metric_values(rows, "training/global_step")[-1],
        "draft_auto_loader_logged": bool(
            re.search(r"Using draft model load_format:\s*['\"]?auto['\"]?", text)
        ),
        "shard_loader_logged": "Multi-thread loading shards" in text,
        "accept_len_mean": mean(accept_lengths),
        "accept_len_min": min(accept_lengths) if accept_lengths else None,
        "accept_len_first2": mean(accept_lengths[:2]),
        "accept_len_last2": mean(accept_lengths[-2:]),
        "accept_rate_mean": mean(accept_rates),
        "gen_s_median_steps2_plus": median(gen_seconds),
        "gen_ms_tok_median_steps2_plus": median(gen_ms_per_token),
        "score_mean": mean(metric_values(rows, "critic/score/mean")),
        "reward_mean": mean(metric_values(rows, "critic/rewards/mean")),
        "any_nonzero_grad": any(value > 0 for value in grad_norms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    for log in args.logs:
        print(json.dumps(summarize(log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

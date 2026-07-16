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
    "critic/acc/mean",
    "critic/overlong_reward/mean",
    "critic/advantages/min",
    "critic/advantages/max",
    "response_length/mean",
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
    advantage_ranges = [
        row["critic/advantages/max"] - row["critic/advantages/min"]
        for row in rows
        if "critic/advantages/max" in row and "critic/advantages/min" in row
    ]
    nonzero_grad_steps = sum(value > 0 for value in grad_norms)
    nonzero_advantage_steps = sum(value > 0 for value in advantage_ranges)

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
        "accuracy_mean": mean(metric_values(rows, "critic/acc/mean")),
        "overlong_reward_mean": mean(metric_values(rows, "critic/overlong_reward/mean")),
        "response_length_mean": mean(metric_values(rows, "response_length/mean")),
        "nonzero_grad_steps": nonzero_grad_steps,
        "nonzero_advantage_steps": nonzero_advantage_steps,
        "valid_training_signal": len(rows) == 10
        and (nonzero_grad_steps > 0 or nonzero_advantage_steps > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    for log in args.logs:
        print(json.dumps(summarize(log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

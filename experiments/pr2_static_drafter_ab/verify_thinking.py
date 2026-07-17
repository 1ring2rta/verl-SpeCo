#!/usr/bin/env python3
"""Assert that the pinned Qwen3.5 rollout prompt opens a thinking block."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def prompt_record(prompt: str) -> dict[str, Any]:
    return {
        "bytes": len(prompt.encode()),
        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "suffix": prompt[-96:],
    }


def main() -> int:
    report: dict[str, Any] = {
        "enable_thinking": True,
        "errors": [],
    }
    errors: list[str] = report["errors"]
    model_value = os.environ.get("MODEL_PATH")
    if not model_value:
        errors.append("MODEL_PATH is not set")
    else:
        model_path = Path(model_value).expanduser().resolve()
        report["model_path"] = str(model_path)
        if not model_path.is_dir():
            errors.append(f"MODEL_PATH is not a local directory: {model_path}")

    if not errors:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from transformers import AutoProcessor
            from verl.utils.chat_template import apply_chat_template

            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            messages = [{"role": "user", "content": "Thinking-mode smoke test."}]
            thinking_prompt = apply_chat_template(
                processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            nonthinking_prompt = apply_chat_template(
                processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            report["processor_class"] = type(processor).__name__
            report["thinking_prompt"] = prompt_record(thinking_prompt)
            report["nonthinking_prompt"] = prompt_record(nonthinking_prompt)
            if not thinking_prompt.endswith("<think>\n"):
                errors.append(
                    "enable_thinking=True did not render an open <think> block"
                )
            if not nonthinking_prompt.endswith("<think>\n\n</think>\n\n"):
                errors.append(
                    "enable_thinking=False did not render a closed empty <think> block"
                )
            if thinking_prompt == nonthinking_prompt:
                errors.append("thinking and non-thinking prompts are identical")
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"could not render Qwen3.5 prompts: {type(exc).__name__}: {exc}"
            )

    report["status"] = "ok" if not errors else "error"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

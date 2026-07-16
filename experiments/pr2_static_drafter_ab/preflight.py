#!/usr/bin/env python3
"""Validate and fingerprint the runtime used by the PR 2 A/B experiment."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_VERL_REVISION = "7aed6b230776f963fa09509c10d9c3a767d1102c"
PR2_COMMIT = "14a54ce51fbf068e7d8d2ab4a4bb821879300305"
REFERENCE_SGLANG_REVISION = "f08726fd56c7ff6d8bd258f1545f98148fa4ef58"
REFERENCE_SGLANG_DIFF_SHA256 = (
    "5839e45bcf4c6fc85f11385866fe7f1b00bbe973aba941ec5cf7bf4415eb5b71"
)
REFERENCE_DFLASH_TIMING_SHA256 = (
    "567cbdf73c6476cfa6aea143bba0f4d2202f7df6d9955cf350771494bbf472c2"
)
REFERENCE_SGLANG_UNTRACKED_RUNTIME_SHA256 = (
    "5f2abdd884b7fa935503c27f8e21dc501a54d976edaa337c6b0fbd533705fc61"
)
EXPECTED_PACKAGE_VERSIONS = {
    "pyarrow": "24.0.0",
    "ray": "2.55.1",
    "safetensors": "0.8.0",
    "sglang": "0.5.13",
    "socksio": "1.0.0",
    "tensordict": "0.10.0",
    "torch": "2.9.1+cu128",
    "transformers": "5.3.0",
    "verl": "0.8.0.dev0",
}
REQUIRED_SERVER_ARGS = (
    "disable_custom_all_reduce",
    "enforce_disable_flashinfer_allreduce_fusion",
    "max_mamba_cache_size",
    "max_total_tokens",
    "mamba_scheduler_strategy",
    "page_size",
    "random_seed",
    "speculative_draft_load_format",
)


def require_env(name: str, errors: list[str]) -> Path:
    value = os.environ.get(name)
    if not value:
        errors.append(f"{name} is not set")
        return Path("/") / f"missing-{name.lower()}"
    path = Path(value).expanduser().resolve()
    if not path.exists():
        errors.append(f"{name} does not exist: {path}")
    return path


def run(
    args: list[str], *, binary: bool = False, timeout: int = 30
) -> str | bytes:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        timeout=timeout,
    )
    return completed.stdout


def git_text(root: Path, *args: str) -> str:
    return str(run(["git", "-C", str(root), *args])).strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_record(module_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    return {
        "module": module_name,
        "path": str(Path(module_file).resolve()) if module_file else None,
        "version": distribution_version(distribution_name or module_name),
    }


def ensure_under(
    label: str, module_path: str | None, expected_root: Path, errors: list[str]
) -> None:
    if module_path is None:
        errors.append(f"{label} has no importable file path")
        return
    try:
        Path(module_path).resolve().relative_to(expected_root.resolve())
    except ValueError:
        errors.append(
            f"{label} imported from {module_path}, expected it under {expected_root}"
        )


def input_record(
    name: str, errors: list[str], *, allow_huggingface_id: bool
) -> dict[str, Any]:
    value = os.environ.get(name)
    if not value:
        errors.append(f"{name} is not set")
        return {"value": None, "exists": None, "kind": None}
    path = Path(value).expanduser()
    exists = path.exists()
    if exists:
        record: dict[str, Any] = {
            "value": value,
            "resolved": str(path.resolve()),
            "exists": True,
            "kind": "local",
        }
        if path.is_file():
            record["sha256"] = sha256_file(path)
        return record
    if allow_huggingface_id and not path.is_absolute():
        return {"value": value, "exists": False, "kind": "huggingface_id"}
    errors.append(f"{name} does not exist: {path}")
    return {"value": value, "exists": False, "kind": "missing"}


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    speco_root = require_env("SPECO_ROOT", errors)
    verl_root = require_env("VERL_ROOT", errors)
    sglang_root = require_env("SGLANG_ROOT", errors)
    configured_python = require_env("PYTHON", errors)
    allow_sglang_drift = os.environ.get("ALLOW_SGLANG_DRIFT") == "1"

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": {
            "configured": os.environ.get("PYTHON"),
            "configured_resolved": str(configured_python),
            "executable": sys.executable,
            "executable_resolved": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
            "version": platform.python_version(),
        },
        "roots": {
            "speco": str(speco_root),
            "verl": str(verl_root),
            "sglang": str(sglang_root),
        },
        "inputs": {
            "model_path": input_record(
                "MODEL_PATH", errors, allow_huggingface_id=True
            ),
            "drafter_path": input_record(
                "DRAFTER_PATH", errors, allow_huggingface_id=True
            ),
            "train_file": input_record(
                "TRAIN_FILE", errors, allow_huggingface_id=False
            ),
            "test_file": input_record(
                "TEST_FILE", errors, allow_huggingface_id=False
            ),
        },
    }
    if sys.version_info[:2] != (3, 11):
        errors.append(
            f"Python is {platform.python_version()}, expected the tested 3.11 runtime"
        )
    if configured_python != Path(sys.executable).resolve():
        errors.append(
            f"PYTHON resolves to {configured_python}, but sys.executable is "
            f"{Path(sys.executable).resolve()}"
        )

    packages: dict[str, Any] = {}
    module_specs = (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("ray", "ray"),
        ("pyarrow", "pyarrow"),
        ("safetensors", "safetensors"),
        ("socksio", "socksio"),
        ("tensordict", "tensordict"),
        ("sglang", "sglang"),
        ("verl", "verl"),
        ("verl_speco", "verl-SpeCo"),
    )
    for module_name, distribution_name in module_specs:
        try:
            packages[module_name] = module_record(module_name, distribution_name)
        except Exception as exc:  # noqa: BLE001 - report every broken import together
            packages[module_name] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"could not import {module_name}: {type(exc).__name__}: {exc}")
    report["packages"] = packages
    for module_name, expected_version in EXPECTED_PACKAGE_VERSIONS.items():
        actual_version = packages.get(module_name, {}).get("version")
        if actual_version != expected_version:
            errors.append(
                f"{module_name} is {actual_version!r}, expected {expected_version!r}"
            )

    ensure_under(
        "verl", packages.get("verl", {}).get("path"), verl_root / "verl", errors
    )
    ensure_under(
        "verl_speco",
        packages.get("verl_speco", {}).get("path"),
        speco_root / "verl_speco",
        errors,
    )
    ensure_under(
        "sglang",
        packages.get("sglang", {}).get("path"),
        sglang_root / "python" / "sglang",
        errors,
    )

    git_report: dict[str, Any] = {}
    empty_diff_sha256 = sha256_bytes(b"")
    try:
        speco_revision = git_text(speco_root, "rev-parse", "HEAD")
        speco_top_level = Path(
            git_text(speco_root, "rev-parse", "--show-toplevel")
        ).resolve()
        speco_diff = bytes(
            run(
                ["git", "-C", str(speco_root), "diff", "--binary", "HEAD"],
                binary=True,
            )
        )
        speco_diff_sha256 = sha256_bytes(speco_diff)
        contains_pr2 = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(speco_root),
                    "merge-base",
                    "--is-ancestor",
                    PR2_COMMIT,
                    "HEAD",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        git_report["speco"] = {
            "revision": speco_revision,
            "top_level": str(speco_top_level),
            "tracked_diff_sha256": speco_diff_sha256,
            "tracked_clean": speco_diff_sha256 == empty_diff_sha256,
            "contains_pr2_commit": contains_pr2,
        }
        if speco_top_level != speco_root:
            errors.append(
                f"SPECO_ROOT is {speco_root}, but its Git top level is {speco_top_level}"
            )
        if speco_diff_sha256 != empty_diff_sha256:
            errors.append("SpeCo has tracked changes; run from a clean experiment commit")
        if not contains_pr2:
            errors.append(f"SpeCo HEAD does not contain PR 2 commit {PR2_COMMIT}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not fingerprint SpeCo: {type(exc).__name__}: {exc}")

    try:
        verl_revision = git_text(verl_root, "rev-parse", "HEAD")
        verl_top_level = Path(
            git_text(verl_root, "rev-parse", "--show-toplevel")
        ).resolve()
        verl_diff = bytes(
            run(
                ["git", "-C", str(verl_root), "diff", "--binary", "HEAD"],
                binary=True,
            )
        )
        verl_diff_sha256 = sha256_bytes(verl_diff)
        git_report["verl"] = {
            "revision": verl_revision,
            "top_level": str(verl_top_level),
            "tracked_diff_sha256": verl_diff_sha256,
            "tracked_clean": verl_diff_sha256 == empty_diff_sha256,
        }
        if verl_top_level != verl_root:
            errors.append(
                f"VERL_ROOT is {verl_root}, but its Git top level is {verl_top_level}"
            )
        if verl_revision != EXPECTED_VERL_REVISION:
            errors.append(
                f"VeRL revision is {verl_revision}, expected {EXPECTED_VERL_REVISION}"
            )
        if verl_diff_sha256 != empty_diff_sha256:
            errors.append("VeRL has tracked changes; use the clean pinned checkout")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not fingerprint VeRL: {type(exc).__name__}: {exc}")

    try:
        sglang_revision = git_text(sglang_root, "rev-parse", "HEAD")
        status = git_text(sglang_root, "status", "--short")
        diff = bytes(
            run(
                ["git", "-C", str(sglang_root), "diff", "--binary", "HEAD"],
                binary=True,
            )
        )
        status_lines = status.splitlines() if status else []
        untracked_paths = git_text(
            sglang_root, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        untracked_runtime_paths = sorted(
            path
            for path in untracked_paths
            if path.startswith("python/sglang/") and path.endswith(".py")
        )
        untracked_runtime_manifest = "".join(
            f"{sha256_file(sglang_root / path)}  {path}\n"
            for path in untracked_runtime_paths
        )
        timing_path = (
            sglang_root
            / "python"
            / "sglang"
            / "srt"
            / "speculative"
            / "dflash_timing.py"
        )
        diff_sha256 = sha256_bytes(diff)
        timing_sha256 = sha256_file(timing_path)
        revision_matches = sglang_revision == REFERENCE_SGLANG_REVISION
        diff_matches = diff_sha256 == REFERENCE_SGLANG_DIFF_SHA256
        timing_matches = timing_sha256 == REFERENCE_DFLASH_TIMING_SHA256
        untracked_runtime_sha256 = sha256_bytes(untracked_runtime_manifest.encode())
        untracked_runtime_matches = (
            untracked_runtime_sha256
            == REFERENCE_SGLANG_UNTRACKED_RUNTIME_SHA256
        )
        git_report["sglang"] = {
            "revision": sglang_revision,
            "reference_revision_match": revision_matches,
            "tracked_diff_sha256": diff_sha256,
            "reference_tracked_diff_match": diff_matches,
            "status_sha256": sha256_bytes(status.encode()),
            "status_entries": len(status_lines),
            "untracked_entries": sum(line.startswith("??") for line in status_lines),
            "dflash_timing_sha256": timing_sha256,
            "reference_dflash_timing_match": timing_matches,
            "untracked_runtime_python_files": untracked_runtime_paths,
            "untracked_runtime_python_manifest_sha256": untracked_runtime_sha256,
            "reference_untracked_runtime_match": untracked_runtime_matches,
        }
        drift_messages = []
        if not revision_matches:
            drift_messages.append("SGLang base revision differs from the reference runtime")
        if not diff_matches:
            drift_messages.append("SGLang tracked diff differs from the reference runtime")
        if not timing_matches:
            drift_messages.append("dflash_timing.py differs from the reference runtime")
        if not untracked_runtime_matches:
            drift_messages.append(
                "untracked SGLang runtime Python files differ from the reference runtime"
            )
        if allow_sglang_drift:
            warnings.extend(drift_messages)
        else:
            errors.extend(drift_messages)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not fingerprint SGLang: {type(exc).__name__}: {exc}")
    report["git"] = git_report

    contract: dict[str, Any] = {}
    try:
        from sglang.srt.server_args import ServerArgs

        field_names = set(getattr(ServerArgs, "__dataclass_fields__", {}))
        if not field_names:
            field_names = set(getattr(ServerArgs, "__annotations__", {}))
        missing_fields = sorted(set(REQUIRED_SERVER_ARGS) - field_names)
        contract["required_server_args"] = list(REQUIRED_SERVER_ARGS)
        contract["missing_server_args"] = missing_fields
        if missing_fields:
            errors.append(f"SGLang ServerArgs is missing: {', '.join(missing_fields)}")

        scheduler_path = (
            sglang_root / "python" / "sglang" / "srt" / "managers" / "scheduler.py"
        )
        scheduler_text = scheduler_path.read_text()
        contract["draft_loader_override_present"] = all(
            marker in scheduler_text
            for marker in (
                "speculative_draft_load_format",
                "Using draft model load_format",
            )
        )
        if not contract["draft_loader_override_present"]:
            errors.append("SGLang scheduler does not expose the draft loader override")
    except Exception as exc:  # noqa: BLE001
        contract["error"] = f"{type(exc).__name__}: {exc}"
        errors.append(f"could not validate SGLang contract: {type(exc).__name__}: {exc}")
    report["sglang_contract"] = contract

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_report: dict[str, Any] = {
            "available": cuda_available,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            cuda_report["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.device_count() < 2:
                errors.append(
                    f"only {torch.cuda.device_count()} CUDA device(s) are visible; expected 2"
                )
        else:
            errors.append("torch.cuda.is_available() is false")
        if torch.version.cuda != "12.8":
            errors.append(
                f"PyTorch CUDA runtime is {torch.version.cuda!r}, expected '12.8'"
            )
        try:
            cuda_report["nvidia_smi"] = str(
                run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,driver_version",
                        "--format=csv,noheader",
                    ]
                )
            ).strip().splitlines()
        except Exception as exc:  # noqa: BLE001
            cuda_report["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
        report["cuda"] = cuda_report
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not validate CUDA: {type(exc).__name__}: {exc}")

    try:
        importlib.import_module("flash_attn.bert_padding")
        report["flash_attn_bert_padding_available"] = True
    except Exception:
        # This environment uses SDPA and disables remove-padding, so FA2's
        # legacy bert_padding module is informative rather than mandatory.
        report["flash_attn_bert_padding_available"] = False

    report["status"] = "ok" if not errors else "error"
    report["errors"] = errors
    report["warnings"] = warnings
    report["allow_sglang_drift"] = allow_sglang_drift
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

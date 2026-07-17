#!/usr/bin/env python3
"""Validate and fingerprint the runtime used by the PR 2 A/B experiment."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any


EXPECTED_VERL_REVISION = "7aed6b230776f963fa09509c10d9c3a767d1102c"
PR2_COMMIT = "14a54ce51fbf068e7d8d2ab4a4bb821879300305"
EXPECTED_TARGET_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_DRAFTER_REVISION = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"
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


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def env_nonnegative_int(name: str, default: int, errors: list[str]) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        errors.append(f"{name} must be a non-negative integer, got {raw_value!r}")
        return default
    if value < 0:
        errors.append(f"{name} must be non-negative, got {value}")
        return default
    return value


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_record(module_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    spec = PathFinder.find_spec(module_name, sys.path)
    if spec is None:
        raise ModuleNotFoundError(module_name)
    module_path = spec.origin
    if module_path is None and spec.submodule_search_locations:
        module_path = next(iter(spec.submodule_search_locations), None)
    return {
        "module": module_name,
        "path": str(Path(module_path).resolve()) if module_path else None,
        "version": distribution_version(distribution_name or module_name),
    }


def python_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"could not find literal assignment {name!r} in {path}")


def class_annotated_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
    raise ValueError(f"could not find class {class_name!r} in {path}")


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
    name: str,
    errors: list[str],
    *,
    allow_huggingface_id: bool,
    expected_revision: str | None = None,
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
        if expected_revision is not None:
            validate_local_model(path, expected_revision, record, errors)
        return record
    if allow_huggingface_id and not path.is_absolute():
        return {"value": value, "exists": False, "kind": "huggingface_id"}
    errors.append(f"{name} does not exist: {path}")
    return {"value": value, "exists": False, "kind": "missing"}


def validate_local_model(
    path: Path,
    expected_revision: str,
    record: dict[str, Any],
    errors: list[str],
) -> None:
    if not path.is_dir():
        errors.append(f"model path is not a directory: {path}")
        return

    config_path = path / "config.json"
    metadata_path = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    incomplete_files = sorted(
        str(candidate.relative_to(path))
        for candidate in (path / ".cache" / "huggingface" / "download").rglob(
            "*.incomplete"
        )
    )
    revision = None
    if metadata_path.is_file():
        metadata_lines = metadata_path.read_text().splitlines()
        revision = metadata_lines[0] if metadata_lines else None

    index_path = path / "model.safetensors.index.json"
    expected_weight_files: set[str] = set()
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        expected_weight_files = set(index.get("weight_map", {}).values())
    else:
        expected_weight_files = {
            candidate.name for candidate in path.glob("*.safetensors")
        }
    missing_weight_files = sorted(
        filename for filename in expected_weight_files if not (path / filename).is_file()
    )
    weight_files = sorted(
        candidate.name for candidate in path.glob("*.safetensors") if candidate.is_file()
    )
    weight_bytes = sum((path / filename).stat().st_size for filename in weight_files)
    weight_records: dict[str, Any] = {}
    for filename in weight_files:
        weight_metadata_path = (
            path
            / ".cache"
            / "huggingface"
            / "download"
            / f"{filename}.metadata"
        )
        metadata_lines = (
            weight_metadata_path.read_text().splitlines()
            if weight_metadata_path.is_file()
            else []
        )
        weight_revision = metadata_lines[0] if metadata_lines else None
        weight_records[filename] = {
            "bytes": (path / filename).stat().st_size,
            "etag": metadata_lines[1] if len(metadata_lines) > 1 else None,
            "revision": weight_revision,
        }
        if weight_revision != expected_revision:
            errors.append(
                f"weight {filename} at {path} has revision {weight_revision!r}, "
                f"expected {expected_revision}"
            )

    record["model"] = {
        "config_sha256": sha256_file(config_path),
        "download_revision": revision,
        "expected_revision": expected_revision,
        "incomplete_files": incomplete_files,
        "missing_weight_files": missing_weight_files,
        "weight_bytes": weight_bytes,
        "weight_files": weight_files,
        "weights": weight_records,
    }
    if not config_path.is_file():
        errors.append(f"model config is missing: {config_path}")
    if revision != expected_revision:
        errors.append(
            f"model at {path} has revision {revision!r}, expected {expected_revision}"
        )
    if incomplete_files:
        errors.append(f"model download is incomplete at {path}: {incomplete_files}")
    if not expected_weight_files:
        errors.append(f"no Safetensors weights were found at {path}")
    if missing_weight_files:
        errors.append(f"model weight shards are missing at {path}: {missing_weight_files}")


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
        "offline": {
            "hf_hub": os.environ.get("HF_HUB_OFFLINE"),
            "transformers": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "inputs": {
            "model_path": input_record(
                "MODEL_PATH",
                errors,
                allow_huggingface_id=True,
                expected_revision=EXPECTED_TARGET_REVISION,
            ),
            "drafter_path": input_record(
                "DRAFTER_PATH",
                errors,
                allow_huggingface_id=True,
                expected_revision=EXPECTED_DRAFTER_REVISION,
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
        except Exception as exc:  # noqa: BLE001 - report every broken path together
            packages[module_name] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"could not resolve {module_name}: {type(exc).__name__}: {exc}")
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
        server_args_path = sglang_root / "python" / "sglang" / "srt" / "server_args.py"
        field_names = class_annotated_fields(server_args_path, "ServerArgs")
        missing_fields = sorted(set(REQUIRED_SERVER_ARGS) - field_names)
        contract["required_server_args"] = list(REQUIRED_SERVER_ARGS)
        contract["missing_server_args"] = missing_fields
        if missing_fields:
            errors.append(f"SGLang ServerArgs is missing: {', '.join(missing_fields)}")

        scheduler_path = (
            sglang_root / "python" / "sglang" / "srt" / "managers" / "scheduler.py"
        )
        scheduler_text = scheduler_path.read_text()
        scheduler_normalized = ast.unparse(
            ast.parse(scheduler_text, filename=str(scheduler_path))
        )
        contract["draft_loader_override_present"] = all(
            marker in scheduler_normalized
            for marker in (
                "if self.server_args.speculative_draft_load_format is not None:",
                "self.server_args.load_format = self.server_args.speculative_draft_load_format",
                "Using draft model load_format",
            )
        )
        if not contract["draft_loader_override_present"]:
            errors.append("SGLang scheduler does not expose the draft loader override")
    except Exception as exc:  # noqa: BLE001
        contract["error"] = f"{type(exc).__name__}: {exc}"
        errors.append(f"could not validate SGLang contract: {type(exc).__name__}: {exc}")
    report["sglang_contract"] = contract

    require_idle_gpus = env_flag("PREFLIGHT_REQUIRE_IDLE_GPUS")
    max_idle_memory_mib = env_nonnegative_int(
        "PREFLIGHT_MAX_IDLE_MEMORY_MIB", 1024, errors
    )
    cuda_report: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_idle_memory_mib": max_idle_memory_mib,
        "require_idle_gpus": require_idle_gpus,
    }
    try:
        torch_path = Path(packages["torch"]["path"]).parent
        torch_version_path = torch_path / "version.py"
        torch_build = {
            "version": python_assignment(torch_version_path, "__version__"),
            "cuda": python_assignment(torch_version_path, "cuda"),
            "git_version": python_assignment(torch_version_path, "git_version"),
        }
        cuda_report["torch_build"] = torch_build
        if torch_build["version"] != EXPECTED_PACKAGE_VERSIONS["torch"]:
            errors.append(
                f"torch/version.py reports {torch_build['version']!r}, expected "
                f"{EXPECTED_PACKAGE_VERSIONS['torch']!r}"
            )
        if torch_build["cuda"] != "12.8":
            errors.append(
                f"PyTorch CUDA runtime is {torch_build['cuda']!r}, expected '12.8'"
            )

        smi_text = str(
            run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        physical_devices = [
            {
                "index": row[0].strip(),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_used_mib": int(row[4].strip()),
                "memory_total_mib": int(row[5].strip()),
            }
            for row in csv.reader(smi_text.splitlines(), skipinitialspace=True)
            if len(row) == 6
        ]
        visible_setting = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_setting is None or not visible_setting.strip():
            visible_devices = physical_devices
        elif visible_setting.strip() == "-1":
            visible_devices = []
        else:
            visible_devices = []
            for token in (part.strip() for part in visible_setting.split(",")):
                matches = [
                    device
                    for device in physical_devices
                    if device["index"] == token or device["uuid"].startswith(token)
                ]
                if len(matches) != 1:
                    errors.append(f"could not map CUDA_VISIBLE_DEVICES token {token!r}")
                else:
                    visible_devices.append(matches[0])

        cuda_report.update(
            {
                "available": bool(visible_devices),
                "device_count": len(visible_devices),
                "devices": visible_devices,
                "physical_devices": physical_devices,
            }
        )
        if len(visible_devices) != 2:
            errors.append(
                f"{len(visible_devices)} CUDA device(s) are visible; expected exactly 2"
            )
        distinct_visible_uuids = {device["uuid"] for device in visible_devices}
        if len(distinct_visible_uuids) != 2:
            errors.append(
                "CUDA_VISIBLE_DEVICES must resolve to exactly 2 distinct GPUs, got "
                f"{sorted(distinct_visible_uuids)}"
            )
        non_h800 = [
            device["name"] for device in visible_devices if "H800" not in device["name"]
        ]
        if non_h800:
            errors.append(f"expected H800 GPUs, found: {non_h800}")

        compute_text = str(
            run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        compute_processes = []
        malformed_compute_rows = []
        for row in csv.reader(compute_text.splitlines(), skipinitialspace=True):
            if len(row) != 4:
                malformed_compute_rows.append(row)
                continue
            try:
                used_memory_mib = int(row[3].strip())
            except ValueError:
                malformed_compute_rows.append(row)
                continue
            compute_processes.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": row[1].strip(),
                    "process_name": row[2].strip(),
                    "used_memory_mib": used_memory_mib,
                }
            )
        cuda_report["malformed_compute_rows"] = malformed_compute_rows
        if malformed_compute_rows:
            message = f"could not parse nvidia-smi compute rows: {malformed_compute_rows}"
            if require_idle_gpus:
                errors.append(message)
            else:
                warnings.append(message)
        visible_compute_processes = [
            process
            for process in compute_processes
            if process["gpu_uuid"] in distinct_visible_uuids
        ]
        cuda_report["compute_processes"] = visible_compute_processes
        if require_idle_gpus and visible_compute_processes:
            process_summary = ", ".join(
                f"pid={process['pid']} gpu={process['gpu_uuid']} "
                f"memory={process['used_memory_mib']} MiB"
                for process in visible_compute_processes
            )
            errors.append(
                "visible GPUs are not idle; stop the unrelated compute processes "
                f"before starting this arm: {process_summary}"
            )
        high_memory_devices = [
            device
            for device in visible_devices
            if device["memory_used_mib"] > max_idle_memory_mib
        ]
        if require_idle_gpus and high_memory_devices:
            memory_summary = ", ".join(
                f"gpu={device['uuid']} memory={device['memory_used_mib']} MiB"
                for device in high_memory_devices
            )
            errors.append(
                "visible GPUs exceed the idle-memory ceiling "
                f"({max_idle_memory_mib} MiB): {memory_summary}"
            )
    except Exception as exc:  # noqa: BLE001
        cuda_report["error"] = f"{type(exc).__name__}: {exc}"
        errors.append(f"could not validate CUDA: {type(exc).__name__}: {exc}")
    report["cuda"] = cuda_report

    flash_attn_spec = PathFinder.find_spec("flash_attn", sys.path)
    flash_attn_locations = (
        list(flash_attn_spec.submodule_search_locations or [])
        if flash_attn_spec is not None
        else []
    )
    report["flash_attn_bert_padding_available"] = any(
        (Path(location) / "bert_padding.py").is_file()
        or (Path(location) / "bert_padding" / "__init__.py").is_file()
        for location in flash_attn_locations
    )

    report["status"] = "ok" if not errors else "error"
    report["errors"] = errors
    report["warnings"] = warnings
    report["allow_sglang_drift"] = allow_sglang_drift
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

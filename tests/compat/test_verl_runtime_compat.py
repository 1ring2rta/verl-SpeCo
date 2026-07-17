from __future__ import annotations

from types import SimpleNamespace

from verl_speco.integration import compat


def test_imported_verl_commit_uses_the_actual_package_source(
    monkeypatch, tmp_path
) -> None:
    repository = tmp_path / "verl-checkout"
    package = repository / "verl"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.touch()

    monkeypatch.setattr(
        compat.PathFinder,
        "find_spec",
        lambda name, path: SimpleNamespace(origin=str(package_file)),
    )

    commit = "7aed6b230776f963fa09509c10d9c3a767d1102c"
    completed = SimpleNamespace(stdout=f"{repository}\n{commit}\n")
    monkeypatch.setattr(compat.subprocess, "run", lambda *args, **kwargs: completed)

    assert compat._read_imported_verl_commit() == commit


def test_resolve_accepts_pinned_import_only_checkout(monkeypatch) -> None:
    commit = "7aed6b230776f963fa09509c10d9c3a767d1102c"
    monkeypatch.setattr(compat, "_read_imported_verl_version", lambda: "0.8.0.dev0")
    monkeypatch.setattr(compat, "_read_imported_verl_commit", lambda: commit)
    monkeypatch.setattr(compat, "_read_distribution_commit", lambda: None)
    monkeypatch.delenv(compat.ALLOW_UNSUPPORTED_ENV, raising=False)

    result = compat.resolve_verl_compatibility()

    assert result.supported is True
    assert result.commit_id == commit
    assert result.reason == "matched commit"


def test_resolve_rejects_unpinned_import_only_checkout(monkeypatch) -> None:
    monkeypatch.setattr(compat, "_read_imported_verl_version", lambda: "0.8.0.dev0")
    monkeypatch.setattr(compat, "_read_imported_verl_commit", lambda: "deadbeef")
    monkeypatch.setattr(compat, "_read_distribution_commit", lambda: None)
    monkeypatch.delenv(compat.ALLOW_UNSUPPORTED_ENV, raising=False)

    result = compat.resolve_verl_compatibility()

    assert result.supported is False
    assert result.commit_id == "deadbeef"

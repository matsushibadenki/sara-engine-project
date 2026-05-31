import os

import pytest

from sara_engine.utils import project_paths


def test_ensure_allowed_output_path_accepts_managed_child(tmp_path, monkeypatch):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    monkeypatch.setattr(project_paths, "ALLOWED_OUTPUT_ROOTS", (str(allowed_root),))

    resolved = project_paths.ensure_allowed_output_path(
        str(allowed_root / "reports" / "summary.json")
    )

    assert resolved == os.path.realpath(str(allowed_root / "reports" / "summary.json"))


def test_ensure_allowed_output_path_rejects_symlink_escape(tmp_path, monkeypatch):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    symlink_path = allowed_root / "escape"
    symlink_path.symlink_to(outside_root, target_is_directory=True)
    monkeypatch.setattr(project_paths, "ALLOWED_OUTPUT_ROOTS", (str(allowed_root),))

    with pytest.raises(ValueError):
        project_paths.ensure_allowed_output_path(str(symlink_path / "leak.json"))

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "research_checkpoint_release.py"
SPEC = importlib.util.spec_from_file_location("research_checkpoint_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_read_sums = MODULE._read_sums


def test_sha256sums_requires_exact_package_files(tmp_path: Path):
    artifact = "sara-example-pretest-v1.sara"
    digest = "a" * 64
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  {artifact}\n{digest}  manifest.json\n")
    assert _read_sums(sums, artifact) == {artifact: digest, "manifest.json": digest}

    sums.write_text(f"{digest}  ../{artifact}\n{digest}  manifest.json\n")
    with pytest.raises(ValueError, match="invalid"):
        _read_sums(sums, artifact)


def test_sha256sums_rejects_non_hex_digest(tmp_path: Path):
    artifact = "sara-example-pretest-v1.sara"
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'z' * 64}  {artifact}\n{'a' * 64}  manifest.json\n")
    with pytest.raises(ValueError, match="digest"):
        _read_sums(sums, artifact)

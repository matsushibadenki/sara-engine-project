import os
import re


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def test_pyproject_and_cargo_versions_match():
    pyproject = _read_text(os.path.join(PROJECT_ROOT, "pyproject.toml"))
    cargo = _read_text(os.path.join(PROJECT_ROOT, "Cargo.toml"))

    pyproject_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cargo_version = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE)

    assert pyproject_version is not None
    assert cargo_version is not None
    assert pyproject_version.group(1) == cargo_version.group(1)


def test_pyproject_declares_expected_console_scripts():
    pyproject = _read_text(os.path.join(PROJECT_ROOT, "pyproject.toml"))

    assert 'sara-chat = "sara_engine.cli:chat"' in pyproject
    assert 'sara-train = "sara_engine.cli:train"' in pyproject

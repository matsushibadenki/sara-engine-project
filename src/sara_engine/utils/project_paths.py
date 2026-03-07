import os
from typing import Iterable


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "workspace")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ALLOWED_OUTPUT_ROOTS = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    WORKSPACE_DIR,
    MODELS_DIR,
)


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def raw_data_path(*parts: str) -> str:
    return os.path.join(RAW_DATA_DIR, *parts)


def processed_data_path(*parts: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, *parts)


def interim_data_path(*parts: str) -> str:
    return os.path.join(INTERIM_DATA_DIR, *parts)


def workspace_path(*parts: str) -> str:
    return os.path.join(WORKSPACE_DIR, *parts)


def model_path(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts)


def resolve_project_relative(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _is_within(path: str, roots: Iterable[str]) -> bool:
    abs_path = os.path.abspath(path)
    for root in roots:
        try:
            if os.path.commonpath([abs_path, root]) == root:
                return True
        except ValueError:
            continue
    return False


def ensure_allowed_output_path(path: str) -> str:
    resolved = resolve_project_relative(path)
    if not _is_within(resolved, ALLOWED_OUTPUT_ROOTS):
        allowed = ", ".join(ALLOWED_OUTPUT_ROOTS)
        raise ValueError(
            f"Output path must be under one of: {allowed}. Received: {resolved}"
        )
    return resolved


def ensure_output_directory(path: str) -> str:
    resolved = ensure_allowed_output_path(path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def ensure_parent_directory(path: str) -> str:
    resolved = ensure_allowed_output_path(path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved

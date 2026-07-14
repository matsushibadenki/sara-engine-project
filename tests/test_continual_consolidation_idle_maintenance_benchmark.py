import importlib.util
import json
import os
import sys
import types
import pytest


def _load_script_with_msgpack_stub(script_name: str):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", script_name)
    )
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    msgpack_stub = types.SimpleNamespace()
    transformers_stub = types.SimpleNamespace()

    def _pack(payload, handle):
        handle.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _unpack(handle, raw=False):
        del raw
        return json.loads(handle.read().decode("utf-8"))

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    msgpack_stub.pack = _pack
    msgpack_stub.unpack = _unpack
    transformers_stub.AutoTokenizer = _AutoTokenizer
    original = sys.modules.get("msgpack")
    original_transformers = sys.modules.get("transformers")
    isolated_modules = (
        "sara_engine.inference",
        "scripts.utils.memory_health",
        "scripts.utils.upgrade_memory",
    )
    original_isolated = {name: sys.modules.get(name) for name in isolated_modules}
    for name in isolated_modules:
        sys.modules.pop(name, None)
    sys.modules["msgpack"] = msgpack_stub
    sys.modules["transformers"] = transformers_stub
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original is None:
            sys.modules.pop("msgpack", None)
        else:
            sys.modules["msgpack"] = original
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers
        for name in isolated_modules:
            sys.modules.pop(name, None)
            if original_isolated[name] is not None:
                sys.modules[name] = original_isolated[name]


def test_continual_consolidation_benchmark_exposes_idle_maintenance_trace():
    try:
        module = _load_script_with_msgpack_stub("continual_consolidation_benchmark.py")
    except (ModuleNotFoundError, TypeError) as exc:
        pytest.skip(f"benchmark import unavailable in current test environment: {exc}")
    report = module.run_continual_consolidation_benchmark()

    assert report["metrics"]["idle_maintenance_trace_integrity_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_phase_alignment_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_cache_refresh_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_multimodal_bundle_visibility_observed"] == 1.0

    maintenance_case = report["details"]["test_results"][-1]
    loop_report = maintenance_case["idle_consolidation_loop_report"]
    assert loop_report["sleep_consolidation_report"]["observed_only"] is True
    assert loop_report["memory_phase_report"]["observed_only"] is True
    assert loop_report["delta_retention_policy_report"]["observed_only"] is True
    assert maintenance_case["selected_count"] >= 1
    assert maintenance_case["refresh_count"] >= 1
    assert "multimodal_bundle_summary" in loop_report

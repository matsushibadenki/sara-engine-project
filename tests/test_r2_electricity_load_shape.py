"""Shape audit treats every load cell as opaque, including sealed years."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/r2_electricity_load_shape.py"
SPEC = importlib.util.spec_from_file_location("electricity_shape", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_opaque_load_cells_are_never_converted_or_reported():
    rows = iter([
        '"";"clientA";"clientB"\n',
        '"2014-01-01 00:00:00";SEALED_NOT_A_NUMBER;SECRET\n',
        '"2014-01-01 00:15:00";OTHER_SECRET;OPAQUE\n',
    ])
    result = MODULE.audit_lines(rows, client_count=2, timestamp_count=2, interval_minutes=15)
    assert result["passed"] is True
    assert result["load_cells_parsed"] == 0
    assert "SECRET" not in str(result)
    assert result["outcome_support_checked"] is False


def test_shape_order_gaps_and_duplicate_clients_fail():
    rows = iter([
        '"";"same";"same"\n',
        '"2013-01-01 00:30:00";opaque;opaque\n',
        '"2013-01-01 00:00:00";opaque;opaque\n',
        'bad timestamp;opaque\n',
    ])
    result = MODULE.audit_lines(rows, client_count=2, timestamp_count=3, interval_minutes=15)
    assert result["passed"] is False
    assert result["duplicate_or_regressive_count"] == 1
    assert result["interval_violation_count"] == 1
    assert result["malformed_row_count"] == 1
    assert result["checks"]["unique_client_columns"] is False

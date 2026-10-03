"""An interrupted diagnostic remains consumed, with a durable failure record."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest
from sara_engine.utils.project_paths import workspace_path


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/eval'
SPEC = importlib.util.spec_from_file_location('electricity_v2_runner', SCRIPTS / 'r2_electricity_development_v2.py')
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPTS))


def test_failed_attempt_cannot_be_silently_retried():
    with TemporaryDirectory(dir=workspace_path(), prefix='electricity_receipt_test_') as directory:
        output = Path(directory) / 'report.json'
        prefix = Path(directory) / 'attempt'
        def fail():
            raise ValueError('SECRET_LOAD_VALUE')
        with pytest.raises(ValueError):
            MODULE.execute_once(output, prefix, fail, {'confirmatory': False})
        failure = json.loads(Path(str(prefix) + '_failed.json').read_text())
        assert failure['attempt_consumed']
        assert failure['error_type'] == 'ValueError'
        assert 'SECRET' not in str(failure)
        assert output.read_text() == ''
        with pytest.raises(FileExistsError):
            MODULE.execute_once(output, prefix, lambda: {}, {})

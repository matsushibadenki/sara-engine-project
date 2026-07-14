import pytest

from sara_engine.core.module import SaraModule


class _CoreStateful(SaraModule):
    def __init__(self):
        super().__init__()
        self.register_parameter("value", 1)

    def forward(self, value):
        return value


def test_core_module_strict_state_restore_is_enforced():
    module = _CoreStateful()
    with pytest.raises(ValueError, match="state_dict mismatch"):
        module.load_state_dict({}, strict=True)
    module.load_state_dict({"value": 4}, strict=True)
    assert module.value == 4


def test_core_module_save_rejects_unmanaged_output(tmp_path):
    module = _CoreStateful()
    with pytest.raises(ValueError, match="Output path"):
        module.save(str(tmp_path / "state.json"))

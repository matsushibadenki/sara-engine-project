import pytest

from sara_engine.nn.module import SNNModule


class _Stateful(SNNModule):
    def __init__(self):
        super().__init__()
        self.value = {"count": 1}
        self.register_state("value")

    def forward(self, value):
        return value


def test_strict_state_restore_rejects_missing_and_unexpected_keys():
    module = _Stateful()
    with pytest.raises(ValueError, match="state_dict mismatch"):
        module.load_state_dict({}, strict=True)
    with pytest.raises(ValueError, match="unexpected"):
        module.load_state_dict({"value": {"count": 2}, "extra": 1}, strict=True)


def test_non_strict_state_restore_preserves_compatibility():
    module = _Stateful()
    module.load_state_dict({"value": {"count": 7}}, strict=False)
    assert module.value == {"count": 7}

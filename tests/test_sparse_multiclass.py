from dataclasses import replace

import pytest

from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig


def test_sparse_mistake_updates_only_target_and_predicted_classes():
    learner = BoundedSparseMulticlassReadout(["a", "b", "c"], SparseMulticlassConfig(learning_rate=.3))
    receipt = learner.predict([1, 2])
    update = learner.observe(receipt, "c")
    assert update.updates == 4
    assert len(learner.snapshot()["weights"]) == 4


def test_receipt_identity_and_pending_contract():
    learner = BoundedSparseMulticlassReadout(["a", "b"])
    receipt = learner.predict([1])
    with pytest.raises(ValueError): learner.observe(replace(receipt), "b")
    with pytest.raises(ValueError): learner.predict([1])
    learner.observe(receipt, "b")


def test_frozen_rate_keeps_weights_empty():
    learner = BoundedSparseMulticlassReadout(["a", "b"], SparseMulticlassConfig(learning_rate=0.0))
    receipt = learner.predict([1]); update = learner.observe(receipt, "b")
    assert update.updates == 0 and learner.snapshot()["weights"] == {}

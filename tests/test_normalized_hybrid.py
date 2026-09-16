from dataclasses import replace
from pathlib import Path
import pytest

from sara_engine.learning.normalized_hybrid import BoundedNormalizedHybrid, NormalizedHybridConfig

def _run(model):
    predictions=[]
    stream=[((1,2),("bos","a"),"b"),((2,3),("a","b"),"c"),((1,3),("b","c"),"a")]*12
    for routes,context,target in stream:
        receipt=model.predict(routes,context=context);predictions.append((receipt.predicted,receipt.overridden,receipt.probabilities));model.observe(receipt,target)
    return predictions

def test_explicit_neuron_and_scalar_modes_are_replay_equivalent():
    labels=("a","b","c")
    explicit=BoundedNormalizedHybrid(labels,NormalizedHybridConfig(explicit_neurons=True,minimum_base_support=2))
    scalar=BoundedNormalizedHybrid(labels,NormalizedHybridConfig(explicit_neurons=False,minimum_base_support=2))
    assert _run(explicit)==_run(scalar)

def test_compact_event_mode_matches_explicit_and_uses_no_neuron_objects():
    labels=("a","b","c");config=NormalizedHybridConfig(explicit_neurons=True,compact_event_units=True,minimum_base_support=2)
    compact=BoundedNormalizedHybrid(labels,config);explicit=BoundedNormalizedHybrid(labels,replace(config,compact_event_units=False))
    assert _run(compact)==_run(explicit)
    assert not compact._neurons and compact._compact_units==compact._routes

def test_absolute_routing_mode_is_supported_for_bpi_replay():
    model=BoundedNormalizedHybrid(("a","b"),NormalizedHybridConfig(routing_mode="absolute",minimum_base_support=1,base_probability_cap=1.0,local_score_margin=0.0))
    _run(BoundedNormalizedHybrid(("a","b","c"),NormalizedHybridConfig(routing_mode="absolute",minimum_base_support=1)))
    assert model.config.routing_mode=="absolute"

def test_receipt_identity_and_pending_contract():
    model=BoundedNormalizedHybrid(("a","b"));receipt=model.predict((1,),context=("x",))
    with pytest.raises(ValueError):model.observe(replace(receipt),"a")
    with pytest.raises(ValueError):model.predict((1,),context=("x",))
    model.observe(receipt,"a")

def test_state_round_trip_preserves_next_prediction_and_managed_save():
    model=BoundedNormalizedHybrid(("a","b","c"),NormalizedHybridConfig(explicit_neurons=False,minimum_base_support=2));_run(model)
    restored=BoundedNormalizedHybrid.from_state_dict(model.state_dict())
    left=model.predict((1,2),context=("bos","a"));right=restored.predict((1,2),context=("bos","a"))
    assert left.predicted==right.predicted and left.probabilities==right.probabilities
    model.observe(left,"b");restored.observe(right,"b")
    path=Path(model.save("test-normalized-hybrid.json"));loaded=BoundedNormalizedHybrid.load(path.name)
    assert loaded.state_dict()==model.state_dict();path.unlink()

def test_serialization_rejects_pending_and_unmanaged_filename():
    model=BoundedNormalizedHybrid(("a","b"));receipt=model.predict((1,),context=("x",))
    with pytest.raises(ValueError):model.state_dict()
    model.observe(receipt,"a")
    with pytest.raises(ValueError):model.save("../outside.json")

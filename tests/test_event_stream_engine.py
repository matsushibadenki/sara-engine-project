from pathlib import Path
import json
import pytest

from sara_engine.evaluation.r2_sepsis_hybrid import SepsisRouteEncoder
from sara_engine.evaluation.sepsis_data import load_traces
from sara_engine.learning.event_stream_engine import BoundedEventRouteEncoder,BoundedEventStreamEngine,EventRouteConfig
from sara_engine.learning.normalized_hybrid import NormalizedHybridConfig

def test_generic_encoder_matches_sepsis_route_identity():
    trace=load_traces()[0];generic=BoundedEventRouteEncoder();legacy=SepsisRouteEncoder(spiking=False)
    for index in range(len(trace.events)-1):assert generic.encode(trace.events,index)==legacy.encode(trace,index)

def test_engine_round_trip_preserves_encoder_and_next_prediction():
    traces=load_traces();labels=tuple(sorted({event.activity for trace in traces for event in trace.events}))
    config=NormalizedHybridConfig(max_active=7,max_classes=16,explicit_neurons=True,compact_event_units=True)
    engine=BoundedEventStreamEngine(labels,hybrid_config=config)
    for trace in traces[:12]:
        for index in range(len(trace.events)-1):
            receipt=engine.predict(trace.events,index);engine.observe(receipt,trace.events[index+1].activity)
    restored=BoundedEventStreamEngine.from_state_dict(engine.state_dict());trace=traces[12]
    left=engine.predict(trace.events,0);right=restored.predict(trace.events,0)
    assert left.predicted==right.predicted and left.probabilities==right.probabilities
    engine.observe(left,trace.events[1].activity);restored.observe(right,trace.events[1].activity)
    path=Path(engine.save("test-event-stream-engine.json"));loaded=BoundedEventStreamEngine.load(path.name)
    assert loaded.state_dict()==engine.state_dict();path.unlink()

def test_engine_rejects_insufficient_active_budget_and_pending_save():
    labels=("a","b")
    with pytest.raises(ValueError):BoundedEventStreamEngine(labels,route_config=EventRouteConfig(include_calendar_route=True),hybrid_config=NormalizedHybridConfig(max_active=7))
    trace=load_traces()[0];engine=BoundedEventStreamEngine(tuple(sorted({e.activity for t in load_traces() for e in t.events})),hybrid_config=NormalizedHybridConfig(max_active=7,max_classes=16))
    receipt=engine.predict(trace.events,0)
    with pytest.raises(ValueError):engine.state_dict()
    engine.observe(receipt,trace.events[1].activity)

def test_artifact_checksum_rejects_corruption_and_legacy_state_remains_loadable():
    traces=load_traces();labels=tuple(sorted({event.activity for trace in traces for event in trace.events}));engine=BoundedEventStreamEngine(labels,hybrid_config=NormalizedHybridConfig(max_active=7,max_classes=16))
    receipt=engine.predict(traces[0].events,0);engine.observe(receipt,traces[0].events[1].activity)
    path=Path(engine.save("test-event-stream-corruption.json"));document=json.loads(path.read_text());document["payload"]["hybrid"]["sequence"]+=1;path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="checksum mismatch"):BoundedEventStreamEngine.load(path.name)
    path.write_text(json.dumps(engine.state_dict()));assert BoundedEventStreamEngine.load(path.name).state_dict()==engine.state_dict();path.unlink()

from sara_engine.risa import RisaObservation, RisaPredictionQuery, SARAAlignedRisaKernel


def _toy_observations():
    return [
        RisaObservation(timestamp=1, event_id="e1", actor="dog", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=2, event_id="e2", actor="human", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=3, event_id="e3", actor="horse", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.8),
        RisaObservation(timestamp=4, event_id="e4", actor="dog", action="rest", observed_effects=["fatigue_down"], verified=True, resonance_score=0.6),
        RisaObservation(timestamp=5, event_id="e5", actor="dog", action="run", observed_effects=["fatigue_up"], verified=False, resonance_score=0.9),
    ]


def test_risa_kernel_builds_concept_cells_and_predicts_nearby_actor() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(_toy_observations())

    concept = kernel.state.graph.get_node("concept:shared_run_fatigue_up")
    assert concept is not None
    assert concept.kind == "concept"
    assert concept.dormant is False
    assert kernel.state.concept_members["concept:shared_run_fatigue_up"] == ["dog", "horse", "human"]

    result = kernel.predict(RisaPredictionQuery(actor="wolf", action="run"))

    assert result.predicted_effects == ["fatigue_up"]
    assert result.score > 0.0
    assert any("concept:shared_run_fatigue_up" in path for path in result.supporting_paths)
    assert "e5" not in result.evidence_event_ids


def test_risa_kernel_can_put_cold_concept_cell_to_sleep() -> None:
    kernel = SARAAlignedRisaKernel(
        min_support=2,
        min_distinct_actors=2,
        dormancy_energy_threshold=0.2,
        dormancy_idle_threshold=4,
        connection_cost_rate=0.05,
    )
    kernel.ingest_observations(_toy_observations()[:3])
    kernel.apply_metabolism(current_timestamp=20)

    concept = kernel.state.graph.get_node("concept:shared_run_fatigue_up")
    assert concept is not None
    assert concept.dormant is True


def test_risa_kernel_snapshot_contains_lineage_and_graph() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(_toy_observations()[:3])

    snapshot = kernel.snapshot().state

    assert "graph" in snapshot
    assert "concept_lineage" in snapshot
    assert snapshot["concept_lineage"]["concept:shared_run_fatigue_up"]["verified_support"] == 3

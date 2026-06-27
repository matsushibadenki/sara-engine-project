from sara_engine.ingest import (
    EventMemoryIngestPipeline,
    FrequentSequenceMiner,
    PredictionGainEstimator,
    ProposalVerifier,
    SynchronyDetector,
    make_candidate_event,
)
from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id


def _candidate(record_id: str, time_ms: int, *, confidence: float, evidence_count: int, source_hash: str = "hash-a"):
    return make_candidate_event(
        {
            "record_id": record_id,
            "modality": "vision",
            "label": "visual_cluster_018",
            "local_time_ms": time_ms,
            "confidence": confidence,
            "source_ref": "session-a",
            "source_hash": source_hash,
            "extractor_name": "candidate_proposals",
            "extractor_version": "v1",
            "evidence_count": evidence_count,
            "counterexample_count": 0,
            "prediction_gain": 0.2,
        }
    )


def test_event_memory_pipeline_runs_end_to_end_with_shared_event_surface():
    pipeline = EventMemoryIngestPipeline(
        synchrony_detector=SynchronyDetector(window_ms=80, cross_modal_only=True),
        prediction_gain_estimator=PredictionGainEstimator(min_support=1, min_gain=0.0, max_delay_ms=120),
        verifier=ProposalVerifier(
            min_confidence=0.1,
            min_evidence_count=1,
            min_prediction_gain=0.0,
            max_counterexample_rate=0.9,
        ),
        sequence_miner=FrequentSequenceMiner(min_support_episodes=1, max_pattern_length=3, max_span_ms=160),
    )
    streams = [
        {
            "stream_id": "audio-1",
            "modality": "audio",
            "samples": [
                {"time_ms": 0, "value": 0.0},
                {"time_ms": 100, "value": 0.7},
                {"time_ms": 200, "value": 0.0},
            ],
        },
        {
            "stream_id": "text-1",
            "modality": "text",
            "samples": [
                {"time_ms": 0, "value": 0.0},
                {"time_ms": 130, "value": 0.8},
                {"time_ms": 230, "value": 0.0},
            ],
        },
    ]
    candidates = [
        _candidate("cand-good", 150, confidence=0.9, evidence_count=3),
        _candidate("cand-bad", 170, confidence=0.2, evidence_count=0),
    ]

    result = pipeline.ingest_streams(
        streams,
        source_ref="session-a",
        source_hash="hash-a",
        candidate_events=candidates,
    )

    assert len(result.change_points) >= 2
    assert len(result.observed_events) >= 2
    assert [item.record_id for item in result.accepted_candidate_events] == ["cand-good"]
    assert [item.record_id for item in result.rejected_candidate_events] == ["cand-bad"]
    assert len(result.episodes) == 1
    assert result.episodes[0].candidate_event_ids == ("cand-good",)
    assert len(result.frequent_sequences) >= 1
    assert len(result.candidate_relations) >= 1
    assert len(result.verified_relations) >= 1
    ledger_types = {entry.record_type for entry in result.lineage_ledger}
    assert "observed_event" in ledger_types
    assert "candidate_event" in ledger_types
    assert "candidate_relation" in ledger_types
    assert "verified_relation" in ledger_types
    assert result.to_dict()["traces"]["verification"]["accepted_candidate_event_count"] == 1


def test_event_memory_pipeline_excludes_rejected_candidates_from_episode_surface():
    pipeline = EventMemoryIngestPipeline(
        verifier=ProposalVerifier(min_confidence=0.8, min_evidence_count=2),
        sequence_miner=FrequentSequenceMiner(min_support_episodes=1, max_pattern_length=3, max_span_ms=160),
    )
    streams = [
        {
            "stream_id": "audio-1",
            "modality": "audio",
            "samples": [
                {"time_ms": 0, "value": 0.0},
                {"time_ms": 100, "value": 0.6},
                {"time_ms": 200, "value": 0.0},
            ],
        }
    ]
    candidates = [_candidate("cand-bad", 120, confidence=0.5, evidence_count=1)]

    result = pipeline.ingest_streams(
        streams,
        source_ref="session-a",
        source_hash="hash-a",
        candidate_events=candidates,
    )

    assert result.accepted_candidate_events == ()
    assert result.rejected_candidate_events[0].record_id == "cand-bad"
    assert all(not episode.candidate_event_ids for episode in result.episodes)


def test_event_memory_pipeline_updates_persistent_self_state_from_events_and_hints():
    controller = PersistentSelfStateController(core_event_ids=(101, 202))
    pipeline = EventMemoryIngestPipeline(
        persistent_self_state=controller,
        synchrony_detector=SynchronyDetector(window_ms=80, cross_modal_only=True),
        prediction_gain_estimator=PredictionGainEstimator(min_support=1, min_gain=0.0, max_delay_ms=120),
        verifier=ProposalVerifier(
            min_confidence=0.1,
            min_evidence_count=1,
            min_prediction_gain=0.0,
            max_counterexample_rate=0.9,
        ),
        sequence_miner=FrequentSequenceMiner(min_support_episodes=1, max_pattern_length=3, max_span_ms=160),
    )
    streams = [
        {
            "stream_id": "audio-1",
            "modality": "audio",
            "samples": [
                {"time_ms": 0, "value": 0.0},
                {"time_ms": 100, "value": 0.7},
                {"time_ms": 200, "value": 0.0},
            ],
        }
    ]

    result = pipeline.ingest_streams(
        streams,
        source_ref="session-a",
        source_hash="hash-a",
        reactivation_hints=(
            {
                "entry_id": "verified-anchor",
                "activation": 0.9,
                "mutates_durable_state": False,
            },
        ),
    )

    self_state = result.to_dict()["traces"]["persistent_self_state"]
    assert self_state["current_active_ids"]
    assert self_state["reactivation_hint_count"] == 1
    assert stable_self_state_id("verified-anchor") in self_state["memory_event_ids"]
    assert self_state["external_event_count"] >= 1


def test_event_memory_pipeline_keeps_self_state_alive_during_idle_ingest():
    controller = PersistentSelfStateController(core_event_ids=(333, 444))
    pipeline = EventMemoryIngestPipeline(
        persistent_self_state=controller,
        verifier=ProposalVerifier(min_confidence=0.8, min_evidence_count=2),
        sequence_miner=FrequentSequenceMiner(min_support_episodes=1, max_pattern_length=3, max_span_ms=160),
    )

    first = pipeline.ingest_streams(
        [],
        source_ref="idle-session",
        source_hash="idle-hash",
    )
    second = pipeline.ingest_streams(
        [],
        source_ref="idle-session",
        source_hash="idle-hash",
    )

    first_state = first.to_dict()["traces"]["persistent_self_state"]
    second_state = second.to_dict()["traces"]["persistent_self_state"]
    assert first_state["current_active_ids"]
    assert second_state["current_active_ids"]
    assert second_state["idle_self_state_ok"] is True

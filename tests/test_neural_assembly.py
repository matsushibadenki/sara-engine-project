# Directory Path: tests/test_neural_assembly.py
# English Title: Neural Assembly Tracker Tests
# Purpose/Content: Verifies bounded sparse co-activation tracking for concept assembly formation.

from sara_engine.learning.neural_assembly import NeuralAssemblyTracker


def test_neural_assembly_tracker_detects_stable_noisy_core() -> None:
    tracker = NeuralAssemblyTracker(window_size=8, min_group_size=3, activation_threshold=2)

    tracker.record_step([1, 2, 3, 40])
    tracker.record_step([1, 2, 3, 41])
    tracker.record_step([1, 2, 3, 42])

    active = tracker.get_active_assemblies()
    report = tracker.get_assembly_report()

    assert {1, 2, 3} in active
    assert report["active_assembly_count"] >= 1
    assert report["top_assemblies"][0]["support"] >= 2


def test_neural_assembly_tracker_forgets_stale_groups_in_sliding_window() -> None:
    tracker = NeuralAssemblyTracker(window_size=3, min_group_size=3, activation_threshold=2)

    tracker.record_step([1, 2, 3])
    tracker.record_step([1, 2, 3])
    assert {1, 2, 3} in tracker.get_active_assemblies()

    tracker.record_step([7, 8, 9])
    tracker.record_step([7, 8, 9])
    tracker.record_step([7, 8, 9])

    active = tracker.get_active_assemblies()
    assert {1, 2, 3} not in active
    assert {7, 8, 9} in active


def test_neural_assembly_tracker_keeps_candidate_map_bounded() -> None:
    tracker = NeuralAssemblyTracker(
        window_size=12,
        min_group_size=3,
        max_candidates_per_step=5,
        max_tracked_assemblies=6,
    )

    for offset in range(12):
        tracker.record_step(list(range(offset, offset + 8)))

    report = tracker.get_assembly_report()
    assert report["candidate_count"] <= 6
    assert report["window_count"] == 12

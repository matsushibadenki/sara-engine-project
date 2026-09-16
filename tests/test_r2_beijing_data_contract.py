from scripts.data.audit_r2_beijing_air_quality import build_audit
from scripts.data.prepare_r2_beijing_manifest import build


def test_source_audit_matches_hash_pinned_hourly_dataset():
    audit = build_audit()
    assert audit["station_count"] == 12
    assert audit["total_rows"] == 420768
    assert audit["total_valid_consecutive_pm25_pairs"] == 409180
    assert audit["all_strictly_chronological"] is True
    assert all(station["non_hourly_gaps"] == 0 for station in audit["stations"])


def test_manifest_and_baselines_exclude_test_metrics_before_preregistration():
    manifest, baselines = build()
    assert manifest["eligible_pair_counts"] == {
        "training": 290312, "development": 50825, "frozen_test": 68019,
    }
    assert manifest["frozen_test_period_metrics_computed"] is False
    assert baselines["frozen_test_period_metrics_computed"] is False
    assert baselines["arms"]["station_hour_direction_transition"]["balanced_accuracy"] > 0.55
    assert manifest["transition_table_occupied"] <= manifest["transition_table_capacity"]

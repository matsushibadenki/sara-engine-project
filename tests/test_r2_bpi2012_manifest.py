from scripts.data.prepare_r2_bpi2012_manifest import build, gap_bucket


def test_case_manifest_is_disjoint_and_keeps_test_metrics_sealed():
    manifest, baselines = build()
    assert manifest["case_counts"] == {"training": 9160, "development": 1963, "frozen_test": 1964}
    assert manifest["prediction_counts"] == {"training": 177026, "development": 39393, "frozen_test": 32694}
    assert manifest["case_overlap"] == 0
    assert manifest["frozen_test_metrics_computed"] is False
    assert baselines["frozen_test_metrics_computed"] is False


def test_real_timing_improves_over_constant_gap_control_on_development():
    _, baselines = build()
    assert baselines["arms"]["second_order"]["top1_accuracy"] > 0.80
    assert baselines["timing_aware_minus_constant_gap_accuracy"] > 0.05


def test_gap_buckets_are_bounded_and_monotonic():
    assert [gap_bucket(value) for value in (0, 1, 60, 61, 300, 301, 10**9)] == [0, 1, 1, 2, 2, 3, 8]

from scripts.data.prepare_r2_sepsis_manifest import build

def test_sepsis_manifest_and_baselines_keep_test_sealed():
    manifest, baselines = build()
    assert manifest["case_counts"] == {"training": 735, "development": 157, "frozen_test": 158}
    assert manifest["case_overlap"] == 0 and not manifest["frozen_test_metrics_computed"]
    assert not baselines["frozen_test_metrics_computed"]
    assert baselines["arms"]["second_order"]["count"] == manifest["prediction_counts"]["development"]

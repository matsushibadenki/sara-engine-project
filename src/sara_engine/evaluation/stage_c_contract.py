# Directory Path: src/sara_engine/evaluation/stage_c_contract.py
# English Title: Stage C Meta-Adaptation Readiness Contract
# Purpose/Content: Shared constants for Stage C minimum checks used by phase3 suite, release gate, and soak summaries.

from typing import Dict, List


STAGE_C_MINIMUM_METRIC_NAMES: List[str] = [
    "meta_adaptation_loop",
    "meta_adaptation_parameter_integrity",
    "temporal_self_distillation_stability",
]


STAGE_C_REQUIRED_MINIMUM_CHECKS: Dict[str, str] = {
    "metric.meta_adaptation_loop": "meta-adaptation loop activation",
    "metric.meta_adaptation_parameter_integrity": "meta-adaptation parameter integrity",
    "metric.temporal_self_distillation_stability": "temporal self-distillation stability",
}


def stage_c_metric_check_name(metric_name: str) -> str:
    return f"metric.{metric_name}"

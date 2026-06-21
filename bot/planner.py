from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


@dataclass
class CapabilityGapSignal:
    text_ratio: float
    image_ratio: float
    audio_ratio: float
    video_ratio: float
    binary_ratio: float
    jp_ratio: float
    en_ratio: float


class CollectionPlanner:
    """Decides crawl focus from observed modality balance."""

    def __init__(self) -> None:
        self.default_seeds = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD",
            "https://arxiv.org/list/cs.AI/recent",
        ]
        self.image_seeds = [
            "https://commons.wikimedia.org/wiki/Main_Page",
            "https://www.pexels.com/search/technology/",
        ]
        self.audio_seeds = [
            "https://librivox.org/",
            "https://freemusicarchive.org/",
        ]
        self.jp_text_seeds = [
            "https://ja.wikipedia.org/wiki/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92",
            "https://ja.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86",
        ]
        self.en_text_seeds = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
        ]

    def next_seeds(self, gap: CapabilityGapSignal, blocked_domains: Optional[Set[str]] = None) -> List[str]:
        blocked_domains = blocked_domains or set()
        seeds = list(self.default_seeds)
        if gap.image_ratio < 0.1:
            seeds.extend(self.image_seeds)
        if gap.audio_ratio < 0.05:
            seeds.extend(self.audio_seeds)
        if gap.jp_ratio < 0.35:
            seeds.extend(self.jp_text_seeds)
        if gap.en_ratio < 0.35:
            seeds.extend(self.en_text_seeds)
        filtered: List[str] = []
        for seed in seeds:
            host = seed.split("//", 1)[-1].split("/", 1)[0].lower()
            if host in blocked_domains:
                continue
            filtered.append(seed)
        return filtered

    def material_requests_from_evaluation(self, report: Dict[str, object]) -> List[Dict[str, object]]:
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        requests: List[Dict[str, object]] = []
        if float(metrics.get("real_data_summary_keyword_coverage", 1.0) or 0.0) < 0.7:
            requests.append(
                {
                    "request_id": "weak_summary_coverage",
                    "material_types": ["summary", "source_claim"],
                    "reason": "weak summary coverage",
                    "priority": 0.85,
                }
            )
        if float(metrics.get("negative_control_abstention_integrity", 1.0) or 0.0) < 1.0:
            requests.append(
                {
                    "request_id": "weak_negative_controls",
                    "material_types": ["negative_query"],
                    "reason": "weak negative controls",
                    "priority": 1.0,
                }
            )
        if float(metrics.get("contrastive_control_accuracy", 1.0) or 0.0) < 1.0:
            requests.append(
                {
                    "request_id": "weak_contrastive_controls",
                    "material_types": ["contrastive_pair"],
                    "reason": "weak contrastive controls",
                    "priority": 1.0,
                }
            )
        if float(metrics.get("real_data_qa_accuracy", 1.0) or 0.0) < 0.9:
            requests.append(
                {
                    "request_id": "weak_retrieval_grounding",
                    "material_types": ["qa_pair", "source_claim"],
                    "reason": "weak retrieval grounding",
                    "priority": 0.9,
                }
            )
        language_balance = report.get("language_balance", {})
        if isinstance(language_balance, dict):
            total = sum(int(value or 0) for value in language_balance.values())
            if total > 0:
                for lang in ("jp", "en"):
                    ratio = float(language_balance.get(lang, 0) or 0) / float(total)
                    if ratio < 0.25:
                        requests.append(
                            {
                                "request_id": f"language_imbalance_{lang}",
                                "material_types": ["summary", "qa_pair"],
                                "language": lang,
                                "reason": "language imbalance",
                                "priority": 0.75,
                            }
                        )
        return requests

    def material_requests_from_fixture_feedback(self, report: Dict[str, object]) -> List[Dict[str, object]]:
        raw_plan = report.get("fixture_expansion_plan", [])
        if not isinstance(raw_plan, list) or not raw_plan:
            raw_plan = report.get("expansion_plan", [])
        if not isinstance(raw_plan, list):
            return []

        requests: List[Dict[str, object]] = []
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "") or "")
            missing_material_types = [
                str(value)
                for value in item.get("missing_material_types_now", item.get("missing_material_types", []))
                if str(value)
            ]
            preferred_material_types = [
                str(value)
                for value in item.get("preferred_material_types", [])
                if str(value)
            ]
            if action == "collect_additional_distinct_sources":
                request_id = "fixture_source_diversity_gap"
                evaluation_gaps = ["retrieval_grounding"]
                reason = "fixture source diversity gap"
            elif action == "add_negative_and_contrastive_materials":
                request_id = "fixture_counterexample_gap"
                evaluation_gaps = ["negative_control", "contrastive_control"]
                reason = "fixture counterexample pressure gap"
            elif action == "manual_review_high_stall_candidates":
                request_id = "fixture_repair_support_gap"
                evaluation_gaps = ["retrieval_grounding"]
                reason = "fixture stalled repair support gap"
            elif action == "resolve_source_revision_conflicts":
                request_id = "fixture_revision_conflict_gap"
                evaluation_gaps = ["retrieval_grounding"]
                reason = "fixture revision conflict gap"
            else:
                request_id = f"fixture_{action or 'coverage'}"
                evaluation_gaps = []
                reason = "fixture coverage gap"
            requests.append(
                {
                    "request_id": request_id,
                    "material_types": preferred_material_types,
                    "missing_material_types": missing_material_types,
                    "reason": reason,
                    "priority": float(item.get("priority", 0.5) or 0.5) / 5.0,
                    "evaluation_gaps": evaluation_gaps,
                    "guidance": str(item.get("guidance", "") or ""),
                }
            )
        return requests

    def write_material_request_plan(
        self,
        report: Dict[str, object],
        output_path: str = workspace_path("autobot", "material_request_plan.json"),
    ) -> Dict[str, object]:
        requests = self.material_requests_from_evaluation(report)
        payload = {
            "schema": "sara-autobot-material-request-plan-v1",
            "request_count": len(requests),
            "requests": requests,
        }
        resolved = ensure_parent_directory(output_path)
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        payload["output_path"] = os.path.abspath(output_path)
        return payload

    def write_fixture_material_request_plan(
        self,
        report: Dict[str, object],
        output_path: str = workspace_path("autobot", "fixture_material_request_plan.json"),
    ) -> Dict[str, object]:
        requests = self.material_requests_from_fixture_feedback(report)
        payload = {
            "schema": "sara-autobot-material-request-plan-v1",
            "request_count": len(requests),
            "request_source": "fixture_feedback",
            "requests": requests,
        }
        resolved = ensure_parent_directory(output_path)
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        payload["output_path"] = os.path.abspath(output_path)
        return payload

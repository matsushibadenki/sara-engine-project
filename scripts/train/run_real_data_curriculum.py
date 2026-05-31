# Directory Path: scripts/train/run_real_data_curriculum.py
# English Title: Real-Data Curriculum Runner
# Purpose/Content: Runs staged real-data training pipelines (small/medium/large) with managed reports and optional gate validation.

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path
from scripts.utils.manage_db import SaraCorpusDB


class CurriculumProfile:
    def __init__(
        self,
        stage: str,
        description: str,
        min_quality_score: float,
        snn_lm_save_dir: str,
        self_org_save_dir: str,
        snn_lm_epochs: int,
        snn_lm_chunk_size: int,
        snn_lm_stride: int,
        snn_lm_learn_epochs: int,
        snn_lm_chat_weight: int,
        include_phase4_gate: bool,
        include_phase5_gates: bool,
        include_operational_readiness: bool,
        recommended_export_count: int,
        external_validity_max_docs: int,
        external_validity_max_cases: int,
        phase3_regression_tolerance: float = 0.05,
    ) -> None:
        self.stage = stage
        self.description = description
        self.min_quality_score = min_quality_score
        self.snn_lm_save_dir = snn_lm_save_dir
        self.self_org_save_dir = self_org_save_dir
        self.snn_lm_epochs = snn_lm_epochs
        self.snn_lm_chunk_size = snn_lm_chunk_size
        self.snn_lm_stride = snn_lm_stride
        self.snn_lm_learn_epochs = snn_lm_learn_epochs
        self.snn_lm_chat_weight = snn_lm_chat_weight
        self.include_phase4_gate = include_phase4_gate
        self.include_phase5_gates = include_phase5_gates
        self.include_operational_readiness = include_operational_readiness
        self.recommended_export_count = recommended_export_count
        self.external_validity_max_docs = external_validity_max_docs
        self.external_validity_max_cases = external_validity_max_cases
        self.phase3_regression_tolerance = phase3_regression_tolerance


def _profiles() -> Dict[str, CurriculumProfile]:
    return {
        "small": CurriculumProfile(
            stage="small",
            description="Small-scale sanity stage for real data learning and Stage B-E/Phase5 visibility.",
            min_quality_score=0.85,
            snn_lm_save_dir=model_path("curriculum", "small", "snn_lm"),
            self_org_save_dir=model_path("curriculum", "small", "self_organized_llm"),
            snn_lm_epochs=1,
            snn_lm_chunk_size=64,
            snn_lm_stride=32,
            snn_lm_learn_epochs=1,
            snn_lm_chat_weight=2,
            include_phase4_gate=False,
            include_phase5_gates=True,
            include_operational_readiness=False,
            recommended_export_count=100,
            external_validity_max_docs=256,
            external_validity_max_cases=24,
        ),
        "medium": CurriculumProfile(
            stage="medium",
            description="Medium-scale stage with stricter quality and continual/scale checks.",
            min_quality_score=0.80,
            snn_lm_save_dir=model_path("curriculum", "medium", "snn_lm"),
            self_org_save_dir=model_path("curriculum", "medium", "self_organized_llm"),
            snn_lm_epochs=2,
            snn_lm_chunk_size=96,
            snn_lm_stride=24,
            snn_lm_learn_epochs=2,
            snn_lm_chat_weight=3,
            include_phase4_gate=True,
            include_phase5_gates=True,
            include_operational_readiness=False,
            recommended_export_count=10_000,
            external_validity_max_docs=1024,
            external_validity_max_cases=64,
        ),
        "large": CurriculumProfile(
            stage="large",
            description="Large-scale stage for production-oriented readiness under strict operational checks.",
            min_quality_score=0.75,
            snn_lm_save_dir=model_path("curriculum", "large", "snn_lm"),
            self_org_save_dir=model_path("curriculum", "large", "self_organized_llm"),
            snn_lm_epochs=3,
            snn_lm_chunk_size=128,
            snn_lm_stride=32,
            snn_lm_learn_epochs=2,
            snn_lm_chat_weight=4,
            include_phase4_gate=True,
            include_phase5_gates=True,
            include_operational_readiness=True,
            recommended_export_count=100_000,
            external_validity_max_docs=4096,
            external_validity_max_cases=128,
        ),
    }


def build_preflight_report(profile: CurriculumProfile, db_path: str = os.path.join("data", "sara_corpus.db")) -> Dict[str, object]:
    resolved_db_path = os.path.abspath(os.path.join(PROJECT_ROOT, db_path))
    db_exists = os.path.exists(resolved_db_path)
    errors: List[str] = []
    warnings: List[str] = []
    material_summary: Dict[str, object] = {}
    review_summary: Dict[str, object] = {}
    export_plan: Dict[str, object] = {"total_count": 0, "items": []}

    if not db_exists:
        errors.append(f"Corpus DB not found: {resolved_db_path}")
    else:
        db = SaraCorpusDB(resolved_db_path)
        material_summary = db.get_material_summary()
        review_summary = db.get_review_summary()
        export_plan = db.summarize_export_plan(
            min_quality_score=profile.min_quality_score,
            show_inactive=False,
        )

    selected_count = int(export_plan.get("total_count", 0) or 0)
    if selected_count <= 0:
        errors.append(
            f"No active material meets min_quality_score>={profile.min_quality_score:.2f}."
        )
    elif selected_count < profile.recommended_export_count:
        warnings.append(
            "Selected material count is below the recommended scale "
            f"for stage={profile.stage} "
            f"(selected={selected_count}, recommended>={profile.recommended_export_count})."
        )

    active_count = int(material_summary.get("active_count", 0) or 0) if material_summary else 0
    if db_exists and active_count <= 0:
        errors.append("Corpus DB has no active materials.")

    return {
        "passed": not errors,
        "stage": profile.stage,
        "db_path": resolved_db_path,
        "db_exists": db_exists,
        "min_quality_score": profile.min_quality_score,
        "recommended_export_count": profile.recommended_export_count,
        "selected_count": selected_count,
        "material_summary": material_summary,
        "review_summary": review_summary,
        "export_plan": export_plan,
        "errors": errors,
        "warnings": warnings,
    }


def build_curriculum_commands(profile: CurriculumProfile, skip_gates: bool = False) -> List[List[str]]:
    commands: List[List[str]] = []
    py = sys.executable

    commands.append(
        [
            py,
            os.path.join("scripts", "sara_cli.py"),
            "db-export",
            "--min-quality-score",
            f"{profile.min_quality_score:.2f}",
        ]
    )
    commands.append(
        [
            py,
            os.path.join("scripts", "train", "train_self_organized.py"),
            "--corpus",
            os.path.join("data", "processed", "corpus.txt"),
            "--save-dir",
            profile.self_org_save_dir,
        ]
    )
    commands.append(
        [
            py,
            os.path.join("scripts", "train", "train_snn_lm.py"),
            "--corpus",
            os.path.join("data", "processed", "corpus.txt"),
            "--chat-data",
            os.path.join("data", "raw", "chat_data.jsonl"),
            "--save-dir",
            profile.snn_lm_save_dir,
            "--epochs",
            str(profile.snn_lm_epochs),
            "--chunk-size",
            str(profile.snn_lm_chunk_size),
            "--stride",
            str(profile.snn_lm_stride),
            "--learn-epochs",
            str(profile.snn_lm_learn_epochs),
            "--chat-weight",
            str(profile.snn_lm_chat_weight),
            "--turboquant",
        ]
    )

    if not skip_gates:
        commands.append(
            [
                py,
                os.path.join("scripts", "eval", "phase3_accuracy_suite.py"),
                "--history-path",
                workspace_path("evaluation", f"phase3_curriculum_{profile.stage}_history.json"),
                "--regression-tolerance",
                f"{profile.phase3_regression_tolerance:.6f}",
            ]
        )
        commands.append([py, os.path.join("scripts", "eval", "phase3_completion_gate.py")])
        if profile.include_phase4_gate:
            commands.append([py, os.path.join("scripts", "eval", "phase4_scale_continual_benchmark.py")])
            commands.append([py, os.path.join("scripts", "eval", "phase4_completion_gate.py")])
        if profile.include_phase5_gates:
            commands.append([py, os.path.join("scripts", "eval", "phase5_predictive_coding_benchmark.py")])
            commands.append([py, os.path.join("scripts", "eval", "phase5_entry_gate.py")])
            commands.append([py, os.path.join("scripts", "eval", "phase5_completion_gate.py")])
            commands.append(
                [
                    py,
                    os.path.join("scripts", "eval", "real_data_external_validity.py"),
                    "--corpus",
                    os.path.join("data", "processed", "corpus.txt"),
                    "--max-docs",
                    str(profile.external_validity_max_docs),
                    "--max-cases",
                    str(profile.external_validity_max_cases),
                    "--report-path",
                    workspace_path("evaluation", f"real_data_external_validity_{profile.stage}.json"),
                    "--summary-path",
                    workspace_path("evaluation", f"real_data_external_validity_{profile.stage}_summary.txt"),
                    "--history-path",
                    workspace_path("evaluation", f"real_data_external_validity_{profile.stage}_history.json"),
                    "--regression-tolerance",
                    f"{profile.phase3_regression_tolerance:.6f}",
                ]
            )
        if profile.include_operational_readiness:
            commands.append(
                [
                    py,
                    os.path.join("scripts", "eval", "operational_readiness.py"),
                    "--refresh-artifacts",
                    "--soak-profile",
                    "extended",
                    "--include-accuracy",
                    "--strict-production",
                    "--phase3-regression-tolerance",
                    "0.05",
                ]
            )

    return commands


def run_real_data_curriculum(
    stage: str,
    dry_run: bool = False,
    skip_gates: bool = False,
    preflight_only: bool = False,
) -> Dict[str, object]:
    profiles = _profiles()
    if stage not in profiles:
        raise ValueError(f"Unknown stage: {stage}")
    profile = profiles[stage]
    commands = build_curriculum_commands(profile, skip_gates=skip_gates)
    preflight = build_preflight_report(profile)

    started_at = time.time()
    command_reports: List[Dict[str, object]] = []
    completed = True if dry_run else bool(preflight.get("passed", False))

    if preflight_only:
        completed = bool(preflight.get("passed", False))
        commands = []
    elif not completed and not dry_run:
        commands = []

    for index, command in enumerate(commands, start=1):
        command_str = " ".join(command)
        print(f"[{index}/{len(commands)}] {command_str}")
        if dry_run:
            command_reports.append({"index": index, "command": command_str, "returncode": 0, "duration_seconds": 0.0, "dry_run": True})
            continue

        command_started = time.time()
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        duration = time.time() - command_started
        command_reports.append(
            {
                "index": index,
                "command": command_str,
                "returncode": int(result.returncode),
                "duration_seconds": duration,
                "dry_run": False,
            }
        )
        if result.returncode != 0:
            completed = False
            break

    report = {
        "suite_name": "RealDataCurriculumRunner",
        "stage": profile.stage,
        "description": profile.description,
        "dry_run": bool(dry_run),
        "skip_gates": bool(skip_gates),
        "preflight_only": bool(preflight_only),
        "passed": bool(completed),
        "preflight": preflight,
        "started_at_unix": started_at,
        "duration_seconds": time.time() - started_at,
        "command_count": len(commands),
        "completed_count": len(command_reports),
        "commands": command_reports,
    }
    return report


def default_report_path(stage: str) -> str:
    return workspace_path("reports", f"real_data_curriculum_{stage}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged real-data curriculum training (small/medium/large).")
    parser.add_argument("--stage", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without executing them.")
    parser.add_argument("--skip-gates", action="store_true", help="Run only export + training steps without evaluation gates.")
    parser.add_argument("--preflight-only", action="store_true", help="Write the preflight report without running training commands.")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional managed path for curriculum report JSON.",
    )
    args = parser.parse_args()

    try:
        report = run_real_data_curriculum(
            stage=str(args.stage),
            dry_run=bool(args.dry_run),
            skip_gates=bool(args.skip_gates),
            preflight_only=bool(args.preflight_only),
        )
    except ValueError as error:
        print(str(error))
        return 1

    report_path = ensure_parent_directory(args.report_path or default_report_path(str(args.stage)))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Saved curriculum report: {report_path}")

    if not report.get("passed", False):
        print("Curriculum run failed.")
        return 1
    print("Curriculum run passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

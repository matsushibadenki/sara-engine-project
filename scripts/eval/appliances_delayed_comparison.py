#!/usr/bin/env python3
"""Register, execute once, or replay the frozen prospective comparison."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation.appliances_learning import (
    DelayedCounts, PROTOCOL_SHA256, decide, load_protocol, run_phase, shuffled_targets,
)
from sara_engine.evaluation.appliances_prospective import extract
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.utils.project_paths import (
    ensure_parent_directory, model_path, processed_data_path, project_path,
    raw_data_path, workspace_path,
)

CODE_PATHS = (
    "src/sara_engine/evaluation/appliances_learning.py",
    "src/sara_engine/evaluation/appliances_prospective.py",
    "src/sara_engine/evaluation/r1_temporal_learning_benchmark.py",
    "src/sara_engine/neuro/neuron.py",
    "src/sara_engine/utils/project_paths.py",
    "scripts/eval/appliances_delayed_comparison.py",
)


def dump_exclusive(path, payload):
    path = Path(ensure_parent_directory(path))
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return sha256(path.read_bytes()).hexdigest()


def code_hashes():
    return {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in CODE_PATHS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--register", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    registration_path = Path(processed_data_path("benchmark_fixtures", "appliances_learning_registration_v1.json"))
    receipt_path = Path(processed_data_path("benchmark_fixtures", "appliances_learning_result_v1.json"))
    if args.register:
        dump_exclusive(str(registration_path), {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": PROTOCOL_SHA256, "code_sha256": code_hashes(),
            "learner_executed": False, "sealed_values_read": False,
        })
        print("Appliance comparison registered before learner execution")
        return
    registration = json.loads(registration_path.read_text())
    if registration["protocol_sha256"] != PROTOCOL_SHA256 or registration["code_sha256"] != code_hashes():
        raise ValueError("Registered protocol or implementation changed")
    if args.execute and receipt_path.exists():
        raise ValueError("One-shot comparison already retained; do not retune or rerun")
    task_bytes = Path(processed_data_path("benchmark_fixtures", "appliances_prospective_v1.json")).read_bytes()
    if sha256(task_bytes).hexdigest() != protocol["task_protocol_sha256"]:
        raise ValueError("Prospective task contract changed")
    source = Path(raw_data_path("r2_appliances", "energydata_complete.csv"))
    if sha256(source.read_bytes()).hexdigest() != protocol["source_sha256"]:
        raise ValueError("Pinned source changed")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        examples, audit = extract(csv.reader(stream), json.loads(task_bytes))
    digest = sha256()
    for example in examples:
        digest.update(repr(example).encode())
    if not audit["support_passed"] or digest.hexdigest() != protocol["expected_examples_sha256"]:
        raise ValueError("Causal input identities or support changed")
    train = [example for example in examples if example.split == "train"]
    development = [example for example in examples if example.split == "development"]
    if args.verify_replay:
        retained = json.loads(receipt_path.read_text())
        if retained["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError("Retained result protocol mismatch")
        for arm in protocol["arms"]:
            identity = retained["arms"][arm]["checkpoint"]
            raw = Path(model_path("appliances", arm + "_v1.json")).read_bytes()
            if sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError("Final model checkpoint changed")
            model = DelayedCounts(protocol)
            model.restore(json.loads(raw))
            replay = run_phase(model, development, arm, protocol, training=False)
            original = retained["arms"][arm]["development"]
            for key in ("prediction_sha256", "day_brier", "pooled_brier", "balanced_accuracy", "actual_counter_writes"):
                if replay[key] != original[key]:
                    raise ValueError("Fresh-process checkpoint replay differs")
        print("All seven frozen checkpoint traces replay exactly; sealed values remain unread")
        return
    # Mark a consumed execution before any learning, including partial failures.
    dump_exclusive(workspace_path("evaluation", "appliances_learning_started_v1.json"), {
        "protocol_sha256": PROTOCOL_SHA256, "started_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha256(registration_path.read_bytes()).hexdigest(),
    })
    targets = shuffled_targets(train, protocol)
    arms = {}
    for arm in protocol["arms"]:
        model = DelayedCounts(protocol)
        training = run_phase(model, train, arm, protocol, training=True,
                             targets=targets if arm == "joint_shuffled" else None)
        checkpoint_path = model_path("appliances", arm + "_v1.json")
        checkpoint_sha = dump_exclusive(checkpoint_path, model.checkpoint())
        evaluation = run_phase(model, development, arm, protocol, training=False)
        restored = DelayedCounts(protocol)
        restored.restore(json.loads(Path(checkpoint_path).read_text()))
        replay = run_phase(restored, development, arm, protocol, training=False)
        exact = all(replay[key] == evaluation[key] for key in (
            "prediction_sha256", "day_brier", "pooled_brier", "balanced_accuracy", "actual_counter_writes"))
        arms[arm] = {"training": training, "development": evaluation,
                     "checkpoint_replay_exact": exact,
                     "replay_work": replay,
                     "checkpoint": {"path": "models/appliances/" + arm + "_v1.json", "sha256": checkpoint_sha}}
        print(f'{arm}: updates={training["actual_counter_writes"]}, day Brier={evaluation["equal_day_brier"]:.6f}, replay={exact}')
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": sha256(registration_path.read_bytes()).hexdigest(),
        "code_sha256": code_hashes(), "source_audit": audit,
        "causal_examples_sha256": digest.hexdigest(), "arms": arms,
        "decision": decide(arms, protocol),
        "evaluator_materialized_state_bytes": _deep_size((examples, train, development, targets)),
        "accounting_scope": "Per-arm state bounds include count tables, owned receipts and conservative scratch. Materialized source/evaluator tape is reported separately. Replay/reference CPU and work are charged explicitly. No GPU, matrix, whole-network gradients, physical energy or sealed metrics.",
        "sealed_numeric_values_read": 0, "promotion_authorized": False,
    }
    result_sha = dump_exclusive(workspace_path("evaluation", "appliances_learning_full_v1.json"), result)
    compact = {key: value for key, value in result.items() if key != "arms"}
    compact["full_result_sha256"] = result_sha
    compact["arms"] = {arm: {key: value for key, value in row.items() if key != "replay_work"}
                       for arm, row in arms.items()}
    dump_exclusive(str(receipt_path), compact)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()

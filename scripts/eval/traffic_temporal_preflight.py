#!/usr/bin/env python3
"""Acquire, register, execute once and replay a fresh temporal-necessity screen."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.traffic_temporal import (
    LocalConjunctionCounts, PROTOCOL_SHA256, contrasts, contexts, decision, extract,
    load_protocol, run_phase, shuffled_targets,
)
from sara_engine.utils.project_paths import (
    ensure_parent_directory, model_path, processed_data_path, project_path,
    raw_data_path, workspace_path,
)

CODE = ("scripts/eval/traffic_temporal_preflight.py", "src/sara_engine/evaluation/traffic_temporal.py",
        "src/sara_engine/evaluation/r1_temporal_learning_benchmark.py", "src/sara_engine/utils/project_paths.py")
PRESERVED = (
    "data/processed/benchmark_fixtures/appliances_learning_v1.json",
    "data/processed/benchmark_fixtures/appliances_learning_registration_v1.json",
    "data/processed/benchmark_fixtures/appliances_learning_result_v1.json",
    "data/processed/benchmark_fixtures/appliances_prospective_v1.json",
    "data/processed/benchmark_fixtures/appliances_prospective_receipt_v1.json",
    "data/processed/benchmark_fixtures/r2_electricity_learner_protocol_v1.json",
    "data/processed/benchmark_fixtures/r2_electricity_learner_protocol_v2.json",
    "data/processed/benchmark_fixtures/r2_electricity_v2_result_receipt.json",
    "data/processed/benchmark_fixtures/temporal_boundary_learning_v1.json",
    "data/processed/benchmark_fixtures/temporal_boundary_learning_result_receipt_v1.json",
    *("models/appliances/" + arm + "_v1.json" for arm in
      ("snapshot", "histogram", "ordinal", "transition_gap", "analog", "joint", "joint_shuffled")),
)


def hashes(names):
    return {name: sha256(Path(project_path(name)).read_bytes()).hexdigest() for name in names}


def save(path, payload, *, compressed=False):
    raw = (json.dumps(payload, sort_keys=True, indent=None if compressed else 2,
                      separators=(",", ":") if compressed else None, allow_nan=False) + "\n").encode()
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    target = Path(ensure_parent_directory(path))
    with target.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest()


def analytic_branch_audit(protocol):
    """Input-only representability witness, not a synthetic learner experiment."""
    cutoff = datetime(2013, 5, 6, 9)
    weather = (56, 0, 0, 2, "None", (("Clouds", "scattered clouds"),))
    histories = ((3250, 3500, 2750, 3000), (2750, 3500, 3250, 3000))
    equal_current = all(contexts(histories[0], weather, cutoff, arm, protocol) == contexts(
        histories[1], weather, cutoff, arm, protocol) for arm in ("current_exact", "current_weather", "history_histogram"))
    a, b = (contexts(history, weather, cutoff, "temporal_pair", protocol) for history in histories)
    if not equal_current or a[:-1] != b[:-1] or a[-1] == b[-1]:
        raise ValueError("Local branch cannot represent the registered order contrast")
    # Separate branch address parameters can represent opposite posteriors.
    model = LocalConjunctionCounts(protocol)
    model.restore({a[-1]: [0, 8], b[-1]: [8, 0]})
    left = model.predict(a, cutoff)
    model.deliver(left, left[1], lambda: 1, training=False)
    right = model.predict(b, left[1])
    model.deliver(right, right[1], lambda: 0, training=False)
    if not left[3] > .5 > right[3]:
        raise ValueError("Representable keys are not discriminated by the local readout")
    return {"passed": True, "current_and_histogram_equal": equal_current,
            "temporal_branch_distinct": True, "representable_probabilities": [left[3], right[3]],
            "known_limit": "Trends beyond +/-500 saturate; larger same-sign increments can collide. This is not complete ordinal retention.",
            "learner_trained": False, "scope": "Analytical address/parameter witness; real-data learning is a separate gate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("acquire", "register", "execute", "verify-replay"):
        group.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    source = Path(raw_data_path("r2_traffic", "Metro_Interstate_Traffic_Volume.csv.gz"))
    registration_path = Path(processed_data_path("benchmark_fixtures", "traffic_temporal_registration_v1.json"))
    result_path = Path(processed_data_path("benchmark_fixtures", "traffic_temporal_result_v1.json"))
    if args.acquire:
        with urlopen(protocol["source_url"], timeout=60) as response:
            archive = response.read(2_000_001)
        if len(archive) > 2_000_000 or sha256(archive).hexdigest() != protocol["source_archive_sha256"]:
            raise ValueError("Official archive exceeds limit or checksum differs")
        with ZipFile(BytesIO(archive)) as zipped:
            member = zipped.getinfo(source.name)
            if member.file_size > 2_000_000:
                raise ValueError("Compressed CSV exceeds size limit")
            raw = zipped.read(member)
        if sha256(raw).hexdigest() != protocol["source_gzip_sha256"]:
            raise ValueError("Compressed source checksum differs")
        with Path(ensure_parent_directory(str(source))).open("xb") as stream:
            stream.write(raw)
        print("Pinned official traffic source acquired; no numeric values inspected")
        return
    if args.register:
        save(str(registration_path), {"registered_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": PROTOCOL_SHA256, "code_sha256": hashes(CODE),
             "preserved_negative_sha256": hashes(PRESERVED), "analytical_branch_audit": analytic_branch_audit(protocol),
             "real_data_learner_scored": False, "sealed_numeric_values_read": 0})
        print("Traffic task, gates and implementation registered before numeric inspection/scoring")
        return
    registration = json.loads(registration_path.read_text())
    if (registration["protocol_sha256"] != PROTOCOL_SHA256 or registration["code_sha256"] != hashes(CODE)
            or registration["preserved_negative_sha256"] != hashes(PRESERVED)):
        raise ValueError("Registered code/protocol or preserved negative evidence changed")
    if args.execute:
        if result_path.exists():
            raise ValueError("Consumed task: no repeated scoring or retuning")
        save(workspace_path("evaluation", "traffic_temporal_started_v1.json"), {
            "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA256})
    compressed_source = source.read_bytes()
    if sha256(compressed_source).hexdigest() != protocol["source_gzip_sha256"]:
        raise ValueError("Pinned compressed source changed")
    raw = gzip.decompress(compressed_source)
    if sha256(raw).hexdigest() != protocol["source_csv_sha256"]:
        raise ValueError("Pinned CSV changed")
    examples, audit = extract(csv.reader(StringIO(raw.decode("utf-8-sig"))), protocol)
    del raw, compressed_source
    train = [example for example in examples if example.split == "train"]
    development = [example for example in examples if example.split == "development"]
    contrast = contrasts(train, protocol)
    examples_sha = sha256("\n".join(repr(e) for e in examples).encode()).hexdigest()
    if args.verify_replay:
        retained = json.loads(result_path.read_text())
        if retained["causal_examples_sha256"] != examples_sha or retained["audit"] != audit or retained["contrasts"] != contrast:
            raise ValueError("Causal source audit replay differs")
        for arm, row in retained["arms"].items():
            checkpoint = Path(model_path("traffic_temporal", arm + "_v1.json.gz")).read_bytes()
            if sha256(checkpoint).hexdigest() != row["checkpoint_sha256"]:
                raise ValueError("Final research checkpoint changed")
            model = LocalConjunctionCounts(protocol)
            model.restore(json.loads(gzip.decompress(checkpoint)))
            replay = run_phase(model, development, arm, protocol, training=False)
            for key in ("prediction_sha256", "day_brier", "balanced_accuracy", "actual_counter_writes"):
                if replay[key] != row["development"][key]:
                    raise ValueError("Frozen checkpoint replay differs")
        print("Source audit and all saved development traces replay exactly; older negative hashes unchanged")
        return
    result = {"protocol_sha256": PROTOCOL_SHA256, "code_sha256": hashes(CODE),
              "registration_sha256": sha256(registration_path.read_bytes()).hexdigest(),
              "preserved_negative_sha256": hashes(PRESERVED), "audit": audit, "contrasts": contrast,
              "causal_examples_sha256": examples_sha, "arms": {}, "learner_scored": False,
              "sealed_scoring_authorized": False, "promotion_authorized": False}
    if not audit["support_passed"] or not contrast["passed"]:
        result["decision"] = {"passed": False, "reason": "Fixed support/current-contrast gate failed before scoring"}
        save(str(result_path), result)
        print(json.dumps(result["decision"]))
        return
    targets = shuffled_targets(train, protocol)
    for arm in protocol["arms"]:
        model = LocalConjunctionCounts(protocol)
        training = run_phase(model, train, arm, protocol, training=True,
                             targets=targets if arm == "temporal_shuffled" else None)
        path = model_path("traffic_temporal", arm + "_v1.json.gz")
        checkpoint_sha = save(path, model.checkpoint(), compressed=True)
        evaluation = run_phase(model, development, arm, protocol, training=False)
        restored = LocalConjunctionCounts(protocol)
        restored.restore(json.loads(gzip.decompress(Path(path).read_bytes())))
        replay = run_phase(restored, development, arm, protocol, training=False)
        exact = all(replay[key] == evaluation[key] for key in
                    ("prediction_sha256", "day_brier", "balanced_accuracy", "actual_counter_writes"))
        result["arms"][arm] = {"training": training, "development": evaluation,
                               "checkpoint_replay_exact": exact, "checkpoint_sha256": checkpoint_sha,
                               "replay_work": replay,
                               "concurrent_original_and_replay_readout_bound_bytes": training["state_bound_bytes"] + replay["state_bound_bytes"]}
        print(f'{arm}: updates={training["actual_counter_writes"]}, day Brier={evaluation["equal_day_brier"]:.6f}, replay={exact}', flush=True)
    persistence = run_phase(LocalConjunctionCounts(protocol), development, "current_exact", protocol,
                            training=False, persistence=True)
    result.update(learner_scored=True, persistence=persistence,
                  decision=decision(result["arms"], persistence, protocol),
                  evaluator_materialized_state_bytes=_deep_size((examples, train, development, targets)),
                  accounting_scope="Counter/state/prediction/history work recorded per model. Original+replay readout bounds and batch evaluator memory are charged separately. Serialization and Python process memory are not claimed as measured peak RSS or physical energy.")
    if hashes(PRESERVED) != registration["preserved_negative_sha256"]:
        raise ValueError("Previous negative evidence was modified")
    full_sha = save(workspace_path("evaluation", "traffic_temporal_full_v1.json"), result)
    for row in result["arms"].values():
        del row["replay_work"]
    result["full_result_sha256"] = full_sha
    save(str(result_path), result)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()

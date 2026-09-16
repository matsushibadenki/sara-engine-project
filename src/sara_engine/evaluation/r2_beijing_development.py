"""Training/development-only R2 evaluation for hourly Beijing sensor data."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import random
import resource
import time
import zipfile

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig
from sara_engine.learning.observable_revision import BoundedObservableRevisionReadout, ObservableRevisionConfig
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path, raw_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_beijing_model_v1.json")
PROTOCOL_SHA256 = "e03d89ac6da6c3b8b5f572d1cf82c4d71d4915ee8e555012dc237778015a6c71"
INNER_SHA256 = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen R2 Beijing model protocol changed")
    return json.loads(raw)


@dataclass(frozen=True, slots=True)
class SensorPair:
    timestamp: datetime
    split: str
    station: str
    hour: int
    month: int
    values: tuple
    label: int


def _number(row: dict[str, str], name: str) -> float | None:
    return None if row[name] in ("", "NA") else float(row[name])


def _split(value: datetime) -> str | None:
    if datetime(2013, 3, 1) <= value <= datetime(2015, 12, 31, 23):
        return "training"
    if datetime(2016, 1, 1) <= value <= datetime(2016, 6, 30, 23):
        return "development"
    return None


def load_pairs() -> list[SensorPair]:
    path = Path(raw_data_path("r2_beijing_air_quality", "PRSA2017_Data_20130301-20170228.zip"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != INNER_SHA256:
        raise ValueError("R2 source archive hash mismatch")
    pairs = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(item for item in archive.namelist() if item.endswith(".csv")):
            payload = archive.read(name)
            rows = list(csv.DictReader(line.decode("utf-8-sig") for line in payload.splitlines(keepends=True)))
            for index in range(len(rows) - 1):
                row, following = rows[index], rows[index + 1]
                timestamp = datetime(int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
                following_time = datetime(int(following["year"]), int(following["month"]), int(following["day"]), int(following["hour"]))
                split = _split(timestamp)
                if split is None or _split(following_time) != split or row["PM2.5"] == "NA" or following["PM2.5"] == "NA":
                    continue
                pm = float(row["PM2.5"])
                def lag_delta(hours: int) -> float | None:
                    if index < hours or rows[index - hours]["PM2.5"] == "NA":
                        return None
                    return pm - float(rows[index - hours]["PM2.5"])
                missing = tuple(name for name in ("TEMP", "DEWP", "PRES", "WSPM", "RAIN", "wd") if row[name] in ("", "NA"))
                values = (
                    pm, lag_delta(1), lag_delta(3), lag_delta(24),
                    _number(row, "TEMP"), _number(row, "DEWP"), _number(row, "PRES"),
                    _number(row, "WSPM"), _number(row, "RAIN"), row["wd"] if row["wd"] not in ("", "NA") else None,
                    missing,
                )
                pairs.append(SensorPair(timestamp, split, row["station"], int(row["hour"]), int(row["month"]), values,
                                        int(float(following["PM2.5"]) > pm)))
    pairs.sort(key=lambda item: (item.timestamp, item.station))
    return pairs


class SparseEventEncoder:
    def __init__(self, protocol: dict, *, spiking: bool, destroy_order: bool = False) -> None:
        self.protocol = protocol
        self.spiking = spiking
        self.destroy_order = destroy_order
        self.routes: dict[tuple, int] = {}
        self.units: list[Neuron] = []

    def _route(self, key: tuple) -> int:
        route = self.routes.get(key)
        if route is None:
            if len(self.routes) >= self.protocol["feature_contract"]["maximum_route_vocabulary"]:
                raise ValueError("R2 route vocabulary exceeded")
            route = len(self.routes)
            self.routes[key] = route
            if self.spiking:
                self.units.append(Neuron(route, num_branches=1))
        return route

    @staticmethod
    def _bin(value: float | None, boundaries: list[float]) -> int:
        return len(boundaries) + 1 if value is None else bisect_right(boundaries, value)

    def encode(self, pair: SensorPair) -> tuple[tuple[int, float], ...]:
        bins = self.protocol["feature_contract"]["numeric_bins"]
        pm, d1, d3, d24, temp, dewp, pres, wind, rain, direction, missing = pair.values
        if self.destroy_order:
            d1, d24 = d24, d1
        pm_bin = self._bin(pm, bins["pm25_level"])
        d1_bin = self._bin(d1, bins["pm25_delta"])
        d3_bin = self._bin(d3, bins["pm25_delta"])
        d24_bin = self._bin(d24, bins["pm25_delta"])
        keys = [
            ("station", pair.station), ("hour", pair.hour), ("month", pair.month),
            ("pm", pm_bin), ("d1", d1_bin), ("d3", d3_bin), ("d24", d24_bin),
            ("temp", self._bin(temp, bins["temperature"])),
            ("dewp", self._bin(dewp, bins["dew_point"])),
            ("pres", self._bin(pres, bins["pressure"])),
            ("wind", self._bin(wind, bins["wind_speed"])),
            ("rain", self._bin(rain, bins["rain"])), ("wd", direction), ("missing", missing),
            ("sxhd", pair.station, pair.hour, d1_bin),
            ("sxpmd", pair.station, pm_bin, d1_bin),
            ("pmxdd", pm_bin, d1_bin, d24_bin),
        ]
        active = []
        for key in keys:
            route = self._route(key)
            if self.spiking:
                unit = self.units[route]
                unit.v = 0.0; unit.spike = False; unit.refractory_time = 0; unit.active_branches.clear()
                unit.add_input_to_branch(0, 1.6)
                if not unit.step():
                    raise RuntimeError("R2 feature neuron did not spike")
            active.append((route, 1.0))
        return tuple(active)


class LocalArm:
    def __init__(self, protocol: dict, arm: str) -> None:
        self.arm = arm
        self.encoder = SparseEventEncoder(protocol, spiking=arm != "scalar_local_residual",
                                          destroy_order=arm == "snn_temporal_order_destroyed")
        learning = protocol["learning"]
        self.three_readout = None
        self.readout = None
        if arm == "snn_three_factor":
            self.three_readout = BoundedObservableRevisionReadout(ObservableRevisionConfig(
                learning_rate=learning["three_factor_rate"], trace_decay=learning["trace_decay_per_hour"],
                max_feedback_age=learning["max_feedback_age_hours"], adaptation_horizon=1,
                max_routes=protocol["feature_contract"]["maximum_route_vocabulary"],
                max_active=protocol["feature_contract"]["maximum_active_routes"], max_delta=learning["max_delta"],
            ))
        else:
            self.readout = BoundedLocalOutcomeReadout(LocalOutcomeConfig(
                learning_rate=0.0 if arm == "snn_frozen" else learning["local_residual_rate"],
                trace_decay=learning["trace_decay_per_hour"], max_age=learning["max_feedback_age_hours"],
                max_routes=protocol["feature_contract"]["maximum_route_vocabulary"],
                max_active=protocol["feature_contract"]["maximum_active_routes"], max_delta=learning["max_delta"],
            ))

    def step(self, pair: SensorPair, target: int, logical_time: int) -> tuple[float, int]:
        active = self.encoder.encode(pair)
        outcome = float(2 * target - 1)
        if self.three_readout is not None:
            receipt = self.three_readout.predict(active, time=logical_time)
            update = self.three_readout.observe(receipt, outcome, time=logical_time + 1)
        else:
            assert self.readout is not None
            receipt = self.readout.predict(active, time=logical_time)
            update = self.readout.observe(receipt, outcome, time=logical_time + 1)
        update_count = len(update.deltas) if self.three_readout is not None else update.route_visits
        return receipt.score, len(active) * (5 if self.encoder.spiking else 1) + update_count


def _metrics(records: list[dict]) -> dict:
    positives = sum(row["label"] for row in records); negatives = len(records) - positives
    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in records)
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in records)
    return {"count": len(records), "accuracy": sum(row["correct"] for row in records) / len(records),
            "balanced_accuracy": 0.5 * (tp / positives + tn / negatives),
            "brier": sum(row["brier"] for row in records) / len(records)}


def run_local_arm(protocol: dict, arm: str, pairs: list[SensorPair]) -> dict:
    learner = LocalArm(protocol, arm)
    shuffled = [pair.label for pair in pairs]
    random.Random(protocol["shuffled_outcome_seed"]).shuffle(shuffled)
    records = []; max_work = 0; max_ms = 0.0
    for index, pair in enumerate(pairs):
        started = time.perf_counter_ns()
        target = shuffled[index] if arm == "snn_shuffled_outcomes" else pair.label
        score, work = learner.step(pair, target, index * 3)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        max_work = max(max_work, work); max_ms = max(max_ms, elapsed)
        if pair.split == "development":
            probability = (max(-1.0, min(1.0, score)) + 1.0) / 2.0
            prediction = int(score > 0.0)
            records.append({"station": pair.station, "label": pair.label, "prediction": prediction,
                            "correct": int(prediction == pair.label), "brier": (probability - pair.label) ** 2,
                            "score": score})
    by_station = {station: _metrics([row for row in records if row["station"] == station])
                  for station in sorted({row["station"] for row in records})}
    digest = hashlib.sha256(json.dumps([(row["score"], row["prediction"]) for row in records], separators=(",", ":")).encode()).hexdigest()
    state_bytes = _deep_size(learner)
    budget = protocol["resource_budgets"]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    contracts = (len(learner.encoder.units) <= budget["max_neurons"] and state_bytes <= budget["max_state_bytes"]
                 and max_work <= budget["max_event_work_per_prediction"] and max_ms <= budget["max_cpu_ms_per_prediction"]
                 and peak_rss <= budget["max_peak_rss_bytes"])
    return {"development": _metrics(records), "per_station": by_station,
            "worst_station_balanced_accuracy": min(value["balanced_accuracy"] for value in by_station.values()),
            "prediction_trace_sha256": digest,
            "resources": {"neurons": len(learner.encoder.units), "routes": len(learner.encoder.routes),
                          "state_bytes": state_bytes, "max_event_work": max_work, "max_cpu_ms": max_ms,
                          "peak_rss_bytes": peak_rss, "contracts_passed": contracts}}


def run_transition(pairs: list[SensorPair]) -> dict:
    counts = defaultdict(lambda: [0, 0]); records = []
    for pair in pairs:
        d1 = pair.values[1]
        direction = "missing" if d1 is None else "up" if d1 > 0 else "down_or_equal"
        key = (pair.station, pair.hour, direction)
        negative, positive = counts[key]
        probability = (positive + 1) / (negative + positive + 2)
        if pair.split == "development":
            prediction = int(probability >= 0.5)
            records.append({"station": pair.station, "label": pair.label, "prediction": prediction,
                            "correct": int(prediction == pair.label), "brier": (probability - pair.label) ** 2})
        counts[key][pair.label] += 1
    by_station = {station: _metrics([row for row in records if row["station"] == station]) for station in sorted({row["station"] for row in records})}
    return {"development": _metrics(records), "per_station": by_station,
            "worst_station_balanced_accuracy": min(value["balanced_accuracy"] for value in by_station.values()),
            "resources": {"states": len(counts), "contracts_passed": len(counts) <= 864}}


def decide(protocol: dict, arms: dict) -> dict:
    gate = protocol["development_acceptance"]; candidate = arms["snn_local_residual"]
    baseline = arms["online_station_hour_direction_transition"]
    scalar = arms["scalar_local_residual"]; destroyed = arms["snn_temporal_order_destroyed"]
    shuffled = arms["snn_shuffled_outcomes"]
    checks = {
        "candidate_balanced_accuracy": candidate["development"]["balanced_accuracy"] >= gate["minimum_candidate_balanced_accuracy"],
        "gain_over_online_transition": candidate["development"]["balanced_accuracy"] - baseline["development"]["balanced_accuracy"] >= gate["minimum_balanced_accuracy_gain_over_online_transition"],
        "brier_gain_over_online_transition": baseline["development"]["brier"] - candidate["development"]["brier"] >= gate["minimum_brier_gain_over_online_transition"],
        "temporal_order_ablation": candidate["development"]["balanced_accuracy"] - destroyed["development"]["balanced_accuracy"] >= gate["minimum_temporal_order_ablation_drop"],
        "shuffled_outcome_gap": candidate["development"]["balanced_accuracy"] - shuffled["development"]["balanced_accuracy"] >= gate["minimum_shuffled_outcome_balanced_accuracy_gap"],
        "worst_station": candidate["worst_station_balanced_accuracy"] >= baseline["worst_station_balanced_accuracy"] - gate["maximum_worst_station_balanced_accuracy_drop_from_baseline"],
        "exact_scalar_equivalence": candidate["prediction_trace_sha256"] == scalar["prediction_trace_sha256"],
        "all_resource_contracts_pass": all(result["resources"]["contracts_passed"] for result in arms.values()),
    }
    return {"checks": checks, "passed": all(checks.values()), "frozen_test_authorized": all(checks.values()),
            "status": "development_pass" if all(checks.values()) else "development_negative_result"}


def run_development(protocol: dict) -> dict:
    pairs = load_pairs()
    arms = {arm: run_local_arm(protocol, arm, pairs) for arm in protocol["arms"] if arm != "online_station_hour_direction_transition"}
    arms["online_station_hour_direction_transition"] = run_transition(pairs)
    return {"schema": "sara-r2-beijing-development-result-v1", "experiment_id": protocol["experiment_id"],
            "pair_counts": {split: sum(pair.split == split for pair in pairs) for split in ("training", "development")},
            "arms": arms, "decision": decide(protocol, arms)}


__all__ = ["decide", "load_pairs", "load_protocol", "run_development", "run_local_arm"]

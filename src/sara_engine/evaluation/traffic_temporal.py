"""Prospective traffic preflight with local conjunction counts and sealed parsing."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from time import process_time

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.utils.project_paths import processed_data_path

PROTOCOL_SHA256 = "f16018e6c66c700c70ce7e4dbafb0aed724b38bf063f50032e4ff54673781f94"


def load_protocol():
    raw = Path(processed_data_path("benchmark_fixtures", "traffic_temporal_preflight_v1.json")).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen traffic protocol changed")
    return json.loads(raw)


@dataclass(frozen=True)
class TrafficExample:
    split: str
    cutoff: datetime
    due: datetime
    volumes: tuple[int, ...]
    weather: tuple
    label: int


def extract(rows, protocol):
    """Admit date first, collapse duplicate hours, never read sealed numeric cells."""
    iterator = iter(rows)
    header = next(iterator)
    if len(header) != len(set(header)):
        raise ValueError("Duplicate source columns")
    names = ("date_time", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all",
             "holiday", "weather_main", "weather_description")
    indices = {name: header.index(name) for name in names}
    start, dev, sealed = (datetime.fromisoformat(protocol[key]) for key in
                          ("train_start", "development_start", "sealed_start"))
    hours, invalid, audit = {}, set(), Counter()
    for row in iterator:
        stamp = datetime.fromisoformat(row[indices["date_time"]])
        if stamp.tzinfo is not None or stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError("Expected source-local whole-hour timestamps")
        if stamp >= sealed:
            audit["sealed_rows_timestamp_only"] += 1
            continue
        if stamp < start:
            continue
        audit["admitted_numeric_rows_read"] += 1
        try:
            volume = int(row[indices["traffic_volume"]])
            numbers = tuple(float(row[indices[name]]) for name in ("temp", "rain_1h", "snow_1h", "clouds_all"))
        except (ValueError, IndexError):
            invalid.add(stamp)
            audit["invalid_rows"] += 1
            continue
        holiday = row[indices["holiday"]]
        categories = tuple(row[indices[name]] for name in ("weather_main", "weather_description"))
        if (not 0 <= volume <= 20000 or not all(math.isfinite(number) for number in numbers)
                or numbers[0] <= 0 or numbers[1] < 0 or numbers[2] < 0 or not 0 <= numbers[3] <= 100
                or any(not value or len(value) > 256 for value in (holiday, *categories))):
            invalid.add(stamp)
            audit["invalid_rows"] += 1
            continue
        current = (volume, numbers, holiday)
        if stamp in hours:
            audit["duplicate_hour_rows"] += 1
            if hours[stamp][0] != current:
                invalid.add(stamp)
                audit["conflicting_duplicate_rows"] += 1
            hours[stamp][1].add(categories)
        else:
            hours[stamp] = (current, {categories})
    for stamp in invalid:
        hours.pop(stamp, None)
    examples = []
    for stamp in sorted(hours):
        split = "train" if stamp < dev else "development"
        beginning, end = (start, dev) if split == "train" else (dev, sealed)
        times = tuple(stamp - timedelta(hours=offset) for offset in reversed(range(protocol["history_hours"] + 1)))
        due = stamp + timedelta(hours=protocol["horizon_hours"])
        if times[0] < beginning or due >= end or any(time not in hours for time in (*times, due)):
            audit["excluded_cutoffs_missing_or_cross_split"] += 1
            continue
        current, categories = hours[stamp]
        _, (temperature, rain, snow, clouds), holiday = current
        weather = (int(temperature // 5), int(rain > 0), int(snow > 0), int(clouds // 25),
                   holiday, tuple(sorted(categories)))
        volumes = tuple(hours[time][0][0] for time in times)
        label = int(hours[due][0][0] >= protocol["future_volume_threshold"])
        examples.append(TrafficExample(split, stamp, due, volumes, weather, label))
    counts = {}
    for split in ("train", "development"):
        selected = [example for example in examples if example.split == split]
        counts[split] = {"positive": sum(e.label for e in selected), "negative": sum(1 - e.label for e in selected),
                         "days": len({e.cutoff.date() for e in selected}), "examples": len(selected)}
    support = all(min(counts[split]["positive"], counts[split]["negative"]) >= protocol[key]
                  and counts[split]["days"] >= protocol[day_key]
                  for split, key, day_key in (("train", "minimum_train_per_class", "minimum_train_days"),
                                             ("development", "minimum_development_per_class", "minimum_development_days")))
    audit.update(valid_unique_hours=len(hours), invalid_hours=len(invalid))
    return examples, {"access": dict(audit), "counts": counts, "support_passed": support,
                      "sealed_numeric_values_read": 0}


def contexts(volumes, weather, cutoff, arm, protocol):
    """Project only historical/current observations into four branch-local keys."""
    if arm not in protocol["arms"] or len(volumes) != protocol["history_hours"] + 1:
        raise ValueError("Invalid arm or bounded history")
    if any(type(value) is not int or not 0 <= value <= 20000 for value in volumes):
        raise ValueError("Invalid traffic history")
    present = volumes[-1]
    weekend = int(cutoff.weekday() >= 5)
    clock = (cutoff.hour, weekend, (cutoff.month - 1) // 3)
    common = (present // protocol["volume_bin_width"], cutoff.hour // 3, weekend)
    full_clock = (cutoff.hour, cutoff.weekday(), cutoff.month)
    snapshot = (present // protocol["volume_bin_width"], full_clock, weather)
    if arm == "current_exact":
        specialized = (present, full_clock)
    elif arm == "current_weather":
        specialized = snapshot
    elif arm == "history_histogram":
        specialized = (snapshot, tuple(sorted(value // protocol["volume_bin_width"] for value in volumes[:-1])))
    else:
        past = tuple(reversed(volumes[:-1])) if arm == "temporal_reversed" else volumes[:-1]
        def trend(previous):
            delta = present - previous
            magnitude = min(protocol["maximum_absolute_trend_bin"], abs(delta) // protocol["trend_bin_width"])
            return -magnitude if delta < 0 else magnitude
        specialized = (snapshot, trend(past[-1]), trend(past[0]))
    return tuple(json.dumps([index, context], separators=(",", ":")) for index, context in
                 enumerate(((), clock, common, specialized)))


def contrasts(training, protocol):
    """Descriptive train-only current-context collisions; not causal matched pairs."""
    groups = defaultdict(lambda: defaultdict(set))
    for example in training:
        current = contexts(example.volumes, example.weather, example.cutoff, "current_weather", protocol)[-1]
        temporal = contexts(example.volumes, example.weather, example.cutoff, "temporal_pair", protocol)[-1]
        groups[current][example.label].add(temporal)
    differing = sum(bool(labels[0] and labels[1] and len(labels[0] | labels[1]) > 1)
                    for labels in groups.values())
    return {"current_contexts": len(groups), "opposite_outcome_temporal_context_groups": differing,
            "passed": differing >= protocol["minimum_current_contrast_groups"]}


class LocalConjunctionCounts:
    """Bounded delayed counters; nested posteriors preserve local conjunctions."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.counts = {}
        self.pending = None
        self.last_time = None
        self.writes = self.changed = self.peak_pending = 0
        self.route_allowance = 0

    def _time(self, now):
        if self.last_time is not None and now < self.last_time:
            raise ValueError("Clock must be monotonic")

    def predict(self, keys, cutoff):
        self._time(cutoff)
        if self.pending is not None or len(keys) != self.protocol["local_updates_per_outcome"] or len(set(keys)) != len(keys):
            raise ValueError("Pending receipt or route-count violation")
        if len(self.counts) + sum(key not in self.counts for key in keys) > self.protocol["maximum_count_routes"]:
            raise ValueError("Local route capacity exceeded")
        if any(not isinstance(key, str) or len(key) > 4096 for key in keys):
            raise ValueError("Unbounded local context")
        additions = sum(2048 + 4 * len(key) for key in keys if key not in self.counts)
        if 163840 + self.route_allowance + additions > self.protocol["maximum_state_bytes_per_arm"]:
            raise ValueError("Local state capacity exceeded before allocation")
        probability = 0.5
        for key in keys:
            negative, positive = self.counts.get(key, (0, 0))
            strength = self.protocol["posterior_prior_strength"]
            probability = (positive + strength * probability) / (negative + positive + strength)
        receipt = (cutoff, cutoff + timedelta(hours=self.protocol["horizon_hours"]), keys, probability)
        self.pending = receipt
        self.last_time = cutoff
        self.peak_pending = 1
        return receipt

    def deliver(self, receipt, now, outcome_supplier, *, training):
        self._time(now)
        if receipt is not self.pending or now != receipt[1]:
            raise ValueError("Validate owned receipt and due time before reading outcome")
        label = outcome_supplier()
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Invalid observed binary outcome")
        if training:
            for key in receipt[2]:
                if key not in self.counts:
                    self.route_allowance += 2048 + 4 * len(key)
                counts = self.counts.setdefault(key, [0, 0])
                old = counts[label]
                counts[label] += 1
                self.changed += int(counts[label] != old)
                self.writes += 1
        self.pending = None
        self.last_time = now
        return label

    def checkpoint(self):
        return {key: list(pair) for key, pair in sorted(self.counts.items())}

    def restore(self, checkpoint):
        if self.counts or self.pending is not None or self.last_time is not None:
            raise ValueError("Restore requires a fresh model")
        if len(checkpoint) > self.protocol["maximum_count_routes"]:
            raise ValueError("Checkpoint capacity exceeded")
        restored = {}
        for key, pair in checkpoint.items():
            decoded = json.loads(key)
            if (not isinstance(decoded, list) or len(decoded) != 2 or type(decoded[0]) is not int
                    or not 0 <= decoded[0] < 4 or len(key) > 4096 or len(pair) != 2
                    or any(type(value) is not int or not 0 <= value <= 100000 for value in pair)):
                raise ValueError("Invalid checkpoint route/counts")
            restored[key] = list(pair)
        allowance = sum(2048 + 4 * len(key) for key in restored)
        if 163840 + allowance > self.protocol["maximum_state_bytes_per_arm"]:
            raise ValueError("Checkpoint state capacity exceeded")
        self.counts = restored
        self.route_allowance = allowance


def shuffled_targets(training, protocol):
    days = defaultdict(list)
    for index, example in enumerate(training):
        days[str(example.cutoff.date())].append(index)
    targets = {}
    for day, indices in sorted(days.items()):
        ranked = sorted(indices, key=lambda i: sha256(f'{protocol["shuffle_seed"]}:{day}:{i}'.encode()).digest())
        for destination, source in zip(indices, ranked):
            targets[training[destination].cutoff] = training[source].label
    return targets


def run_phase(model, examples, arm, protocol, *, training, targets=None, persistence=False):
    """One pending owned receipt; due outcomes released before same-time predictions."""
    pending = None
    trace = sha256()
    losses = defaultdict(lambda: [0., 0])
    confusion = [0, 0, 0, 0]
    before = model.checkpoint()
    writes_before = model.writes
    start = process_time()
    def release(receipt, example):
        if receipt[1] != example.due:
            raise ValueError("Outcome/receipt delay mismatch")
        label = model.deliver(receipt, example.due,
                              lambda: targets[example.cutoff] if targets is not None else example.label,
                              training=training)
        if not training:
            probability = float(example.volumes[-1] >= protocol["future_volume_threshold"]) if persistence else receipt[3]
            losses[str(example.cutoff.date())][0] += (probability - label) ** 2
            losses[str(example.cutoff.date())][1] += 1
            confusion[2 * label + int(probability >= .5)] += 1
            trace.update(json.dumps([example.cutoff.isoformat(), label, probability], separators=(",", ":")).encode())
    for example in examples:
        if example.split != ("train" if training else "development"):
            raise ValueError("Invalid learning/evaluation split")
        if pending is not None:
            if pending[0][1] > example.cutoff:
                raise ValueError("Hourly target overlap exceeds registered pending bound")
            release(*pending)
        keys = contexts(example.volumes, example.weather, example.cutoff, arm, protocol)
        pending = (model.predict(keys, example.cutoff), example)
    if pending is not None:
        release(*pending)
    writes = model.writes - writes_before
    if writes != (4 * len(examples) if training else 0) or (not training and model.checkpoint() != before):
        raise ValueError("Actual writes/frozen-state integrity failed")
    state_bound = 163840 + model.route_allowance
    measured = _deep_size(model)
    if state_bound > protocol["maximum_state_bytes_per_arm"] or measured > state_bound:
        raise ValueError("Model state bound exceeded")
    daily = {day: loss / count for day, (loss, count) in sorted(losses.items())}
    return {"examples": len(examples), "actual_counter_writes": writes,
            "changed_counters_total": model.changed, "peak_pending": model.peak_pending,
            "pending_after_phase": int(model.pending is not None), "count_routes": len(model.counts),
            "state_bound_bytes": state_bound, "measured_state_bytes_at_end": measured,
            "history_value_visits": 4 * len(examples), "prediction_count_visits": 4 * len(examples),
            "checkpoint_instrumentation_entries": len(before) + len(model.counts),
            "cpu_seconds_including_instrumentation": process_time() - start,
            "frozen": not training, "day_brier": daily,
            "equal_day_brier": None if training else sum(daily.values()) / len(daily),
            "pooled_brier": None if training else sum(loss for loss, _ in losses.values()) / len(examples),
            "balanced_accuracy": None if training else (confusion[0] / sum(confusion[:2]) + confusion[3] / sum(confusion[2:])) / 2,
            "confusion_tn_fp_fn_tp": confusion, "prediction_sha256": None if training else trace.hexdigest()}


def decision(arms, persistence, protocol):
    metrics = {name: row["development"] for name, row in arms.items()}
    metrics["persistence"] = persistence
    reference = min(protocol["current_references"], key=lambda name: metrics[name]["equal_day_brier"])
    temporal = metrics["temporal_pair"]
    days = sorted(temporal["day_brier"])
    if any(sorted(row["day_brier"]) != days for row in metrics.values()):
        raise ValueError("Paired day mismatch")
    gains = [metrics[reference]["day_brier"][day] - temporal["day_brier"][day] for day in days]
    rng = random.Random(protocol["bootstrap_seed"])
    draws = sorted(sum(rng.choice(gains) for _ in days) / len(days) for _ in range(protocol["bootstrap_replicates"]))
    gates = protocol["gates"]
    checks = {
        "history_gain": sum(gains) / len(days) >= gates["minimum_reference_day_brier_gain"],
        "balanced_accuracy_gain": temporal["balanced_accuracy"] - metrics[reference]["balanced_accuracy"] >= gates["minimum_balanced_accuracy_gain"],
        "paired_day_uncertainty": draws[100] > gates["minimum_bootstrap_lower_gain"],
        "order_reversal": metrics["temporal_reversed"]["equal_day_brier"] - temporal["equal_day_brier"] >= gates["minimum_reversal_day_brier_gain"],
        "shuffled_outcomes": metrics["temporal_shuffled"]["equal_day_brier"] - temporal["equal_day_brier"] >= gates["minimum_shuffled_day_brier_gain"],
        "matched_changed_updates": len({row["training"]["actual_counter_writes"] for row in arms.values()}) == 1 and all(row["training"]["actual_counter_writes"] == row["training"]["changed_counters_total"] for row in arms.values()),
        "frozen_exact_replay": all(row["checkpoint_replay_exact"] and row["development"]["actual_counter_writes"] == 0 for row in arms.values()),
    }
    return {"best_current_reference": reference, "reference_day_brier_gain": sum(gains) / len(days),
            "paired_day_lower_95_gain": draws[100], "gates": checks, "passed": all(checks.values()),
            "sealed_scoring_authorized": False, "spike_benefit_claimed": False}

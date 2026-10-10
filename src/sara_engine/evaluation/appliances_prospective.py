"""Causal, sealed extraction for a measured future appliance-demand task."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DemandExample:
    split: str
    cutoff: datetime
    outcome_time: datetime
    history: tuple[float, ...]
    label: int

    def controls(self) -> dict[str, tuple]:
        # Changes carry positions so the non-spiking reference loses no timing.
        changes = [(0, self.history[0])]
        changes.extend((i, value) for i, value in enumerate(self.history[1:], 1)
                       if value != self.history[i - 1])
        return {
            "current_value": (self.history[-1],),
            "histogram": tuple(sorted(Counter(self.history).items())),
            "full_ordinal": self.history,
            "transition_gaps": tuple(changes),
        }


def extract(rows: Iterable[Sequence[str]], protocol: dict) -> tuple[list[DemandExample], dict]:
    """Inspect timestamp before numeric cells; never inspect sealed values.

    Outcome values are read into a separate chronological tape. Only the
    cutoff-ended slice reaches feature controls; future values supply labels.
    Histories and exact future observations must remain in the same split.
    """
    parse = datetime.fromisoformat
    start, dev, sealed = (parse(protocol[key]) for key in
                          ("train_start", "development_start", "sealed_start"))
    n, horizon = protocol["history_samples"], protocol["horizon_samples"]
    step = timedelta(minutes=protocol["sample_minutes"])
    if not start < dev < sealed or n < 2 or horizon < 1 or step <= timedelta(0):
        raise ValueError("Invalid frozen temporal boundaries")
    header = next(iterator := iter(rows))
    date_index, value_index = header.index("date"), header.index("Appliances")
    tape: dict[datetime, tuple[str, float]] = {}
    previous = None
    sealed_rows = 0
    for row in iterator:
        stamp = parse(row[date_index])
        if previous is not None and stamp <= previous:
            raise ValueError("Source timestamps must be strictly increasing")
        previous = stamp
        if stamp >= sealed:
            sealed_rows += 1
            continue
        if stamp < start:
            continue
        value = float(row[value_index])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Energy observations must be finite and nonnegative")
        tape[stamp] = ("train" if stamp < dev else "development", value)
    examples = []
    for stamp, (split, current) in tape.items():
        times = [stamp - step * offset for offset in reversed(range(n))]
        outcome_time = stamp + step * horizon
        required = times + [stamp + step * offset for offset in range(1, horizon + 1)]
        if any(time not in tape or tape[time][0] != split for time in required):
            continue
        history = tuple(tape[time][1] for time in times)
        label = int(tape[outcome_time][1] >= current + protocol["increase_wh"])
        examples.append(DemandExample(split, stamp, outcome_time, history, label))
    counts = {}
    for split in ("train", "development"):
        subset = [example for example in examples if example.split == split]
        counts[split] = {
            "positive": sum(example.label for example in subset),
            "negative": sum(1 - example.label for example in subset),
            "days": len({example.cutoff.date() for example in subset}),
        }
    passed = all(
        min(counts[split]["positive"], counts[split]["negative"]) >= protocol[key]
        and counts[split]["days"] >= protocol[day_key]
        for split, key, day_key in (
            ("train", "minimum_train_per_class", "minimum_train_days"),
            ("development", "minimum_development_per_class", "minimum_development_days")))
    return examples, {"counts": counts, "support_passed": passed,
                      "sealed_rows_timestamp_only": sealed_rows,
                      "numeric_rows_read": len(tape), "learner_scored": False}

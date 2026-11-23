"""Aggregate development-only availability and exact-key support for KSEA."""

from collections import Counter, defaultdict

from sara_engine.evaluation.asos_ksea_delayed_contract import delayed_outcomes
from sara_engine.evaluation.asos_ksea_first_sensor import first_sensor_minutes


def summarize(months):
    """Consume the registered streams and return counts without publishing examples."""
    valid_dates = set()
    class_dates = {False: set(), True: set()}
    indexed = valid = 0

    def counted_minutes():
        nonlocal indexed, valid
        for minute in first_sensor_minutes(months):
            indexed += 1
            if minute.extinction is not None:
                valid += 1
                valid_dates.add(minute.at.date())
            yield minute

    keys = defaultdict(Counter)
    classes = Counter()
    for outcome in delayed_outcomes(counted_minutes()):
        keys[outcome.current_key][outcome.history_direction, outcome.positive] += 1
        classes[outcome.positive] += 1
        class_dates[outcome.positive].add(outcome.prediction_at.date())

    repeated = both = contrasting = opposing = supported = 0
    for cells in keys.values():
        repeated += sum(cells.values()) > 1
        both += all(any(cells[d, side] for d in (-1, 0, 1)) for side in (False, True))
        rising = cells[1, False] + cells[1, True]
        falling = cells[-1, False] + cells[-1, True]
        contrasting += bool(rising and falling)
        has_opposite = bool(cells[-1, False] and cells[1, True] or cells[-1, True] and cells[1, False])
        opposing += has_opposite
        supported += has_opposite and min(rising, falling) >= 80

    eligible_dates = len(class_dates[False] | class_dates[True])
    return {
        "indexed_minutes": indexed,
        "missing_rows": 182 * 1440 - indexed,
        "unrecognized_visibility": indexed - valid,
        "valid_minutes": valid,
        "valid_dates": len(valid_dates),
        "eligible_prediction_dates": eligible_dates,
        "coverage_180": eligible_dates >= 180,
        "negative": classes[False],
        "positive": classes[True],
        "negative_dates": len(class_dates[False]),
        "positive_dates": len(class_dates[True]),
        "exact_keys": len(keys),
        "repeated_keys": repeated,
        "keys_with_both_classes": both,
        "keys_with_rising_and_falling": contrasting,
        "keys_with_opposite_outcomes_across_directions": opposing,
        "opposing_keys_with_80_per_direction": supported,
    }

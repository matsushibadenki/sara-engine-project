def reconstruct(tick, history, weighting, allocated, left, right, outcome):
    if not isinstance(tick, int) or isinstance(tick, bool) or not 0 <= tick <= 1023:
        raise ValueError("Tick must be an integer from 0 through 1023.")
    for mass in (left, right):
        if not isinstance(mass, int) or isinstance(mass, bool) or not 0 <= mass <= 6717440:
            raise ValueError("Mass must be an integer from 0 through 6717440.")
    if not isinstance(history, str) or history not in ("cumulative", "discounted"):
        raise ValueError("Invalid history rule.")
    if not isinstance(weighting, str) or weighting not in ("uniform", "public_exponential"):
        raise ValueError("Invalid error weighting.")
    if not isinstance(allocated, bool):
        raise ValueError("Allocation must be a boolean.")
    if not isinstance(outcome, str) or outcome not in ("mass_left", "mass_right"):
        raise ValueError("Invalid outcome side.")

    scale = 2 ** (tick // 64)
    prior = 1 if history == "cumulative" else scale
    if allocated:
        adjusted_right = right + prior
        total = left + prior + adjusted_right
        probability = adjusted_right / total
        error_probability = probability
    else:
        if left != 0 or right != 0:
            raise ValueError("An unallocated key requires zero masses.")
        probability = None
        error_probability = 0.5

    difference = error_probability - int(outcome == "mass_right")
    squared_difference = difference * difference
    error_weight = 1.0 if weighting == "uniform" else float(scale)
    return probability, squared_difference * error_weight

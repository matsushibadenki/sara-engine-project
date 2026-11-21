def reconstruct(tick, history, weighting, allocated, left, right, outcome):
    if type(tick) is not int or not 0 <= tick <= 1023:
        raise ValueError("Invalid tick")
    if type(history) is not str or history not in ("cumulative", "discounted"):
        raise ValueError("Invalid history")
    if type(weighting) is not str or weighting not in ("uniform", "public_exponential"):
        raise ValueError("Invalid weighting")
    if type(allocated) is not bool:
        raise ValueError("Invalid allocation flag")
    if type(left) is not int or not 0 <= left <= 6717440:
        raise ValueError("Invalid left mass")
    if type(right) is not int or not 0 <= right <= 6717440:
        raise ValueError("Invalid right mass")
    if type(outcome) is not str or outcome not in ("mass_left", "mass_right"):
        raise ValueError("Invalid outcome")
    if not allocated and (left != 0 or right != 0):
        raise ValueError("Unallocated keys require zero masses")

    public_scale = 1 << (tick // 64)
    if allocated:
        prior = 1 if history == "cumulative" else public_scale
        numerator = right + prior
        denominator = left + right + 2 * prior
        probability = float(numerator) / float(denominator)
        error_probability = probability
    else:
        probability = None
        error_probability = 0.5

    binary_outcome = 0.0 if outcome == "mass_left" else 1.0
    residual = error_probability - binary_outcome
    squared_residual = residual * residual
    weight = 1.0 if weighting == "uniform" else float(public_scale)
    error_increment = squared_residual * weight
    return probability, error_increment

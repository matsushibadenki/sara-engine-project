"""Reconstruct local arithmetic from the registered written specification."""


def reconstruct(tick, history, weighting, allocated, left, right, outcome):
    """Return the probability and receipt-time error increment."""
    if type(tick) is not int or not 0 <= tick <= 1023:
        raise ValueError("Tick must be an integer from 0 through 1023")
    if any(type(mass) is not int or not 0 <= mass <= 6717440 for mass in (left, right)):
        raise ValueError("Invalid class mass")
    if type(allocated) is not bool or (not allocated and (left or right)):
        raise ValueError("Invalid allocation or mass for an unallocated key")
    if type(history) is not str or history not in ("cumulative", "discounted"):
        raise ValueError("Invalid history rule")
    if type(weighting) is not str or weighting not in ("uniform", "public_exponential"):
        raise ValueError("Invalid error weighting")
    if type(outcome) is not str or outcome not in ("mass_left", "mass_right"):
        raise ValueError("Invalid outcome")

    scale = 2 ** (tick // 64)
    prior = scale if history == "discounted" else 1
    probability = (right + prior) / (left + right + 2 * prior) if allocated else None
    error_probability = 0.5 if probability is None else probability
    target = int(outcome == "mass_right")
    weight = float(scale) if weighting == "public_exponential" else 1.0
    return probability, (error_probability - target) ** 2 * weight

"""Independent, stateless scalar arithmetic for literal hand records only."""
from math import isfinite


def calculate(tick, history, error_weight, allocated, n0, n1, side):
    """Return a fresh calculation; expected values are not kernel inputs."""
    if type(tick) is not int or not 0 <= tick < 1024:
        raise ValueError('Invalid public tick')
    if type(history) is not str or history not in ('cumulative', 'discounted'):
        raise ValueError('Unknown count rule')
    if type(error_weight) is not str or error_weight not in ('uniform', 'public_exponential'):
        raise ValueError('Unknown error rule')
    if type(allocated) is not bool:
        raise ValueError('Allocation must be explicit')
    for mass in (n0, n1):
        if type(mass) is not int or not 0 <= mass <= 6717440:
            raise ValueError('Invalid integer evidence mass')
    if type(side) is not str or side not in ('mass_left', 'mass_right'):
        raise ValueError('Unknown credited side')

    public_scale = 1 << (tick // 64)
    prior_size = public_scale if history == 'discounted' else 1
    if allocated:
        numerator = n1 + prior_size
        denominator = (n0 + n1) + (prior_size + prior_size)
        probability = numerator / denominator
        residual_probability = probability
    else:
        if n0 != 0 or n1 != 0:
            raise ValueError('Unallocated key cannot contain evidence')
        probability = None
        residual_probability = 0.5

    target = 0 if side == 'mass_left' else 1
    difference = residual_probability - target
    squared_difference = difference ** 2
    scale = public_scale * 1.0 if error_weight == 'public_exponential' else 1.0
    increment = scale * squared_difference
    if not isfinite(increment) or increment <= 0:
        raise ValueError('Nonpositive or nonfinite residual increment')
    return {'prior': prior_size, 'probability': probability,
            'error_weight': scale, 'error_increment': increment,
            'residual_fallback': not allocated}

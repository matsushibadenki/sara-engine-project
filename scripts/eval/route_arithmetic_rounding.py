"""Compare the two frozen reconstructions over the registered binary64 cases."""
import json
from fractions import Fraction
from itertools import product
import platform

from sara_engine.evaluation.route_arithmetic_reconstruction import reconstruct as power
from sara_engine.evaluation.route_blind_arithmetic import reconstruct as multiply


def audit():
    probability_mismatches = 0
    increment_mismatches = 0
    multiplication_matches_oracle = 0
    first_mismatch = None
    for left, right, side in product(range(64), range(64), ("mass_left", "mass_right")):
        args = (0, "cumulative", "uniform", True, left, right, side)
        probability, multiplied = multiply(*args)
        other_probability, powered = power(*args)
        probability_mismatches += probability.hex() != other_probability.hex()
        if multiplied.hex() == powered.hex():
            continue
        increment_mismatches += 1
        residual = probability - int(side == "mass_right")
        oracle = float(Fraction(residual) ** 2)
        multiplication_matches_oracle += multiplied.hex() == oracle.hex()
        if first_mismatch is None:
            first_mismatch = {
                "left": left, "right": right, "side": side,
                "probability": probability.hex(), "multiplication": multiplied.hex(),
                "power": powered.hex(), "oracle": oracle.hex(),
            }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": 64 * 64 * 2,
        "probability_mismatches": probability_mismatches,
        "increment_mismatches": increment_mismatches,
        "multiplication_matches_oracle_on_mismatch": multiplication_matches_oracle,
        "first_increment_mismatch": first_mismatch,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))

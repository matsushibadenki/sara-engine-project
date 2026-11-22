# Local arithmetic reconstruction audit v1

The reconstruction takes seven fields in order: tick, history rule, error weighting, allocation flag, left mass, right mass and outcome side. Accept integer ticks from 0 through 1023 and integer masses from 0 through 6717440. Booleans are valid only for allocation. History is `cumulative` or `discounted`. Error weighting is `uniform` or `public_exponential`. Outcome side is `mass_left` or `mass_right`.

Let the public scale be two raised to the integer quotient of tick divided by 64. Add prior mass of one to each class under cumulative history, or the public scale to each class under discounted history. Divide the resulting right mass by the total for an allocated key. An unallocated key requires zero masses, returns a null probability and uses one half for error arithmetic. For the error increment, subtract the binary outcome (left = 0, right = 1) from the probability used for error, square the difference and multiply by either 1.0 or the public scale converted to float. Use Python binary64 operations in that order. Reject invalid inputs. Return probability and error increment.

Register these expectations before implementation. Compare the eleven existing literal records in `route_error_binding_preflight_selection_v1.json` and the boundary rows below with exact type and binary64 equality. Hex values use Python's float notation.

| Tick | History | Weight | Allocated | Left | Right | Outcome | Probability | Increment |
|---|---|---|---|---|---|---|---|---|
| 63 | discounted | public_exponential | yes | 2 | 0 | left | 0x1p-2 | 0x1p-4 |
| 64 | discounted | public_exponential | yes | 4 | 0 | right | 0x1p-2 | 0x1.2p+0 |
| 1023 | cumulative | public_exponential | no | 0 | 0 | left | null | 0x1p+13 |
| 1023 | cumulative | public_exponential | yes | 0 | 0 | left | 0x1p-1 | 0x1p+13 |

The reconstruction uses the written rules without importing existing arithmetic helpers. Verification uses hand records only. Historical results and sealed test inputs are outside the audit.

## Rounding audit v2 preregistration

Use every ordered pair of integer masses from 0 through 63, both outcome sides, tick 0, cumulative history, uniform weighting and allocated keys. This yields 8,192 cases including probabilities with non-power-of-two reduced denominators. Compare the probability and error increment by binary64 hexadecimal representation between the frozen blind and standalone implementations. Count mismatches separately, report the first error mismatch in left-major, right-minor, left-outcome-first order, and record the runtime. Use exact rational arithmetic on the already rounded residual to identify the correctly rounded square for any mismatch. Preserve both implementations and the prior results regardless of the outcome.

### Rounding audit results

On CPython 3.10.22 (macOS 27.2.1 arm64), all 8,192 probabilities matched and eight increments differed. Multiplication matched the rational oracle in all eight mismatches. The first case uses masses 6 and 47 with a right outcome. Its probability is 48/55 before rounding, represented by `0x1.bed61bed61bedp-1`. The multiplication and oracle yield `0x1.0964cac4d84f7p-6`. Power yields `0x1.0964cac4d84f6p-6`, one ULP lower.

The v1 instruction to square the residual admits both Python expressions. Their equivalence on the 15 original cases leaves the operator choice unresolved for bitwise reproduction. A future version should specify the exact operator and rounding order. Multiplication is the candidate when the intended result is a correctly rounded square of the binary64 residual. Historical implementations and expectations are retained as evidence. The audit generated inputs from the registered range and did not open sealed test data.

Reproduce from the repository root with `PYTHONPATH=src workspace/.venv310/bin/python scripts/eval/route_arithmetic_rounding.py`.

## Local arithmetic specification v2

The v2 implementation keeps the v1 input domain, allocation rules and output types. Compute prior, numerator and denominator as exact integers. Round the probability division to binary64, nearest with ties to even. For an unallocated key, return null probability and use exact 0.5 in the residual. Subtract the binary outcome and round the residual to binary64. Square that rounded residual with one binary64 multiplication `residual * residual`. Round this product before multiplying it by the exactly representable binary64 weight and rounding again. Each rounding uses nearest with ties to even.

Verify the eleven literal records, four registered boundaries and 8,192 registered rounding cases. Build an independent oracle from exact rational arithmetic, converting to binary64 after each specified operation. Compare type and hexadecimal representation for both outputs. Also retain the original literal expectations as checks on the first fifteen cases. The historical specifications, implementations, negative results and sealed test inputs remain outside this change.

### v2 limit and transition audit preregistration

Use ticks 0 and 1023 and both sides of every scale transition (64k − 1 and 64k for k = 1 through 15). For allocated keys, cross left and right masses independently over 0, 1, 6717439 and 6717440. Include an unallocated zero-mass case at each tick. Cross all 17 allocation/mass cases with both history rules, both error weights and both outcome sides. This yields 4,352 cases per implementation. Compare both frozen v2 reconstructions with the existing rational reference by exact type and binary64 hexadecimal representation for each output. Report discrepancies without modifying the implementations, historical evidence or sealed inputs.

### v2 invalid input audit preregistration

Start from tick 0, cumulative history, uniform weighting, an allocated key, zero masses and a left outcome. Replace tick with −1, 1024, False and True in separate cases. For each mass field independently, replace it with −1, 6717441, False and True. For each of history, weighting and outcome, replace it with the string `unknown`. Finally use an unallocated key and set the masses to (1, 0), (0, 1) and (1, 1). Both frozen v2 implementations must raise ValueError in all 18 cases per implementation. Verify the allocated and unallocated zero-mass base cases still agree with the rational reference. Use only generated inputs.

### v2 invalid type audit preregistration

Use the prior audit's allocated zero-mass base. Change one field at a time. For each of tick, left mass and right mass, use 0.0, the string `0`, NaN, positive infinity and negative infinity. For allocation use integer 0, integer 1, float 0.0, the string `true` and null. For each of history, weighting and outcome use integer 0, True, null and empty bytes. Both frozen implementations must raise ValueError on all 32 cases per implementation. Generate all inputs locally.

### v2 subclass and coercion audit preregistration

Use the allocated zero-mass base from the invalid type audit. Substitute one field at a time. Test a subclass of int carrying the valid base value in each of the three integer fields and a subclass of str carrying the valid base value in each of the three selector fields. Then substitute a conversion probe in each of the seven fields. The probe raises AssertionError if Python invokes its integer, index, float, string, truth-value or equality method. Both frozen implementations must raise ValueError on all 13 cases per implementation without invoking a probe method. Generate all inputs locally.

### v2 all-tick audit preregistration

For each tick from 0 through 1023, use allocated mass pairs (6, 47), (47, 6), (0, 6717440) and (6717440, 0). Cross them with both history rules, both error weights and both outcomes. This yields 32,768 cases per frozen v2 implementation. Compare probability and error increment with the existing rational reference using exact types and binary64 hexadecimal forms. The asymmetric small masses exercise non-dyadic probabilities, including the earlier rounding counterexample. The asymmetric maximum masses exercise probabilities near zero and one. The audit uses generated inputs.

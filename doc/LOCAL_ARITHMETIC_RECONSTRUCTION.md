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

# Bounded Local Outcome Learning

## Decision — 2026-09-16

Use `BoundedLocalOutcomeReadout` with its default `integral_rate=0.0` as an experimental readout for changing or noisy mappings. This is a scalar prediction-error update with bounded eligibility, rather than a full PC-ALM implementation. The optional error integrator is retained only as a reproducible ablation. A later fixed-topology integration did not justify replacing the existing three-factor rule; the existing R1 implementation and results are preserved.

The user authorized method selection and implementation. A component comparison can improve a local readout without reopening paused topology expansion or asserting that R2 passed. This module is explicitly instantiated; it is not connected automatically to normal chat or durable evidence.

## Local contract

For active routes `j`, eligibility `e_j > 0`, and local weights in `[-1, 1]`:

```text
prediction = sum(weight_j * e_j) / sum(e_j)
residual = outcome - saved_prediction
delta_j = clip(rate * trace_decay**feedback_age * e_j / sum(e_j) * residual)
weight_j = clip(weight_j + delta_j, -1, 1)
```

This is a conventional local delta rule; it does not differentiate through an encoder, spike history or whole network. Multiplication of local scalars is allowed. There is no matrix calculation, GPU dependency or backward computation graph. The caller owns the meaning and verification of the outcome. Valid numeric input alone is not proof that a label is true.

The class owns one local readout and one pending prediction. An immutable receipt captures the score and eligible routes before feedback. The same receipt object must be returned exactly once to `observe` or `discard`. Copied, foreign, consumed and stale receipts cannot update the model. Feedback later than `max_age` consumes the receipt without changing weights or integrals. Untrusted/missing feedback can be discarded explicitly.

Defaults: 256 retained routes, 8 active routes per decision, maximum feedback age 64 logical ticks, update magnitude 0.5, and weights bounded to `[-1, 1]`. At most nine iterator probes detect active-input overflow before persistent mutation. Capacity exhaustion raises an explicit error; there is no silent eviction. The protocol uses 32 retained routes and presents 16. Identifiers, counters and timestamps have finite integer ranges. Status strings support English, Japanese and Simplified Chinese. Instances are single-owner and in-memory; checkpointing and concurrent callers are outside this contract.

## Usage

```python
from sara_engine.learning import BoundedLocalOutcomeReadout

readout = BoundedLocalOutcomeReadout()
prediction = readout.predict([(42, 1.0), (71, 0.5)], time=10)
decision = prediction.score > 0.0
# The independently observed outcome arrives after the prediction.
update = readout.observe(prediction, outcome=1.0, time=14)
```

The component has no built-in spike encoder. Route IDs and eligibility must come from an explicit local producer. Repeated feedback for a different prediction is not permitted; applications needing overlapping decisions require a separately bounded design.

## Frozen experiment

Protocol: `data/processed/benchmark_fixtures/local_outcome_rule_v1.json`.
SHA-256: `021c32b811580693b8805fca6105b10b9ba796edb9b1f21a609daefcb4f7d6e2`.
Result: `workspace/evaluation/local_outcome_v1_result.json`.
Attempt/source hashes: `workspace/evaluation/local_outcome_v1_attempt.json`.

Five fresh seeds, four synthetic scenarios, six arms, 256 warmup and 1,024 scored decisions per seed/scenario/arm. This gives 20,480 scored inputs per arm, or 122,880 scored arm decisions. Accuracy and Brier loss are computed before each outcome is supplied; learning continues online during evaluation. Routes and inputs are identical across arms. This does not test unseen representations or independent real-world data.

The three-factor arm calls the existing `ThreeFactorLearningManager` with RPE disabled, as in R1, while matching learning rate, weight bounds, eligibility, delay and per-update cap to the other arms. It is a new matched component control, not a rerun of R1. The non-spiking scalar EMA performs the same scalar residual update without the receipt machinery; its prediction traces match the residual arm exactly.

| Scenario | Three-factor accuracy | Residual accuracy | Three-factor Brier | Residual Brier | Integrated Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stationary | 100.00% | 100.00% | 0.000000 | 0.000197 | 0.000116 |
| Rule reversal | 66.60% | 78.52% | 0.226599 | 0.153296 | 0.147299 |
| 20% label noise | 79.82% | 79.51% | 0.193664 | 0.173610 | 0.174975 |
| Expired feedback | 40.66% | 40.66% | 0.250000 | 0.250000 | 0.250000 |

Brier is squared error of the predicted probability; lower is better. No-update models predict score zero, so expired-feedback accuracy reflects label imbalance and the fixed negative tie decision, not learning. All expired-feedback arms made zero updates.

Across the three useful scenarios, mean Brier gain is 0.031054 for residual versus three-factor (threshold 0.01). Per-seed gains range from 0.029807 to 0.031836. The noisy accuracy reduction is 0.31 percentage points, inside the preregistered two-point tolerance. These are descriptive, synthetic observations; no statistical-significance or real-world superiority claim is made.

The integral ablation maintains a route-local signed residual accumulator, with rate 0.25, per-tick retention 0.999, cap 1 and maximum age 4,096. Its composite signal is `(residual + integral) / (1 + integral_rate)` so its first update matches the residual control. Its mean additional Brier gain is 0.001571, below the frozen 0.005 threshold. It also worsens noisy Brier, so the simpler residual rule is selected.

## Resources and evidence limits

The residual arm's maximum recursively measured maintained Python state was 3,696 bytes, including pending receipt and configuration, versus 4,598 for three-factor and 6,000 for integrated. Residual mean predict-plus-feedback latency was about 0.0042–0.0044 ms on the three useful scenarios; three-factor was about 0.0030–0.0032 ms and the scalar EMA about 0.0006–0.0007 ms. The residual API is not faster here. Resource gates passed for all arms.

Timing excludes diagnostic object traversal, report hashing and the externally supplied route generator. The probe measures a readout, not full SNN encoding, a deep network, event scheduling or chat. Reported memory is recursively measured Python object size, not process RSS or peak temporary allocation. No physical joule measurements were made. The same predictions from the scalar EMA explicitly prevent an SNN advantage claim.

## Fixed-topology follow-up

Validation: 166 focused tests pass across the local-outcome/revision contracts, three benchmarks and existing three-factor, R1, predictive-coding and adaptive-credit modules. This includes receipt replay/forgery, invalid input atomicity, expiry, bounded iterator consumption, capacity rejection, integral age/caps, monotonic revision signals, deterministic replay, public lazy exports, actual `Neuron` spike-to-readout updates and fixed-topology controls. Evaluated source hashes still match the saved results. The complete repository suite passes 2,134 tests.

[Done] The fixed-topology integration was frozen and executed. Both rules reached 100% accuracy on the stable mapping, while existing three-factor learning reached Brier 0.000000 and residual learning reached 0.008018. The replacement gate failed. See [LOCAL_OUTCOME_SNN_INTEGRATION.md](LOCAL_OUTCOME_SNN_INTEGRATION.md).

[Next] Define a new independently useful nonstationary task where change is observable without hidden labels. Any R2 claim still requires causal timing and independent usefulness. An isolated readout improvement cannot waive those conditions.

日本語: 蓄積器なしの局所予測誤差更新を、変化・ノイズ向けの実験部品として採用しました。予測と結果を一対一で結び付け、期限切れ・二重更新・容量超過を防ぎます。固定対応の統合評価では現行三因子則を置き換えられませんでした。非スパイクの同一式とも結果が一致しており、本番統合とSNN優位性は未成立です。

简体中文：无积分器的局部预测误差更新仅作为变化和噪声条件下的实验组件，并限制过期、重复更新和容量溢出。固定映射集成评估未能支持替换现有三因子规则；它也与相同公式的非脉冲基线完全一致，因此尚未成立生产集成或SNN优势。

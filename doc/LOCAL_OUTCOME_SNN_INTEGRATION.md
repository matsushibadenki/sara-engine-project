# Fixed-Topology SNN Readout Integration

## Decision — 2026-09-16

[Done] The frozen integration ran once and retained a negative replacement result. Keep the existing three-factor rule as the fixed stable-map control. Keep `BoundedLocalOutcomeReadout` isolated and opt-in for future nonstationary research. Do not connect it to normal chat, verified evidence, durable memory or production learning.

This decision does not undo the earlier component result. Prediction-error updates adapted better to reversals and produced better noisy probability estimates there. The integration task tested a different property: repeated stable mappings after every address had been trained. Existing three-factor learning was optimal on that narrow task.

## Frozen protocol and result

- Protocol: `data/processed/benchmark_fixtures/local_outcome_snn_integration_v1.json`
- Protocol SHA-256: `2bcc62e0de66683299e71df749e0845771db6259bad272160d4ed0ee2a65708b`
- Attempt/provenance: `workspace/evaluation/local_outcome_snn_integration_v1_attempt.json`
- Result: `workspace/evaluation/local_outcome_snn_integration_v1_result.json`

Each of five fresh seeds uses 192 fixed addresses: eight left symbols × eight right symbols × three interval bins. Every address appears eight times in training and eight times in evaluation, in independently shuffled order. Labels are balanced and assigned after a seed-specific address shuffle. Evaluation identities are disjoint, while address identities are deliberately seen during training. This tests bounded readout integration and recall, not unseen representation generalization.

The intact SNN arms instantiate 192 actual `Neuron` objects. Each episode activates one addressed neuron's dendritic branch with two `0.8` currents and requires a real spike before readout. Dynamic sensory state resets between episodes. The scalar control directly calculates the identical address. Timing destruction preserves symbols, labels and episode order while replacing every interval with the short interval.

| Arm | Mean accuracy | Mean Brier | Maximum maintained bytes | Maximum event work |
| --- | ---: | ---: | ---: | ---: |
| SNN residual | 100.00% | 0.008018 | 149,183 | 6 |
| SNN three-factor | 100.00% | 0.000000 | 145,264 | 6 |
| SNN timing-destroyed residual | 74.38% | 0.197556 | 139,088 | 6 |
| Scalar residual | 100.00% | 0.008018 | 21,821 | 3 |
| SNN frozen | 50.00% | 0.250000 | 135,301 | 5 |
| SNN shuffled feedback | 51.97% | 0.262790 | 149,192 | 6 |

The residual rule's mean Brier gain over three-factor is `-0.008018`; all five seed gains are negative. The preregistered minimum was `+0.01`, so both the mean-gain and all-seed gates fail. There is no accuracy regression because both reach 100% accuracy.

The residual candidate beats shuffled feedback by 48.03 accuracy points. Timing destruction lowers accuracy by 25.63 points, exceeding the frozen 15-point threshold. This shows that the fixed encoder's interval address is functionally used on this designed workload. It does not establish that spike timing is uniquely useful: the intact scalar residual arm has the exact same prediction trace as the SNN residual arm.

All resource contracts passed. The SNN residual path carries about 6.8 times the maintained Python state and twice the counted event work of the scalar arm. These are Python object and operation proxies. Timing excludes state-size traversal and report hashing. No process RSS, temporary allocation, hardware energy or physical joules were measured.

Validation: 149 focused tests and the complete 2,117-test repository suite pass. An initial full-suite invocation intentionally placed temporary files under `workspace/`, which invalidated one test that verifies rejection of unmanaged output; that test passed in the normal system temporary directory, and the complete suite then passed under the normal condition. Both saved experiment reports still match their recorded source hashes.

## Interpretation

Three-factor learning adds a constant signed step for every correct repeated outcome and reaches the bounded target weight exactly. Residual learning scales its step by `target - prediction`; it approaches the target smoothly and remains slightly inside the bound after finite training. That behavior helps reversals and noisy targets but leaves a small calibration gap in a stable deterministic mapping.

Selecting between these rules using an evaluation label or retrospective knowledge of the scenario would leak information. A legitimate adaptive policy needs an observable local change signal specified before evaluation. Simply combining the rules after seeing this result is not authorized by the evidence.

## Next

[Done] The observable-revision task was preregistered and executed with fresh identities. See [OBSERVABLE_REVISION_POLICY.md](OBSERVABLE_REVISION_POLICY.md). Its early recovery passed, but its full post-revision Brier failed against always-residual.

[Next] Test a bounded larger residual step only during a supplied revision window, using fresh identities and the same false-notification and scalar-equivalence controls. Production remains unchanged regardless of a component pass.

日本語: 固定された対応を十分に反復する課題では、現行三因子則が正解率100%、Brier 0.000000で最良でした。予測誤差則は正解率100%でもBrier 0.008018のため、標準則への置換を見送ります。時間破壊による低下は確認できましたが、同じ番地を使う非スパイク対照と完全一致するため、SNN固有の優位性ではありません。

简体中文：在充分重复固定映射的任务中，现有三因子规则达到100%准确率和0.000000 Brier，表现最佳。预测误差规则虽也达到100%准确率，但Brier为0.008018，因此不替换默认规则。破坏时间信息会降低性能，但与使用相同地址的非脉冲对照完全一致，不能证明SNN特有优势。

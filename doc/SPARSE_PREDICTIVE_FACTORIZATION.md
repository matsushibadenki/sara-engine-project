# Sparse Predictive Factorization — SARA Research Proposal

## Status

🟢 [Done] A Memory Mosaics-inspired **provided-factor memory component** comparison is implemented and verified on five synthetic seeds (2026-10-10). Results below do not complete anonymous sparse predictive factorization.

🟢 [Done] A separate bounded learned-route study now covers independent/interacting Boolean transitions and nuisance shifts across ten seed/task cases. Correct routes can be selected without supplied input/output correspondence, but the registered input-overlap specialization gate fails. Preserve this negative result; do not promote the mechanism.

🟢 [Done] The actual-update-matched candidate-pruning study passes all registered component gates across two tasks × five fresh seeds. Every arm executes 4,608 outcome route updates. Residual pruning reduces active candidates from 45 to nine, preserves the required shared dependencies and 100% exact accuracy, and lowers mean shifted Brier by 92.02% versus round-robin. This is a stationary toy-task component result, not production promotion.

🟢 [Done] [Nonstationary candidate reactivation](SPARSE_ROUTE_REACTIVATION.md): separately preregistered 20 cases × five arms, each with 9,216 actual route updates / 36,864 changed learning fields. All ten changed cases fail 256-tick recovery; three fail exact new-dependency retention. Stationary false starts are zero. Whole-report replay fails (known tuple/list comparator defect; full numerical equality unknown). Preserve primary and consumed replay failure; 121 tests pass and 399 prior pins verify. No tuning, restart, SNN/resource promotion or sealed access.
🟢 [Done] [Saved-report serialization/equality preflight](SPARSE_ROUTE_REACTIVATION_CODEC.md): registered before implementation, parses only the saved primary/failure reports (465,308 bytes). All 19 checks and 35 tests pass; exact static replay preserves 412 hashes. Tuple/list interchange is restricted to 80 initial-pool paths; numeric/type/order/field differences are not hidden. Zero learner execution/scoring/acquisition/measurement. Original replay failure and negative gates remain unchanged; full numerical equality stays unknown.
🟢 [Done] [Fixed-condition numerical reproduction v2](SPARSE_ROUTE_NUMERIC_V2.md): separately registered new primary and fresh-process replication, unchanged learner/conditions/codec. All numerical payloads and registered historical-reference comparisons match exactly (only two explicit provenance-field bindings). Each phase runs 20 cases × five arms with matched 9,216 route updates / 36,864 changed learning fields per arm/case. All ten changed recovery failures remain negative. 56 tests and read-only proof verification pass; 422 prior pins and old failed replay/seals preserved. No SNN/resource promotion.
🟢 [Done] [Saved-report recovery/retention failure audit](SPARSE_ROUTE_FAILURE_AUDIT.md): preregistered seven inputs / 1,593,723 bytes, all 100 pool lifecycles reconcile. Trigger lag 6–13 and commit lag 186–193 ticks leave only 63–70 ticks to the fixed deadline; no claim that the original gate was impossible. Three exact-retention failures split into two high-accuracy cases and one poor-accuracy case; missing true-route identities/ranks/support and counterfactual effects remain unknown. 90 tests, exact pure replay and 432 prior pins; zero learner/scoring/acquisition/measurement/condition changes. Old failed replay, negatives and seals preserved.
🟢 [Done] [Prospective bounded route snapshot contract](SPARSE_ROUTE_SNAPSHOT_CONTRACT.md): external read-only all-45-candidate collector, pre/post selection invariants, recent-window count/error differences and evaluator-only dependency diagnostics implemented. Fixed saved-metadata plan: 160 future hooks; conditional encoded bound 1,376,417 bytes <2 MiB. Static preflight/replay and 127 tests pass, preserving 441 prior pins. Actual collection/learner/scoring remains zero; no causal attribution, SNN advantage or trial authorization.
🟢 [Done] [Fixed-condition real route capture v3](SPARSE_ROUTE_CAPTURE_V3.md): separately preregistered baseline/capture executions, unchanged science/collector. All numerical payloads match; 160 actual boundaries and 160 full-state read-only checks pass. Missing dependencies have 72/412/72 observations and commit ranks 4/8/4: score exclusions, not ties or original 16-observation failure. Window re-ranking does not recover them; causal/counterfactual conclusions remain unknown. 143 tests, 450 prior pins, original negatives/replay failure/seals preserved; no promotion/resource claim.
⭕️ [Pending] Further mechanism changes require separately justified causal/task evidence and a new protocol. No automatic tuning, old-attempt restart or additional unregistered run. Completed v3 evidence collection does not reopen nonstationary promotion or the authoritative R2 independent-task/resource holds.

🔴 [Later] Structural growth and runtime integration require independent evidence for learned specialization.

⭕️ [Pending] Physical-energy validation requires a suitable measurement setup; CPU time and state counts do not establish an energy advantage.

🔴 [Later] The full anonymous-factor proposal translates the factorized predictive principle in [JEPA-Anything](https://arxiv.org/abs/2609.20800) into a SARA-compatible research hypothesis. It is not a production change, a claim that SARA has a world model, or evidence for a spike-specific advantage. The paper reports orthogonal predictive factorization across heterogeneous domains, intervention prediction, out-of-distribution evaluation, and long-horizon dynamics; its abstract also makes clear that the reference training uses a conventional predictive architecture, so SARA must not import its gradient-based optimization as a runtime requirement. [Paper source](https://arxiv.org/abs/2609.20800)

## Memory Mosaics contribution and boundaries

Useful additions from [Memory Mosaics, Sections 3–6](https://arxiv.org/html/2405.06394v3#S3):

- Pair a recent-context key with an observed future outcome. Query first, then admit the outcome only when it arrives; never expose the pending value to its own prediction.
- Let separate memories predict reusable components, then compose predictions. Evaluate unseen **combinations** separately from known component transitions.
- Distinguish adding contextual experiences from learning key extractors, factor selection and output composition. Memorization alone does not establish those learning capabilities.
- Measure adaptation against observation count. The three-moons example motivates a small periodic-event diagnostic.

The paper uses gradient-trained extractors and persistent memories, dense vector operations, and quadratic sequence retrieval. Its persistent memory is fixed during inference. SARA does not adopt these as runtime requirements, or interpret the paper as proof of gradient-free factor discovery or low physical energy. SARA's durable evidence/revision store remains distinct from the paper's persistent parameters.

## Executed component study — 2026-10-10

The [frozen protocol](../data/processed/benchmark_fixtures/sparse_predictive_memory_v1.json) was written before candidate implementation and execution. SHA-256: `767a13e47a906844b03b70b5ab5bf221ab46c54af8d844d6cc2a274e8607532f`. This exploratory synthetic study does not reopen sealed datasets or authorize promotion.

The generator emits three integer event IDs per tick from independent cycles of lengths 5, 7, and 11. Each seed changes code assignment and initial phases. The predictor sees events, **not periods or phase variables**, but input slots and output correspondence explicitly supply factor identity. Five seeds check relabeling/phase robustness within one task family, not five independent tasks.

All arms predict before the next tick, learn from 44 chronological transitions, then freeze memory for 385 continuation predictions. Of those contexts, 341 are absent from training and 44 recur. Report overall and novel-context accuracy separately. A missing factor abstains, counting as incorrect. The evaluator's training-context set is unavailable to the predictor.

| Arm | Representation | Maximum retention |
| --- | --- | --- |
| `joint` | One three-event key → three-event outcome | 22 records × 6 scalar slots |
| `provided_factors` | Three supplied one-event keys → corresponding one-event outcomes | 22 records per factor × 2 slots |
| `redundant_factors` | Each memory receives the entire key and predicts one output | 11 records per factor × 4 slots |

All arms have the same **132 scalar-slot ceiling** and **three outcome scalar writes per training tick**, with bounded FIFO eviction. Occupancy and interpreter overhead differ. Key insertion and lookup work differ: this is not equal wall-clock cost or equal total memory bytes. Redundant factors are a whole-context control, **not** learned routing without inhibition. Neither anonymous routing nor overlap suppression is implemented in this study.

| Measurement (all five seeds) | Joint | Provided factors | Redundant factors |
| --- | ---: | ---: | ---: |
| Frozen exact accuracy | 5.714% | 100% | 2.857% |
| Novel-context exact accuracy | 0% | 100% | 0% |
| Retained key/value scalar slots | 132 | 46 | 132 |
| Peak measured predictor state, bytes | 4,303 | 5,170 | 5,967 |
| Training outcome scalar writes | 132 | 132 | 132 |
| Table lookups per prediction | 1 | 3 | 3 |
| First correct prequential prediction | None in 44 steps | Observation 12 | None in 44 steps |

Removing the middle factor reduces exact accuracy to 0%, leaving the other components correct (66.667% component accuracy). Consistent factor permutation preserves predictions after inverse permutation. Shuffled training outcomes reduce exact accuracy to 0–1.039%; copying the current state gives 0%. Frozen replay is exact. All eight registered gates pass for each seed.

Only bounded dictionaries, integer events and local outcome writes are used; no dense matrix library, optimizer, global gradient or GPU. This is a **sparse-event memory component**, not a spiking-neuron implementation. Tests also check an independent scalar lookup reference. The result supports composition when factorization is supplied, not learned specialization, causal discovery, interference robustness, SNN superiority or real-world transfer. Fewer logical scalar slots do not imply fewer Python bytes: the factor arm uses more bytes than the joint arm.

The [report](../workspace/evaluation/sparse_predictive_memory_v1.json) includes all seeds, prequential curves, CPU times, counters, prediction hashes, protocol/source hashes, environment metadata and false promotion/energy/spike-advantage flags. CPU times include evaluation hashing and training state instrumentation; they are diagnostic only. State is sampled before/after updates; transient allocation peaks and process RSS are not measured. No final model artifact is produced.

Implementation: [evaluator](../src/sara_engine/evaluation/sparse_predictive_memory.py), [runner](../scripts/eval/sparse_predictive_memory.py), [tests](../tests/test_sparse_predictive_memory.py).

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_sparse_predictive_memory.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/sparse_predictive_memory.py --output workspace/evaluation/sparse_predictive_memory_v1_replay.json
```

Output paths are validated against managed directories. Existing reports are never overwritten; choose a fresh filename for another replay. An executed five-seed replay matched every nontiming field, and source hashes matched the current implementation. The focused nine tests plus existing predictive-coding regression tests passed (41 total). CPU times may differ.

A compact [result receipt](../data/processed/benchmark_fixtures/sparse_predictive_memory_v1_result_receipt.json) retains report/source hashes and per-seed metrics outside the ignored workspace; the full reports remain in managed workspace storage.

日本語: 因子の対応を与えた疎イベント記憶では、未知の組み合わせ341件すべてを5 seedで正しく予測しました。匿名因子の発見・重複抑制・SNN固有の優位性は未検証です。次は別プロトコルで局所的な分担学習を評価し、本番への昇格は行いません。

简体中文：在预先提供因子对应关系的稀疏事件记忆实验中，五个 seed 均正确预测了全部341个未见组合。匿名因子发现、重叠抑制和脉冲特有优势尚未验证。下一步使用独立协议研究局部分工学习，不升级到生产运行。

## Bounded learned-route study — 2026-10-10

**Decision: retain a negative input-overlap result.** All ten seed/task cases pass execution integrity, but none passes the full registered specialization gate. Output-only inhibition and full inhibition produce identical prediction trace hashes in all 40 case/split comparisons. Thus the added input-overlap penalty has no demonstrated predictive benefit here.

The [protocol](../data/processed/benchmark_fixtures/sparse_factor_routing_v1.json) was fixed before implementation and registered execution: SHA-256 `7de44a4e078bdb6fe370df3a9839ca32d1464bfd066e643201bcdd025d4ed78f`. Coefficients, seeds, ceilings and gates remain unchanged after evaluation. The stop rule retains all negative gates; no real datasets or previous sealed tests were accessed.

### What is learned, and what remains supplied

The predictor receives five binary input channels and, one tick later, three labeled binary outcome slots. The generator privately permutes channels and output slots per seed. It never supplies the true input/output dependencies to learned arms. Forty-five fixed candidate routes cover every one/two-channel key and each output slot. Each route updates a bounded conditional count table and its own prequential squared-error estimate. A three-packet selector ranks those estimates plus a fixed key-width cost. There are no gradients, matrix operations, GPU dependencies or semantic factor names.

This is supervised selection from a hand-defined finite route vocabulary. Input channels, output addresses, factor count, candidate widths and the synthesis rule are supplied. It is not unconstrained factor discovery, learned representation extraction, anonymous concept emergence, recurrent world modeling or a spiking-neuron result. A shared candidate bank is updated locally; individual heads do not grow or learn separate encoders.

### Task and controls

Each case trains on 512 delayed input/outcome examples. Private latent bits are `(a,b,c)`; outputs are either `(a,b,c)` or `(a,b XOR c,c)`. Two nuisance channels copy `a` and `b` with probability 0.9 during training. State `111` is excluded from training. Five fresh seeds randomize channel correspondence and examples in each task family; they are not independent real-world replications.

Frozen evaluation uses 256 draws per split: fresh in-distribution examples, reversed nuisance correlation, excluded `111` with varied nuisances, and paired single-latent-bit interventions with fixed nuisances. The novel split has **four distinct input packets**, not 256 unique compositions. Samples are toy one-step supervised transitions, not a continuous physical trajectory; no multi-step rollout claim follows.

| Arm | Selection and evidence |
| --- | --- |
| `joint_single` | Whole five-channel key, three scalar outcome tables; no decomposition |
| `learned_no_inhibition` | Three distinct candidates with lowest local error plus key-width cost |
| `learned_output_inhibition` | Also penalizes multiple selected routes addressing the same output |
| `learned_full_inhibition` | Also penalizes shared input channels among selected routes |
| `provided_routes` | Generator supplies the true dependencies, explicitly diagnostic only |

Learned arms share exactly the same candidate tables, prequential errors, observations and update schedule. Only prediction-time selection differs, permitting a direct ablation of the input penalty. Distinct candidate selection avoids a trivial baseline that repeats the same route three times. Missing outputs or conflicting component predictions abstain and count as incorrect. Brier scoring uses probability 0.5 for abstentions. Duplicate agreeing packets are averaged; packet order does not matter.

### Results

Exact accuracy below averages five seeds; 100% and 0% entries also hold per seed.

| Task / arm | In distribution | Nuisance shift | Novel `111` | Intervention pairs |
| --- | ---: | ---: | ---: | ---: |
| Both tasks: joint | 99.453% | 61.953% | 0% | 83.281% |
| Independent: learned without inhibition | 100% | 100% | 100% | 100% |
| Interacting: learned without inhibition | 0% | 0% | 0% | 0% |
| Both tasks: output-only inhibition | 100% | 100% | 100% | 100% |
| Both tasks: full inhibition | 100% | 100% | 100% | 100% |
| Both tasks: provided dependencies | 100% | 100% | 100% | 100% |

In the interacting task, uninhibited selection fails to cover one output, so exact accuracy is zero through abstention; it does not lose all component information. Full/output-only inhibition both reach mean nuisance-shift Brier 0.00003132 versus 0.08335210 without inhibition. In the independent task all learned arms already succeed (mean shift Brier 0.00001527).

Full inhibition leaves input overlap at zero for independent outputs and one third for interacting outputs. Shared input is **necessary** for the XOR output and the corresponding single-bit output. Removing a selected factor while freezing selection reduces exact shift accuracy to zero. Untrained models abstain entirely; shuffled outcomes give 11.719–28.125% shift exact accuracy. Scalar readout equivalence, packet permutation, frozen state and cloned replay pass.

The registered two-percentage-point gain over output-only inhibition fails in every case: measured gain is zero. Overlap reduction also fails in nine cases; gain over no inhibition fails in all five independent cases, where no inhibition was needed. Passing accuracy gates therefore cannot be reported as passing specialization. Output coverage is the demonstrated contribution in the interacting toy task; input-overlap suppression is not established as useful.

### Resource and evidence boundaries

All arms use common ceilings of 4,096 logical scalar slots, 131,072 measured predictor bytes and 45 local route updates per tick. Binary key cardinality bounds each candidate to four records and each joint table to 32. Actual use is different: learned arms perform 23,040 route updates over training, joint/provided arms 1,536. Thus the comparison has equal **ceilings**, not equal consumed compute. All three learned ablation arms consume the same update count. Each update changes a count, a local error estimate and an observation counter; it is not one scalar write.

Maximum observed state was 842 logical slots and 41,846 Python bytes. Counts are bounded by this fixed-length study; the module is not an unlimited-duration runtime service. Pre/post-update state sampling excludes temporary allocation peaks, generator/audit data and process RSS. Candidate visits, active packet counts, output-usage entropy, overlap, Brier, coverage, training curves and CPU time including instrumentation are reported. No physical energy was measured and no model was promoted or saved as a production checkpoint.

Artifacts: [full report](../workspace/evaluation/sparse_factor_routing_v1.json), [compact result receipt](../data/processed/benchmark_fixtures/sparse_factor_routing_v1_result_receipt.json), [evaluator](../src/sara_engine/evaluation/sparse_factor_routing.py), [runner](../scripts/eval/sparse_factor_routing.py), [tests](../tests/test_sparse_factor_routing.py).

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_sparse_factor_routing.py tests/test_sparse_predictive_memory.py tests/test_predictive_coding.py tests/test_phase5_predictive_coding.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/sparse_factor_routing.py --output workspace/evaluation/sparse_factor_routing_v1_new_replay.json
```

Reports use exclusive creation under validated managed paths. A failed scientific gate is preserved in JSON; the command succeeds when execution integrity passes. Replays verify the fixed protocol, not a new confirmatory trial, and must not be used for tuning. The next independent protocol should compare output coverage with error-contribution-based candidate pruning under equal consumed update budgets, preserving genuinely useful shared inputs.

Validation completed: 18 new development tests plus 41 previous/regression tests pass (59 total). A fresh-process replay of all ten cases matches every nontiming field; current source hashes match both reports. The compact receipt preserves these checks and both report hashes. Tests cover causal receipt order, rejected-update atomicity, exhaustive binary-key capacity, hidden-route isolation, equal learned-arm evidence, generator split/intervention semantics, separate input/output inhibition, scalar residual updates, missing/conflicting packets, packet permutation, frozen evaluation and factor reset.

日本語: 2課題×5 seedで、入力・出力の対応を与えずに有限候補から経路を選ぶ実験を完了しました。出力の重複回避は相互作用課題で有効でしたが、入力重複の抑制による追加効果はゼロで、事前の採用基準は不合格です。相互作用に必要な共有入力を残し、次は更新回数を揃えた候補選択を別プロトコルで検証します。本番統合やSNN優位性は主張しません。

简体中文：已完成两个任务、五个 seed 的有限候选路径学习实验，学习条件不接收真实输入输出对应关系。避免输出重复有助于交互任务，但额外抑制输入重叠没有预测收益，未通过预先规定的采用门槛。保留负面结果和必要的共享输入，后续在独立协议下比较实际更新预算相同的候选选择；不升级到生产，也不声称脉冲优势。

## Actual-update-matched pruning study — 2026-10-10

**Decision: the bounded component gate passes; production promotion remains false.** All ten seed/task cases pass the per-case gates and both aggregate gates pass. The earlier negative input-overlap result remains unchanged.

The [protocol](../data/processed/benchmark_fixtures/sparse_route_budget_v1.json) was fixed before implementation and execution (SHA-256 `c8c99b539917e5e1f3847d5ab3e4f419ea80529d0be05fd4aa7d8cb0086eb7af`). The stationary Boolean generator is reused with five fresh seeds, independent/interacting tasks, 512 training examples and four frozen 256-draw evaluation splits. The excluded `111` split still contains four distinct input packets, not 256 unique compositions. Source hashes pin the reused generator, readout and evaluation implementation as well as the new module/runner.

### Equal actual updates and selection rules

All four arms use the same 45 candidate routes, conditional count memory, scalar local residual update, one-prediction-per-output readout and seed-shuffled initial schedule. Exactly three distinct routes per output are updated on every tick: **nine updates × 512 ticks = 4,608 updates per arm**. Each route update assigns its observed count, local error and observation count, totaling 13,824 such assignments. Table initialization, scheduling, metadata writes and candidate inspection are separate operations; this is not equality of all CPU operations.

The first 180 ticks are identical round-robin learning, giving every candidate 36 observations. Before predicting example 181, pruning arms select three candidates per output using only already observed training evidence. They freeze this nine-route active pool and update all nine on every remaining tick. The round-robin arm continues to distribute the same nine updates per tick over all 45 candidates. A per-tick ledger verifies the exact changed observation counters, and final table counts independently sum to 4,608. Tests also verify every local error update and that inactive tables/errors/counters stay unchanged.

| Arm | Rule after common warmup |
| --- | --- |
| `round_robin` | Keep all 45 routes eligible; continue the uniform nine-route update schedule |
| `random_prune` | Keep a seeded random three routes per output, without consulting outcomes |
| `residual_prune` | Keep the three lowest local prequential squared-error-plus-width-cost routes per output; never penalize shared input |
| `overlap_prune` | Greedily keep three routes per output, adding a penalty for sharing inputs with routes already selected |

All learners receive input events and delayed labeled outcomes, never the true dependencies. The only use of generator truth is in evaluator-only retention checks and a shared-input masking ablation. Retired count tables remain frozen for audit and are excluded from normal prediction and updates. Active candidate reduction therefore does **not** mean physical memory reclamation. The shortlist does not reactivate after a distribution change; such recovery remains untested.

### Results and interpretation

Exact accuracies are five-seed means. Round-robin and residual pruning attain 100% in every individual case and split.

| Task / arm | In distribution | Nuisance shift | Novel `111` | Intervention pairs |
| --- | ---: | ---: | ---: | ---: |
| Both tasks: round-robin | 100% | 100% | 100% | 100% |
| Both tasks: residual pruning | 100% | 100% | 100% | 100% |
| Independent: random pruning | 88.672% | 0% | 45.156% | 46.719% |
| Independent: overlap pruning | 100% | 100% | 100% | 100% |
| Interacting: random pruning | 67.813% | 11.563% | 20.625% | 44.453% |
| Interacting: overlap pruning | 72.109% | 71.016% | 50.391% | 72.109% |

Across all ten cases, mean shifted Brier decreases from **0.000564836** with round-robin to **0.000045069** with residual pruning (92.02% reduction). The registered minimum was 20%. Mean shifted exact accuracy exceeds random pruning by 94.22 percentage points (minimum 10). All accuracy, Brier noninferiority, shortlist reduction, outcome-shuffle and shared-dependency gates also pass.

This is a probability-error improvement, **not an accuracy improvement over round-robin**. On deterministic toy targets, concentrating the same updates on useful routes supplies more observations per retained key and reduces the influence of the fixed smoothing prior. The result does not establish calibrated confidence on noisy real data, better concept discovery, or a spike-specific mechanism.

Residual pruning retains the true shared dependency routes in all five interacting cases. Masking the shared input to zero at evaluation, while leaving the target unchanged, reduces intervention exact accuracy from 100% to 47.27–57.03%. This is an input-occlusion ablation; the separate paired-intervention split recomputes outcomes under latent interventions. Shuffled training outcomes reduce shifted exact accuracy to 0–17.19%. The overlap-penalized control's loss supports retaining useful shared inputs rather than treating every overlap as redundant.

### Resources, validation and limits

Maximum measured predictor state is 40,565 Python bytes and 919 conservative logical slots, below the common 131,072-byte/4,096-slot ceilings. All arms execute exactly 4,608 actual outcome updates and 13,824 count/error/observation assignments, including shuffled-outcome controls. The active pool is 45 for round-robin and nine for each pruning arm.

Readout candidate score visits over training are 23,040 for round-robin and 11,088 for pruning arms; residual pruning adds 45 one-time pruning score visits. Frozen prediction inspects nine active candidates instead of 45. Scheduler work, key lookups, table initialization and pruning costs are not identical. Measured CPU time includes instrumentation and is diagnostic only; no total-CPU or physical-energy advantage is claimed. Retained state sampling excludes transient allocation peaks, evaluator data and process RSS. Counts are bounded by this fixed-length study, not an unlimited-duration runtime contract.

Artifacts: [full report](../workspace/evaluation/sparse_route_budget_v1.json), [result receipt](../data/processed/benchmark_fixtures/sparse_route_budget_v1_result_receipt.json), [evaluator](../src/sara_engine/evaluation/sparse_route_budget.py), [runner](../scripts/eval/sparse_route_budget.py), [tests](../tests/test_sparse_route_budget.py).

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_sparse_route_budget.py tests/test_sparse_factor_routing.py tests/test_sparse_predictive_memory.py tests/test_predictive_coding.py tests/test_phase5_predictive_coding.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/eval/sparse_route_budget.py --output workspace/evaluation/sparse_route_budget_v1_new_replay.json
```

Reports are written exclusively to validated managed paths, preserving old evidence. This isolated experiment imports no GPU or dense-matrix learning dependencies, reads no real datasets, and saves no production model. Future work must independently preregister dependency changes and bounded reactivation rather than silently revising this successful stationary protocol.

Validation: ten new tests and 59 prior/regression tests pass (69 total). Fresh-process replay matches every nontiming field across all ten cases. Current source hashes match both reports, summary gates recompute exactly, all 40 comparison arms verify 4,608 actual updates, and all five interacting cases retain their shared dependencies. The compact receipt retains source/report hashes and per-case evidence outside ignored workspace storage.

日本語: 全条件の実更新を4,608回に揃えた比較で、誤差に基づく削減は候補を45から9へ絞り、共有依存と正解率100%を維持しました。巡回更新も正解率100%ですが、確率予測の二乗誤差は約92%減りました。入力の重複を罰する削減は相互作用課題を損ないました。69件のテストを通過した限定的な部品実験であり、本番採用・実電力削減・SNN固有の優位性は主張しません。

简体中文：在所有条件实际执行4,608次路径更新的比较中，基于局部误差的裁剪将候选从45个减至9个，同时保留必要的共享依赖和100%的准确率。轮询更新也达到100%准确率，但裁剪的概率平方误差降低约92%。惩罚输入重叠会损害交互任务。69项测试通过；这仅是受限组件结果，不代表生产采用、实际节能或脉冲特有优势。

## Adopted principle

This plan also adopts ideas from [Memory Mosaics](https://arxiv.org/abs/2405.06394), in addition to the JEPA-Anything motivation above. The executed component studies above are separate from the unfinished full anonymous-factor hypothesis.

Instead of learning one undifferentiated predictive state, SARA may maintain a bounded set of local predictive factors:

```text
shared sparse event state
        ├── factor route A + local predictor
        ├── factor route B + local predictor
        ├── factor route C + local predictor
        └── bounded synthesis / abstention
```

The factors are not assigned human meanings in advance. A factor is useful only when it repeatedly reduces a future-state prediction residual, survives a held-out intervention or nuisance shift, and remains within event and state budgets. A branch that merely fires often, or receives a semantic label after the fact, is not a discovered concept.

## SARA translation

The paper's orthogonality and variance objectives become sparse event signals rather than matrix losses:

| Predictive-factor idea | SARA-compatible signal |
| --- | --- |
| factor prediction error | factor-local `prediction_error` event bound to recent eligibility |
| cross-factor overlap penalty | bounded pairwise active-route overlap counter and lateral inhibition event |
| dead-factor/activity regularization | local underactivity / starvation event with homeostatic threshold |
| latent recombination | bounded factor packet synthesis with explicit missing-factor abstention |
| intervention prediction | source-addressed counterfactual outcome event and held-out intervention split |
| dynamic factor count | later structural growth/pruning proposal, default-off |

No dense projection matrix, global covariance, whole-network gradient, or GPU is required. Pairwise overlap must be computed only for the capped active-factor set in the current event window. The implementation must expose overlap, prediction residual, factor count, event count, retained state, and abstention decisions for audit.

## Remaining anonymous-factor study before integration

Neither the supplied-factor study nor the negative input-overlap study above satisfies this integration gate. A future task must distinguish learned factor specialization from a larger monolithic control under matched consumed update budgets. Freeze the generator, private target factors, nuisance factors, intervention actions, and causal split before candidate implementation. Compare:

1. scalar / single-factor local predictor;
2. multi-factor routing without overlap suppression;
3. multi-factor routing with bounded overlap inhibition;
4. (later) overlap inhibition plus structural growth/pruning.

All arms receive the same observations, local update budget, factor capacity, and delayed outcome. The fourth arm must not be scored until the first three establish that factor separation helps independently. The first study must not use BPI or Sepsis frozen checkpoints for tuning and must not reopen CartPole held-out ranges.

## Required measurements and gates

Record prediction accuracy and Brier or calibrated residual where applicable, but also:

- held-out intervention and nuisance-shift accuracy;
- long-horizon rollout error only when the task supplies a bounded future stream;
- active-factor count and per-factor usage entropy;
- pairwise route-overlap rate before and after suppression;
- events per prediction, peak retained state, and local update count;
- exact replay and scalar-equivalence traces;
- factor-reset and factor-permutation controls;
- missing-factor abstention and malformed-packet rejection.

Promotion requires a preregistered improvement over the single-factor control on at least one held-out causal metric without a material regression in the resource envelope, and a targeted overlap ablation must remove the claimed specialization effect. Accuracy gain alone is insufficient. A positive result would support sparse predictive factorization, not a general JEPA or backpropagation replacement.

## Explicit non-adoptions

- Do not port the paper's projector, EMA target encoder, orthogonality matrix loss, or gradient optimizer into SARA's normal runtime.
- Do not name factors `speed`, `position`, `goal`, or `cause` before a factor survives anonymous held-out validation; names may be attached afterward as metadata only.
- Do not equate low overlap with truth. Provenance, contradiction handling, verification, and revision remain separate.
- Do not add dynamic factor growth until fixed-factor overlap suppression has a positive, reproducible causal result.

日本語: JEPA-Anythingから採用するのは勾配学習ではなく、予測状態を複数の局所因子へ分解し、重複を抑えながら予測・再合成するという仮説です。因子の意味は事前に命名せず、予測残差・介入汎化・資源上限・再現性で検証します。まず固定因子数の小規模課題を行い、構造成長は後段に分離します。

简体中文：从 JEPA-Anything 采用的不是梯度训练，而是把预测状态分解为多个局部因子、抑制冗余并重新合成的研究假设。因子不得预先命名，必须通过预测残差、干预泛化、资源上限和精确重放验证。先研究固定因子数的小任务，再单独研究结构增长。

## Repeated-state real-source prerequisite — 2026-10-11

🟢 [Done] [Two fresh source screens](R2_ROOM_TEMPORAL_PREFLIGHT.md) retain failures: incomplete room-motion inputs and absent occupied development examples in a PIR-only occupancy task. Separately registered training-only diagnosis shows that shared current inputs plus sparse recency branches can identify four opposite-posterior groups, with exactly 10,400 changed local writes for each of six readouts and exact scalar equivalence. Outcomes are delivered through a chronological receipt queue, including overlapping prediction horizons. This establishes local branch identification only; it does not justify factor inhibition, structural growth, generalization or SNN superiority.

🟠 [Next] Require complete recurring observed state and both-class chronological development support in a longer independent source before predictive-factor comparisons. Preserve all prior negative thresholds and seals; current input availability must be specified prospectively rather than dropping sensors after scoring.

日本語: 局所的な履歴の識別は訓練データで確認しましたが、比較条件は不成立です。因子分解の優位性は未確認で、次は開発期間の両クラスが成立する独立データ源を優先します。

简体中文：训练数据证实了局部历史可区分，但开发比较条件未成立。因子分解优势尚未验证，下一步优先寻找开发期包含两类结果的独立完整数据源。

## Longitudinal notification prerequisite — 2026-10-12

🟢 [Done] [Fresh CASAS HH102](R2_CASAS_LONGITUDINAL_COMPARISON.md) resolves both-class development support and raw current-state recurrence, but not identifiable rich-current temporal contrasts (0/10). A separately preregistered post-support diagnostic matches eight arms at 18,192 changed local writes each, consumes only due outcomes, freezes development and replays checkpoints exactly. Joint improves daily Brier by only 0.000726; all four quality/mechanism thresholds fail and scalar equivalence remains exact. This supports the bounded delayed-readout implementation, not temporal necessity, learned factor separation or spike-specific usefulness.

🟠 [Next] Require a genuinely longer unconsumed observed stream (metadata screen: >=180 days) and sufficient current/clock/history branch support before new factor comparisons. Keep failed source and performance gates, frozen periods, shared current inputs and adequate non-spiking controls. Do not resume inhibition, growth or production integration from these scores.

日本語: 開発期間の両クラス不足は解消しましたが、強い現在状態対照下の履歴対照群不足と性能閾値未達は残ります。条件を緩めず、未消費のさらに長期な独立ストリームを優先します。

简体中文：开发期两类支持已成立，但强当前状态条件下的历史对照不足，性能门槛仍未通过。保持现有条件，优先寻找未使用的更长独立数据流。

## Qualified long-household prerequisite — 2026-10-12

🟢 [Done] [HH103 long-source comparison](R2_CASAS_180_DAY_COMPARISON.md) passes source qualification: 2,071 recorded days, 73 same-current history contrasts and both classes across 60 development days. Eight arms consume exactly 56,028 real local writes each with due-ordered feedback and exact checkpoint replay. Joint loses to strong current/history controls and improves when gap information is fixed; retain this learner negative. Correct outcome pairing matters, but scalar traces remain identical and predictive factor usefulness is not established.

🟠 [Next] Preregister training-only gap-branch support and bounded confidence/backoff work before fresh independent validation. A post-score training-state audit finds 5,660 of 8,346 joint detail routes are singletons; this motivates an auditable support hypothesis, not tuning on development scores. Preserve current features, non-spiking references, actual update matching, all frozen quality conditions and sealed tails. Anonymous growth, inhibition and runtime integration remain deferred.

日本語: 長期独立データで比較の成立条件は満たしましたが、時間分岐の有用性は未確認です。次は訓練限定で局所枝の支持数と信頼度付き読み出しを事前登録し、新しい独立評価で確認します。

简体中文：长时间独立数据满足比较成立条件，但时间分支的实用性未通过。下一步预注册仅训练期的分支支持与置信度读取研究，再用新的独立评估验证。

## Fixed support confidence prerequisite — 2026-10-13

🟢 [Done] [Training-only support-8 diagnostic](R2_CASAS_CONFIDENCE_DIAGNOSTIC.md) preserves current inputs, due feedback and three changed writes per outcome. Original eight arms plus two sufficient-reference supplements each perform 27,972 writes; exact replay/scalar equivalence pass. Support gating improves ungated joint slightly but fails all performance/gap gates versus the complete references. Preserve this negative and the original development/test seals; the internally reused confirmation is not unseen generalization.

🟢 [Done] This prerequisite was executed on fresh HH104; see the [current-relative evidence result](R2_CASAS_EVIDENCE_DIAGNOSTIC.md). Eleven preregistered arms each consume 82,062 changed writes with frozen confirmation and exact replay. All source gates pass, but evidence activates 0/15,552 times and exactly matches current. Quality/gap gates fail; preserve the negative and 3,845,589 sealed records.

🟢 [Done] [Training-only bounded pooling](R2_CASAS_POOLING_FEASIBILITY.md) executes 52,659 actual changed writes per 13 arms with exact replay and 113 passing related tests. The fixed map preserves current context and distinguishes supplied opposite-future witnesses, reducing routes 15,322→9,219. Real opposing-evidence contexts and confirmation activations are both zero; feasibility/quality/gap gates fail. Preserve the negative and 4,249,759 sealed records.

🟢 [Done] [Count-only support/effect audit](R2_CASAS_COUNT_SUPPORT_AUDIT.md) separates absent complements, sample-size impossibility, fixed-margin impossibility and insufficient observed contrast. Pooled branches split 2,140 / 3,200 / 99 / 112, with zero passes; both input models retain identical current counts and 52,659 historical writes, zero new writes and exact replay. All 121 related tests pass. Weak observed separation is not proof of a small population effect; prior negatives and seals remain unchanged.

🟢 [Done] [Count-only observation-budget forecast](R2_CASAS_OBSERVATION_BUDGET.md) fixes rates, current conditions and evidence thresholds. Closed-world targets require +52,659 pooled / +70,212 joint observations within acquisition/write ceilings; conservative state growth exceeds the 100,000-route cap. Oracle bounds are branchwise, not a global plan. No actual acquisition or learning; exact replay, 128 related tests and preserved negatives/seals.

🟢 [Done] [Count-only vocabulary state ceilings](R2_CASAS_VOCABULARY_STATE_BUDGET.md) recover 20 retained event types, including a history-only type. Their full known-event envelope is 67,201 pooled routes, versus an uncertified 86,029,441 joint envelope. Pool certifies <=29 total types without rejecting unknown events; unrestricted growth remains unresolved. Zero acquisition/learning, exact replay and 135 related tests pass. Prior negative results, exact current/evidence conditions and all seals remain unchanged.

🟢 [Done] [Fresh HH105 input preflight](R2_CASAS_ARRIVAL_PREFLIGHT.md) measures the complete registered training prefix of 180 recorded dates without unknown-event filtering. Novelty/recurrence and observed state budgets pass; arrival 29,552 fails fixed 70,212 target. Both arms have 0 actual learner writes and 88,656 hypothetical updates, exact process/growth/inventory replay and 145 passing related tests. Preserve all negatives and 5,731,389 sealed independent rows; original class/quality conditions remain unassessed and unchanged.

🟢 [Done] [Timestamp-only safe upper bound](R2_CASAS_TIMESTAMP_UPPER_BOUND.md) retains all original timing/current/evidence conditions. Minute-bucket certification covers every actual binary-subsequence arrival; 8,192 mask checks and all three existing fit windows pass. Their 25,334/22,411/37,748 upper bounds exclude the current 70,212 planning target. Zero event-value inspection/learning, peak 6 buckets, exact replay and 157 related tests; old negatives and seals preserved.

🟢 [Done] [Finite unused-household timestamp screen](R2_CASAS_FINITE_ROSTER_SCREEN.md) preregisters HH106→HH107→HH108, code and aggregate source/timestamp budgets before acquisition. Complete 180-date training bounds 37,857/59,892/24,569 all miss the fixed 70,212 target. Stop this roster with no extension, window change or event-value inspection. Acquisition 101,722,176 body bytes and 2,253,772 training timestamps remain within budgets; exact replay and 177 related tests pass. All 326 earlier artifacts/negatives, sensor/message fields and independent prediction information remain preserved; zero learning writes. This is a source-arrival exclusion, not a temporal-readout quality result.

🟢 [Done] [Public metadata acquisition feasibility](R2_PUBLIC_SOURCE_METADATA_FEASIBILITY.md) preregisters four non-CASAS source families and aggregate review budgets before searches. TIHM duration fails; IDEAL's documented high-rate stream/full bundle is incompatible/over budget; SPHERE/ECO usable PIR coverage, density and access remain unresolved. No payload acquisition is justified. The original 180-fit split requires 270 source dates; conditional density arithmetic is verified against the unchanged extractor, with 215 passing related tests and exact replay. All 338 previous artifacts/negatives, thresholds/current conditions and seals remain preserved; no payload, preview or learning.

🟢 [Done] [Finite SPHERE/ECO metadata contract audit](R2_SPHERE_ECO_METADATA_AUDIT.md) preregisters candidates, code, strict Kitchen ON semantics and total limits before new external inspection. Eight searches/13 pages retain six ECO metadata entries but no payload/preview/archive range/learning. Documented PIR packets and entrance/occupancy/electrical streams are incompatible with the fixed target. Label-day support is not PIR coverage/density; package/terms gaps remain unknown. Exact replay and 241 related tests preserve 345 earlier artifacts and code pins, old negatives and all seals. No SNN-specific advantage is inferred from metadata.

## Retained temporal evidence and SNN work-accounting decision — 2026-10-22

🟢 [Done] [Read-only R2 decision audit](R2_RESEARCH_DECISION_AUDIT.md) registers 19 existing aggregate receipts and a 1 MiB parsing budget before execution. Actual reads total 216,721 bytes; changed-write parity includes all ten confidence arms, with zero reported frozen-period updates. Controlled local temporal identification is retained, while sufficient history/scalar controls and all prior negative quality gates block SNN predictive promotion. Eleven separate work dimensions expose missing observer/receipt/expiry/deployment costs; instrumented CPU, route counts and queue peaks are not complete resource evidence. Zero new learner scores/writes/raw or model payload reads, exact replay, 333 related tests and 353 prior artifact pins plus registered code preserved.

🟢 [Done] [Input-only event-work accounting](R2_EVENT_WORK_ACCOUNTING.md) selects the contract before implementation and verifies all 48 fixed cases. The shared observer runs once physically and is charged fully to each arm. Existing arrival-indexed neurons and an independent event-driven scalar recurrence have identical state/receipt traces: 224 backend steps and timer pops per arm, including 112 expiries and 112 stale timers. Extending idle to one billion units adds no operation work. Learning/scoring remains zero; 356 tests and exact fresh-process replay preserve 358 prior artifacts and code pins. The sampled explicit-neuron graph is larger; CPU, energy and transient peaks are unknown, so no SNN advantage is established.

🟢 [Done] [Input-only instrumentation separation](R2_EVENT_WORK_SEPARATION.md): selection and code registered before finite execution; all 48 cases match frozen-reference state at 1,184 operation and 48 teardown boundaries, prior receipt hashes and receipt-only execution. Shared observation remains fully charged to both arms. Returned buffers/control objects are included in external graph samples; intra-call byte peaks remain unknown. 381 related tests and exact fresh-process replay preserve 363 prior artifacts/code pins. Zero learning/scoring; no SNN advantage or seal opening.

🟢 [Done] [Isolated Python allocation diagnostic](R2_EVENT_ALLOCATION_DIAGNOSTIC.md): contract/code registered before 48 cases in 144 fresh workers. Implementation and scope tests pass, but the final report exceeds the registered 8 MiB limit; overall audit fails, no raw measurements/result/replay are retained. Preserve the consumed execution and failure receipt without increasing the limit or rerunning. 411 tests pass with one expected pre-execution skip; 368 prior artifacts/code pins, conditions, negatives and seals remain intact. No quantitative allocation or SNN advantage claim.

🟢 [Done] [Bounded report-representation preflight](R2_ALLOCATION_REPORT_PREFLIGHT.md): selected before implementation; every accepted report is bounded by 7,302,888 bytes within fixed 8 MiB, with explicit site/incidence/string capacities and overflow refusal. Lossless unknown-site/ownership round-trip, saved-result-before-launch gate, 41 tests and exact fresh-process static replay pass. Preserve 372 prior artifacts/code pins, v1 failure and all conditions/seals. Zero new allocation measurements/workers; future live-site capacity and SNN advantage remain unestablished.

🟢 [Done] [Retained allocation diagnostic v2](R2_EVENT_ALLOCATION_V2.md): fresh selection/code registered before the same 48 cases and 144 workers per phase. Primary and independent replication each retain 608,412 bytes; all 3,840 windows match raw allocations, semantic state and owned graphs. 51 tests pass; 378 earlier artifacts/code pins, v1 failure, conditions and seals preserved. Standalone new-call peaks are neuron 3,048 / scalar 2,280 bytes, so no SNN memory advantage is established; CPU/energy/full process peaks remain unknown.

🟢 [Done] [Finite retained-evidence mechanism/task audit](R2_MECHANISM_DECISION_GATE.md): 26 aggregate receipts / 1,062,066 bytes; original predictive decisions and accounting/replication reconcile. Conditional branch-local delayed eligibility remains untested; no qualified fresh independent task is established. Additional resource work is held. Exact replay, 63 passing tests, 386 prior artifact pins; zero new measurement/scoring/acquisition, unchanged conditions, negatives and seals.

🟢 [Done] [Branch-local delayed-credit static contract](R2_LOCAL_CREDIT_CONTRACT.md): preregistered causal receipt/membership and six-control proposal checks reject no-ops, duplicate updates, counter padding, private history, expiry and shared-observer mismatches. Four symbolic proposed changed fields per arm, zero applied learning writes; 100 tests and exact static replay pass under Python 3.10.22. No learning-rule proof, actual execution parity, independent task or SNN advantage is established. Original conditions, negatives and seals remain unchanged; further experiments/resources stay held.

⭕️ [Pending] Qualify a causal branch-local eligibility effect and a fresh independent task under the unchanged original gates before considering a separately registered resource protocol. No automatic diagnostic v3, source expansion, scoring, condition relaxation or seal opening.

⭕️ [Pending] Independent usefulness, physical energy, source acquisition, structural growth and production promotion retain their original gates. No ANN-exclusive capability is established by sparse scheduling or local learning alone.

日本語: 既存19件の監査と合成入力の計上契約に続き、計測分離を事前登録して検証しました。同じ48ケースで1,184操作境界と48終了境界の状態、既存配送記録、診断なし実行が一致し、381テストと完全再現で363証跡・条件・負の結果・封印を保持しました。一時メモリは返却バッファを計上し、関数内の割当ピークは未知のままです。続く隔離割当診断は固定8 MiBの結果保存上限で不合格となり、生値は未保存です。411テスト成功・1件の実行前スキップと失敗証跡を保持し、続く形式事前検証は上限7,302,888バイト・41テスト・完全再現で成功しました。続くv2も別登録し、各608,412バイトの結果保存と3,840窓の一致、51テストを確認しました。SNNのメモリ優位は示されず、続く26件の有限判断監査も完了し、63テストと完全再現で386証跡を保持しました。枝ごとの遅延適格性は条件付き設計候補ですが、因果効果と新しい独立課題の根拠が不足するため追加資源計測を保留します。SNN固有の優位は未確認です。

简体中文：在既有19份证据审计和合成核算后，预注册并验证测量分离。相同48个案例在1,184个操作边界与48个清理边界的状态、既有回执及无诊断执行结果一致。381项测试与精确重放保留363项证据、原条件、负结果和封存。内存范围包含返回缓冲区，函数内分配峰值仍未知。后续隔离分配诊断因结果超过固定8 MiB上限而失败，原始数值未保存。保留411项测试通过、1项执行前跳过及失败证据；后续格式预检以7,302,888字节上界、41项测试和精确重放通过，后续v2也已单独注册，各保存608,412字节，3,840个窗口一致且51项测试通过。未显示SNN内存优势；后续26项证据的有限判断审计也已完成，63项测试与完全重放通过。分支局部延迟资格迹仅为有条件的设计候选，因果效果和新独立任务证据不足，暂停额外资源测量。SNN独有优势仍未确认。

🟢 [Done] The previously queued bounded read-only R2 decision audit is complete; its retained findings and subsequent accounting/separation checks are recorded above. The allocation diagnostic subsequently failed its result-storage budget; the subsequent report-representation preflight passes; allocation-diagnostic v2 now retains complete replicated results; the retained-evidence mechanism decision is complete and further resource work is held pending causal mechanism and independent-task prerequisites. No new search/acquisition, re-scoring, tuning, current-task change or R3 optimization is authorized by these results.

⭕️ [Pending] New acquisition and independent usefulness/resource evidence require a compatible justified source/task contract and all original input, opposing-history, class/support, state/write-budget and quality gates. The SPHERE/ECO documented-stream exclusions are closed; no automatic retries or CASAS additions. Sparse event scheduling/local updates can be shared with non-spiking implementations, so actual matched-control evidence remains essential.

日本語: 支持数だけのゲートに続き、未使用HH104で現在状態に対する局所予測根拠も事前登録して検証しました。両方の負の結果を保持し、独立期間は開きません。

简体中文：继仅基于支持数的门控之后，已在未使用的HH104上预注册并检验相对于当前状态的局部预测证据。保留两项负结果与全部独立期间封存。

日本語: HH104の固定局所根拠ゲートは0件の有効化にとどまり、現在状態対照を上回りませんでした。続く訓練期間だけの限定集約も事前登録して検証し、負の結果を保持しています。

简体中文：HH104的固定局部证据门控启用0次，未超过当前状态对照。随后已预注册并检验仅训练期的有界时间历史聚合，保留负结果与全部封存。

日本語: 訓練期間だけの4枝集約は状態数を減らしましたが、実データの識別と予測改善は不合格でした。続く保存済み訓練カウントの監査で、固定した根拠条件に対する支持不足・クラス比率の制約・観測差の不足を区別しました。

简体中文：仅训练期的4分支聚合减少了状态量，但真实数据的识别与预测改进未通过。随后仅用保存的训练计数完成审计，区分固定证据条件下的支持不足、类别边际约束和观测差不足。

日本語: 監査は学習も封印期間の読み取りも行わず完了しました。続く追加観測量の事前登録計算では、既知状態の再来なら取得予算内ですが、未知状態の保守的上限では状態予算を保証できませんでした。

简体中文：审计不进行学习或读取封存期间。随后已预注册额外观测预算：已知状态重复时采集预算可行，但未知状态增长的保守上界无法保证状态预算。

日本語: 既知組合せと未知イベントを分けた上限も事前登録して検証しました。既知20種類での集約上限は67,201セルですが、無制限の未知イベントは未確定で、予測改善とは扱いません。

简体中文：已预注册并检验区分已知组合与未知事件的状态上界。已知20类的聚合上界为67,201单元，无限未知事件仍未确定，不据此宣称预测改进。

日本語: 新しい訓練対象HH105の事前確認も完了しました。未知イベントを保持して反復と状態数は成立しましたが、到着29,552件が固定目標に届かず不合格です。独立期間の封印と有用性の条件は維持します。

简体中文：已完成新训练身份HH105的预检，保留未知事件且重复与状态预算满足条件，但29,552次到达未达到固定目标。独立期间封存和效用要求保持不变。

日本語: 日時だけの安全な上限も事前登録して検証しました。既存訓練3期間はいずれも固定目標に届き得ないと判断でき、入力値・ラベルを再検査せず、既存の負の結果と封印を保持しました。

简体中文：已预注册并验证仅时间戳的安全上界，三个既有训练窗口均不可能达到固定目标；不重读事件值或标签，保留既有负结果与封存。

日本語: 続くSPHERE/ECOの有限メタデータ監査も事前登録し、公表ストリームが元のKitchen ON定義に不適合と判断して終了しました。未確定のPIR期間・密度・サイズ・利用条件を残し、本体取得・候補追加・条件緩和・封印解除はしません。さらに既存19証跡の監査と合成入力48ケースの計上検証を完了し、356テストと別プロセス再現で負の結果を保持しました。続く計測分離は48ケース・381テストで成立しましたが、隔離割当診断は固定8 MiBの結果保存上限で不合格です。失敗と封印を保持し、続く形式事前検証は上限7,302,888バイト・41テスト・完全再現で成功しました。続くv2も別登録し、各608,412バイトの結果保存と3,840窓の一致、51テストを確認しました。SNNのメモリ優位は示されず、続く26件の有限判断監査も完了し、63テストと完全再現で386証跡を保持しました。枝ごとの遅延適格性は条件付き設計候補ですが、因果効果と新しい独立課題の根拠が不足するため追加資源計測を保留します。

简体中文：另行预注册并完成SPHERE/ECO有限元数据审计，判定其已描述数据流不符合原Kitchen ON定义并终止。保留PIR期间、密度、体积和使用条件的未知项，不采集本体、不扩充候选、不放宽条件或解除封存。随后完成19份既有证据审计和48个合成输入案例的核算验证，356项测试与新进程重放保留原负结果。后续测量分离通过48个案例和381项测试，但隔离分配诊断因结果超过固定8 MiB上限而失败。保留失败与封存；后续格式预检以7,302,888字节上界、41项测试和精确重放通过，后续v2也已单独注册，各保存608,412字节，3,840个窗口一致且51项测试通过。未显示SNN内存优势；后续26项证据的有限判断审计也已完成，63项测试与完全重放通过。分支局部延迟资格迹仅为有条件的设计候选，因果效果和新独立任务证据不足，暂停额外资源测量。

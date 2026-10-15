# Sparse Predictive Factorization — SARA Research Proposal

## Status

🟢 [Done] A Memory Mosaics-inspired **provided-factor memory component** comparison is implemented and verified on five synthetic seeds (2026-10-10). Results below do not complete anonymous sparse predictive factorization.

🟢 [Done] A separate bounded learned-route study now covers independent/interacting Boolean transitions and nuisance shifts across ten seed/task cases. Correct routes can be selected without supplied input/output correspondence, but the registered input-overlap specialization gate fails. Preserve this negative result; do not promote the mechanism.

🟢 [Done] The actual-update-matched candidate-pruning study passes all registered component gates across two tasks × five fresh seeds. Every arm executes 4,608 outcome route updates. Residual pruning reduces active candidates from 45 to nine, preserves the required shared dependencies and 100% exact accuracy, and lowers mean shifted Brier by 92.02% versus round-robin. This is a stationary toy-task component result, not production promotion.

🟠 [Next] Register a separate nonstationary dependency-change study comparing permanent pruning with bounded candidate reactivation under the same actual update budget. Freeze revision timing, unseen dependencies and recovery/false-reactivation gates before execution; do not tune this completed study.

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

🟠 [Next] Preregister training-only feasibility of bounded temporal-history pooling before acquiring another household. Keep current context and opposing-history identifiability; establish usable disjoint evidence coverage under fixed state/write budgets. No threshold sweeps on HH104 confirmation, independent-period opening or factor/runtime promotion.

日本語: 支持数だけのゲートに続き、未使用HH104で現在状態に対する局所予測根拠も事前登録して検証しました。両方の負の結果を保持し、独立期間は開きません。

简体中文：继仅基于支持数的门控之后，已在未使用的HH104上预注册并检验相对于当前状态的局部预测证据。保留两项负结果与全部独立期间封存。

日本語: HH104の固定局所根拠ゲートは0件の有効化にとどまり、現在状態対照を上回りませんでした。次は訓練期間だけで、対照可能な時間履歴を限定的に集約する実現性を事前登録します。

简体中文：HH104的固定局部证据门控启用0次，未超过当前状态对照。下一步预注册仅训练期的有界时间历史聚合可行性，保留可比较性与全部封存。

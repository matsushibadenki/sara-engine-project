# SARA Engine Roadmap

This page tracks current progress, priorities and the conditions for advancing. The historical roadmap contains the full experiment records, earlier decisions and detailed phase plans. [Roadmap history](ROADMAP_HISTORY.md)

## Current progress

| Status | Area | Evidence and remaining work |
| --- | --- | --- |
| 🟢 [Done] | Local route learning and capture | Actual local updates, delayed outcomes, retained-key probabilities and receipt acknowledgements verified in bounded synthetic runs. [Factorization evidence](SPARSE_PREDICTIVE_FACTORIZATION.md) |
| 🟢 [Done] | Independent scalar comparison | 700 receipts and 6,291 local updates matched through two reactivation cycles and their cooldowns. [Comparison](SPARSE_ROUTE_MULTICYCLE_COMPARISON.md) |
| 🟢 [Done] | Durable receipt processing | Nine-route updates and receipt consumption commit together. Sequential processing, duplicate refusal and recovery were verified with local SQLite. [Durability](SPARSE_ROUTE_DURABLE_COMMIT.md) |
| 🟢 [Done] | Authenticated delivery | Separate test producers send authenticated JSON to the stdin CLI. Key rotation, tamper rejection and idempotent replay across restarts pass. [Ingress](SPARSE_ROUTE_DURABLE_COMMIT.md#separate-process-delivery-ingress) |
| 🟢 [Done] | Pilot observation summary | Client reports aggregate refresh acceptance, answer rates, measured request/body-byte totals and maximum refresh latency. Missing measurements remain explicit; independent acceptance remains pending. [Measurement scope](ROADMAP_NEXT_READINESS.md#pilot-observation-summary) |
| ⭕️ [Pending] | Independent task and SNN effect | Qualify an independent source and test the causal contribution of the SNN mechanism against matched event-driven scalar controls. [Decision gate](R2_MECHANISM_DECISION_GATE.md) |
| ⭕️ [Pending] | Operational deployment | Independent producers, reviewed key distribution and broader writer coverage are still required. Candidate V2 launch remains disabled. [Capture candidate](SPARSE_ROUTE_CAPTURE_CANDIDATE_V2.md) |

## Research direction review — 2026-11-27

Continue CPU-first local temporal learning. The highest-value research question is whether branch-local memory of past activity helps assign a delayed outcome on a fresh independent task. The existing results support retaining the local learner and its independent scalar reference.

| Evidence reviewed | Implication for the next decision |
| --- | --- |
| On synthetic development cases, explicit temporal state raised accuracy from 0.708 to 0.900. Adding refractory dynamics gave 0.900 again. The multicycle scalar reference matched on 700 receipts, 699 outcome deliveries and 6,291 route updates. [Temporal-state ablation](EVENT_UNIT_CAUSAL_ISOLATION.md#factorial-v2-result--2026-09-16) · [Scalar comparison](SPARSE_ROUTE_MULTICYCLE_COMPARISON.md) | Keep explicit temporal state as a strong non-spiking control. Matching the same equations establishes implementation fidelity. |
| HH103 met its source gates with 73 supported history contrasts. With 56,028 changed writes per arm, joint balanced accuracy was 59.9764% versus 62.5424% for current state. Its daily Brier was 0.069395 versus 0.069184 for the transition control (lower is better). [Independent comparison](R2_CASAS_180_DAY_COMPARISON.md) | A qualified source can yield a negative learner result. Both current-state prediction and a sufficient-history readout belong in the comparison. |
| On ten changed synthetic cases, discounted counts with time-weighted error retained the final dependencies in all ten. Recovery took 320 ticks against the 256-tick limit. Route allocations had already diverged in nine of ten pairs by boundary 512. [Factorial result](SPARSE_ROUTE_MEMORY_FACTORIAL_V4.md) · [Saved allocation audit](SPARSE_ROUTE_FACTORIAL_AUDIT.md) | Preserve the recovery failure. The final gain includes feedback from route selection, so total write parity alone cannot isolate a local probability effect. |

🟢 [Done] Screened MIDAS Open at Camborne using public documentation. At one eligible cutoff per hour, 180 calendar days allow at most 4,320 arrivals and a 365-day development year allows at most 8,760. Both fall short of the fixed 70,212-arrival planning target. The station-specific visibility window, measurement provenance and effective coverage remain unresolved. Acquisition stays on hold. [Camborne metadata audit](ROADMAP_NEXT_READINESS.md#camborne-hourly-visibility-metadata-audit)

🟠 [Next] Prioritize an independent candidate whose documented cadence and split could meet the fixed support budget. Check source provenance, measurement timing and development-only isolation before proposing acquisition. A weather outcome requires its own preregistered definition. The Kitchen ON and KSEA contracts retain their original semantics.

⭕️ [Pending] Once a separately registered development audit shows 180 eligible days, recurrence at the exact current state, opposing histories and adequate support for both classes, preregister the smallest delayed-eligibility comparison. Share inputs, delivery times and state budgets, keep routing fixed and verify positive, equal actual changed writes. Compare intact, disabled and permuted eligibility with the current-state and sufficient-history controls. Require parity with the equation-matched scalar emulator. Reserve new resource work for a useful independent result, a supported causal effect and a cost question that could change the decision. The fixed negative results, old reproduction failure and test seals remain in force. [Control contract](R2_LOCAL_CREDIT_CONTRACT.md) · [Resource decision](R2_MECHANISM_DECISION_GATE.md)

Autobot dashboard verification and the publisher pilot have their own operational milestones. Their results cover monitoring and delivery reliability. [Autobot roadmap](../bot/ROADMAP.md) · [Pilot readiness](ROADMAP_NEXT_READINESS.md#live-pilot-acceptance-procedure)

日本語：研究の最優先は、独立した開発期間で、同じ現在状態でも時間履歴によって未来を識別できる課題を確保すること。成立後に枝ごとの遅延適格性を対照群と比較する。SNN固有の予測効果と省エネルギー性は未確認。

简体中文：研究首先要在独立开发期确认任务成立，即当前状态相同时，不同历史具有可检验的未来结果差异。随后比较分支局部延迟资格迹及其对照。SNN特有的预测收益和节能效果仍待验证。

## Priorities

1. 🟠 [Next] Identify a fresh independently observed delayed-outcome task. Require repeated current states, distinguishable histories, both classes and the existing support and source gates before preregistering a comparison. Use the strongest current-state control and match actual local writes. [MIDAS Open priority and Schiphol station review](ROADMAP_NEXT_READINESS.md#schiphol-station-evidence-and-midas-open-priority) · [KNMI Schiphol ten-minute archive audit](ROADMAP_NEXT_READINESS.md#schiphol-ten-minute-archive-metadata-audit) · [KSEA 112-byte format follow-up](R2_ASOS_KSEA_112_BYTE_FORMAT_REVIEW.md#legacy-documentation-and-published-examples-follow-up) · [KSEA transport V2 framing stop](R2_ASOS_KSEA_TRANSPORT_V2.md#execution-result) · [KSEA format audit and acquisition stop](R2_ASOS_KSEA_FORMAT_AUDIT.md) · [KSEA first-sensor normalization](R2_ASOS_KSEA_FIRST_SENSOR_NORMALIZATION.md) · [KSEA delayed outcome contract](R2_ASOS_KSEA_DELAYED_CONTRACT.md) · [KSEA development timestamp preflight](R2_ASOS_KSEA_TIMESTAMP_PREFLIGHT.md) · [ASOS KSEA inventory and visibility specification](ROADMAP_NEXT_READINESS.md#asos-ksea-inventory-and-visibility-specification) · [ASOS independent task screen](ROADMAP_NEXT_READINESS.md#asos-independent-task-metadata-screen) · [IEBB appendix follow-up](ROADMAP_NEXT_READINESS.md#iebb-appendix-follow-up) · [Overijssel IEBB dwelling screen](ROADMAP_NEXT_READINESS.md#overijssel-iebb-dwelling-screen) · [Minneapolis homedata release screen](ROADMAP_NEXT_READINESS.md#minneapolis-homedata-release-screen) · [Fritz and Cook provenance follow-up](ROADMAP_NEXT_READINESS.md#fritz-and-cook-supplement-and-provenance-follow-up) · [Longitudinal ambient sensing and ARAS screen](ROADMAP_NEXT_READINESS.md#longitudinal-ambient-sensing-and-aras-screen) · [TigerPlace deployment follow-up](ROADMAP_NEXT_READINESS.md#tigerplace-deployment-follow-up) · [TigerPlace and Tokyo smart-room screen](ROADMAP_NEXT_READINESS.md#tigerplace-and-tokyo-smart-room-screen) · [TIHM release follow-up](ROADMAP_NEXT_READINESS.md#tihm-release-follow-up) · [CART dictionary follow-up and TIHM screen](ROADMAP_NEXT_READINESS.md#cart-dictionary-follow-up-and-tihm-screen) · [ORCATECH and Orange4Home screen](ROADMAP_NEXT_READINESS.md#orcatech-and-orange4home-screen) · [E-care@home supplement follow-up](ROADMAP_NEXT_READINESS.md#e-carehome-supplement-follow-up) · [Native-timestamp entrance sensor screen](ROADMAP_NEXT_READINESS.md#native-timestamp-entrance-sensor-screen) · [HOMESENSE release-code audit](ROADMAP_NEXT_READINESS.md#homesense-release-code-audit) · [SHAPES follow-up and HOMESENSE priority](ROADMAP_NEXT_READINESS.md#shapes-follow-up-and-homesense-priority) · [Household uptime screen](ROADMAP_NEXT_READINESS.md#independent-source-screen-with-household-uptime) · [Four-country coverage follow-up](ROADMAP_NEXT_READINESS.md#four-country-coverage-follow-up) · [Alternative-source screen](ROADMAP_NEXT_READINESS.md#refit-alternatives--2026-11-14) · [REFIT metadata review](ROADMAP_NEXT_READINESS.md#independent-delayed-outcome-task) · [Independent-task history and gates](ROADMAP_HISTORY.md#existing-research-queue)
2. 🟠 [Next] On the operations track, prepare the independent producer pilot with reviewed trust material, revision handling and key distribution. Measure authority review, revision completeness, availability, latency and complete-path cost under the existing pilot conditions. [Live pilot acceptance procedure](ROADMAP_NEXT_READINESS.md#live-pilot-acceptance-procedure) · [Publisher pilot readiness](ROADMAP_NEXT_READINESS.md#independent-https-publisher-pilot) · [Publisher pilot context](ROADMAP_HISTORY.md#existing-research-queue)
3. ⭕️ [Pending] Promote a mechanism or begin new resource comparisons only after the independent task and causal gates pass. Existing event-work and allocation diagnostics retain their recorded results. [Research focus and promotion gates](ROADMAP_NEXT_LEVEL.md)
4. 🔴 [Later] Compare bounded local representations at equal total byte budgets, including indexes, decoding, residuals and rare-route retention. [Early route selection](#early-route-selection) · [LittleBit design reference](SPARSE_PREDICTIVE_FACTORIZATION.md#littlebit-design-reference)
5. 🔴 [Later] Evaluate exact reuse of local readouts when their dependencies are unchanged, after the independent-task and causal gates pass. [Evaluation contract](#local-readout-reuse)

6. 🔴 [Later] Evaluate bounded inference escalation after the same independent-task and causal gates pass. [Evaluation contract](#bounded-inference-escalation)
7. 🔴 [Later] Evaluate local repair with explicit constraints and a bounded stopping rule. [Evaluation design](#local-constraint-repair)
8. 🔴 [Later] Compare targeted local-state edits on isolated copies. [Evaluation design](#local-state-edit-evaluation)

### Early route selection

The Lottery Ticket Hypothesis motivates controlling separately for the retained structure and its initial state. In SARA, test whether a bounded early development prefix can identify local routes worth retaining, using only evidence available at the checkpoint. Take the checkpoint after pending outcomes have settled within the registered horizon. Preserve rare routes with insufficient support and evaluate masking before committing any structural deletion. [Frankle and Carbin](https://arxiv.org/abs/1803.03635)

At the same retained byte budget, compare selected routes with retained local state, the same selection with reset local state, a seeded random selection at the same checkpoint, and a sparse-from-start arm. Keep the unpruned run as a quality reference. Use identical follow-up receipts and delayed outcomes. Fix the update allocation in advance and verify actual local writes for the mechanism comparison. If workloads differ in an end-to-end saving comparison, report the write difference separately. Give the event-driven scalar control the same selection opportunity. Measure immediate loss separately from recovery and dependency retention, including rare histories and regime shifts. Include selection, temporary branches, indexes and fallback in total cost.

日本語：初期の開発データだけで残す経路を選び、経路選択と局所状態の保持の効果を分けて比較する将来候補を追加。

简体中文：新增后续候选，仅在开发期早段选择保留路径，分别检验路径选择与局部状态保留的效果。

### Local readout reuse

The referenced computation-reduction proposal motivates reusing a derived local readout while its key, contributing state versions and time-dependent inputs remain equal. Start with a bounded per-route cache for a read-only probability. Recompute on a missing entry, dependency change or uncertain validity. An unchanged input can still coincide with trace decay, a delayed result, expiry or a restored state, so these transitions must be covered by the reuse check. Capture the probability at receipt time and preserve it through delivery.

Require exact output and state parity with the uncached implementation, including delayed results, expiry, recovery and rare-route cases. Apply the same reuse opportunity to the event-driven scalar control and match actual local writes. Account for dependency checks, cache memory, invalidation, fallback and shared event processing using the existing [work accounting contract](R2_EVENT_WORK_ACCOUNTING.md). Attribute a benefit to SNN dynamics only if an advantage survives the matched scalar comparison and the existing causal gate.

Treat error-based omission of local writes as a separate learning intervention. Small errors alone provide insufficient evidence to skip updates that may preserve rare dependencies or aid recovery. Semantic planning and text rendering separation remains outside this route-level comparison.

日本語：依存状態が一致する局所読み出しの再利用を将来候補に追加。遅延結果・期限切れ・復旧を含む完全一致と、同じ再利用を備えたスカラー対照を条件とする。

简体中文：将依赖状态一致时复用局部读出列为后续候选，要求在延迟结果、过期和恢复时完全一致，并采用具备相同复用机制的标量对照。

### Bounded inference escalation

Start with the existing current-state readout. Use development-only evidence of support and uncertainty to decide whether to consult the temporal readout. Escalate on unknown keys or insufficient support. Fix the policy and quality margins before evaluation. Inference routing leaves the receipt lifecycle and local write schedule intact.

Compare always-current, always-temporal and conditional readouts, and give the event-driven scalar control the same routing opportunity. Check aggregate quality and the quality of predictions that stayed on the cheap path, including rare histories and delayed outcomes. Use the existing work-accounting contract to include routing, both readouts on fallback, shared event processing and memory. Report fallback frequency and tail latency alongside total work. Evaluate this policy separately from exact reuse before testing their combination.

日本語：現在状態から読み出し、根拠が不足する場合に時間履歴を参照する段階推論を将来候補に追加。

简体中文：将分级推理列为后续候选，先读取当前状态，证据不足时再读取时间历史。

### Local constraint repair

CRAM motivates updating one variable at a time, scoring a change from its incident constraints, and checking whether further search is needed. Explore this on a small CPU-only constraint task with fixed observations. Maintain an index from variables to constraints and update only affected scores. Learn any weights from local training evidence without differentiating through the search. Compare learned weights with hand-set weights and a direct constraint solver. [Hoover, Blayney and Whittington, arXiv v1](https://arxiv.org/html/2611.17309v1)

Accept early only when an independent verifier confirms the required constraints. For tasks with soft constraints, register the quality requirement separately. Keep a fixed step budget and record exhaustion or stalled search. Compare against fixed-budget search and give the event-driven scalar control the same stopping rule. Count score updates, queue operations and verification work at matched solution quality. The paper's GPU timings and puzzle results apply to its own benchmarks. Use the existing promotion gates before new resource comparisons or claims of SNN-specific benefit.

日本語：局所的な制約修復と検証付き早期停止を将来候補に追加。手設計と直接ソルバーも対照に含める。

简体中文：将局部约束修复和经验证的提前停止列为后续候选，并与手工设置及直接求解器比较。

### Local-state edit evaluation

Adapt Fan et al.'s paired edit evaluation to a bounded copy of SARA's local route state. Use development diagnostics to select a fixed set of fields, apply the proposed edit to a copy, and validate the exact write set and resulting numerical deltas. Keep the original snapshot available for recovery. Treat recovery from injected damage and improvement of intact state as separate hypotheses. [Fan et al., arXiv v1](https://arxiv.org/html/2611.21885v1)

Compare targeted edits with a no-edit reference and a seeded control that changes the same number of local fields by comparable magnitudes. Fix the evaluation protocol before proposing edits and use paired inputs and delayed outcomes. Check target recovery and unrelated-route retention, including rare histories. Record withdrawals and regressions. Candidate selection uses development evidence, with sealed evaluation subject to the existing gates.

🔴 [Later] Use evolution strategies as a design reference for a seed-reproducible, CPU-only search over a fixed set of branch-local fields on isolated snapshots. Register the allowed write set, seed and generator version, candidate count, perturbation scale, delayed-outcome horizon and stop rule before evaluation. Replay the same development receipts and settled outcomes from the same snapshot for each candidate. Restore from the original snapshot between trials and verify the restored state, since applying inverse floating-point edits is insufficient for exact recovery. Compare with the unchanged snapshot, a budget-matched control that evaluates the same proposals and selects one at random and an event-driven scalar with the same search opportunity. Keep actual changed local writes matched in comparisons that allow learning. Record candidate replay, temporary state, writes and delayed delivery work separately. [Qiu et al., arXiv v3, §3.2 and Appendix A.8](https://arxiv.org/html/2509.24372v3)

The paper evaluates population search with global response-level rewards on GPU-based LLMs. SARA's end-to-end cost and spike-specific effect require separate evidence. This candidate evaluates offline local edits and preserves the existing online credit assignment, independent-task and causal gates, negative findings and test seals. Any experiment requires separate preregistration after those gates.

日本語：局所状態の複製に対する編集を対応比較し、故障からの回復と正常状態の改善を別々に検証する。固定した枝の局所状態を対象に、乱数シードで再現できるオフライン探索を将来候補に追加する。

简体中文：在局部状态副本上配对比较编辑效果，分别验证故障恢复与完好状态的改进。将可用随机种子重现的分支局部状态离线搜索列为后续候选。

### Reconstruct local arithmetic from a written specification

Adopt the iWSP reconstruction audit as a research and implementation target for SARA's local readout. The paper tests whether a fresh session can rebuild a model from a short description. SARA can apply a stricter, numerical check to its deterministic local arithmetic. [Mehta et al., arXiv v1](https://arxiv.org/html/2611.18114v1)

The specification defines allocation, prior mass, probability and the receipt-time error increment. Give only the specification and input interface to an independent implementer. Freeze the reconstructed implementation before comparing it against the existing eleven literal records. Require exact agreement for probability and error increment, including the distinction between an unallocated key and allocated zero counts. Add separately registered boundary cases to distinguish rules that happen to agree on the existing examples. Keep the original expected values and test seals intact. A pass would establish reproducibility of the documented rules on those cases. [Existing independent arithmetic test](../tests/test_route_independent_scalar_preflight.py)

🟢 [Done] The written rules and boundary expectations were registered in commit `e714ec3`. The standalone implementation was committed as `0f1d8be` before running the comparison. All eleven literal records and four registered boundary cases matched exactly for both outputs. This was a separate implementation with no imports from the existing arithmetic, developed in the same session as the specification. [Registered specification](LOCAL_ARITHMETIC_RECONSTRUCTION.md)

🟢 [Done] A separate agent received only the registered calculation rules and interface, with no inherited conversation or tool use. Its implementation was frozen in `00c46bf` before comparison. Probability and receipt-time error increment matched exactly on all 11 literal records and four registered boundaries. The unallocated and allocated-zero cases retained distinct probabilities of null and 0.5 with equal error increments. The comparison used only registered hand records.

🟢 [Done] The registered rounding audit found zero probability mismatches and eight increment mismatches across 8,192 cases. Multiplication matched the rational oracle in those eight cases. Specify the squaring operator explicitly in any future bitwise contract. [Audit and first counterexample](LOCAL_ARITHMETIC_RECONSTRUCTION.md#rounding-audit-results)

🟢 [Done] Specification v2 fixes the square to binary64 multiplication and defines the rounding order. The versioned implementation matched the rational reference for both outputs on all 8,207 cases. The eleven literal records and four boundary cases also retained exact agreement with their registered expectations. [Specification v2](LOCAL_ARITHMETIC_RECONSTRUCTION.md#local-arithmetic-specification-v2)

🟢 [Done] A separate agent received only the v2 calculation rules and interface, without inherited context, repository access or expected values. Its code was frozen in `bb8df9b` before testing. The blind reconstruction matched both rational-reference outputs on all 8,207 cases, including the unallocated and allocated-zero boundaries. The audit used the existing 11 literal records, four boundaries and 8,192 generated cases.

🟢 [Done] The v2 transition and count-limit audit was registered in `3c3fa46`. Both frozen v2 implementations matched the rational reference for both outputs on all 4,352 cases per implementation. Coverage included every tick scale transition, maximum and near-maximum masses, and all history, weighting and outcome combinations. The audit used generated inputs and kept the historical evidence and test seals intact. [Registered cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-limit-and-transition-audit-preregistration)

🟢 [Done] Both frozen v2 implementations rejected all 18 registered invalid inputs with `ValueError`. The cases covered out-of-range ticks and counts, booleans in integer fields, unknown rules and outcomes, and nonzero masses on unallocated keys. Both zero-mass controls still matched the rational reference. [Registered invalid input cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-invalid-input-audit-preregistration)

🟢 [Done] Each frozen v2 implementation rejected all 32 registered invalid types with `ValueError`. The audit covered floats, numeric strings and nonfinite numbers in integer fields, non-booleans in allocation, and non-strings in history, weighting and outcome. [Registered type cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-invalid-type-audit-preregistration)

🟢 [Done] Both frozen v2 implementations rejected all 13 registered subclass and conversion-probe cases with `ValueError`. No instrumented conversion or comparison method ran. [Registered audit](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-subclass-and-coercion-audit-preregistration)

🟢 [Done] Both frozen v2 implementations matched the rational reference for both outputs on 32,768 generated cases each. The audit covered every tick, asymmetric small and maximum counts, and all history, weighting and outcome combinations for those count pairs. [Registered cases](LOCAL_ARITHMETIC_RECONSTRUCTION.md#v2-all-tick-audit-preregistration)

日本語　短い仕様だけから局所計算を再実装し、固定記録で数値と欠損の意味を照合する。

简体中文　仅依据简短规格重建局部计算，并用固定记录核对数值与缺失状态。

## Selected execution gates

- 🟢 [Done] Fixed CartPole comparison completed with a negative V2 gate; no retry. <!-- gate:cartpole-fixed-comparison -->
- ⭕️ [Pending] New task/hypothesis and fresh development identities before another local-credit trial. <!-- gate:cartpole-fresh-task -->
- ⭕️ [Pending] Trusted owner/revision/pause and outcome-free access before actual receipt capture. <!-- gate:authoritative-receipt-capture -->
- ⭕️ [Pending] SNN-specific causal benefit over matched event-driven scalar controls. <!-- gate:snn-specific-advantage -->

## Evidence and phase plans

- [Roadmap history through the delivery ingress milestone](ROADMAP_HISTORY.md) preserves the previous roadmap, including negative and failed attempts, execution decisions and Phases 6–20.
- [Implemented features](IMPLEMENTED_FEATURES.md) lists completed capabilities. [Pre-v1.1 archive](old/ROADMAP_v1_1_completion_archive.md) records the release baseline.
- [Next-level roadmap](ROADMAP_NEXT_LEVEL.md) tracks promotion conditions and links to the detailed plans for Phases 21–46.
- Consumed experiments and sealed splits retain their recorded status. The selected execution gates are checked by the [roadmap gate checker](ROADMAP_GATE_CHECK.md).

日本語：合成入力での局所学習、独立スカラーとの照合、復旧可能な永続化、認証付き受信まで完了。次は独立課題の選定と外部送信元の試験準備。SNN固有の効果と資源上の優位性は検証待ち。

简体中文：已完成合成输入下的局部学习、独立标量对照、可恢复持久化与认证接收。下一步是筛选独立任务和准备外部发布方试点。SNN 特有的因果效果和资源优势仍待验证。

## Release baseline audit

* **ROADMAP closure audit:**
  - DONE: release-critical path is complete for the v1.1 baseline.
  - DONE: observed-only evidence remains labeled until stronger external or physical measurements pass.
  - DONE: long-term research backlog is separated into Phase 6 and later roadmap work.
  - DONE: roadmap completion audit is present for the current baseline.
  - DONE: research product completion gate is part of the active release evidence surface.

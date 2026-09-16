# Minimal Event-Computing Unit Isolation

## Research decision — 2026-09-16

[Next] The checkpoint prerequisite is satisfied under the explicit Python 3.10 runtime boundary. The current result supports the usefulness of sparse routing, local temporal state, bounded local updates and confidence routing, but it does not identify membrane potential, spike emission or refractory dynamics as the cause of the observed quality.

[Done] The immutable v1 preregistration is frozen at SHA-256 `f8c2c2d2122af7ff4691e8738a9f8b490d94e8af029195737fe04fab459feac7`. It fixes the four arms, five seeds, disjoint causal families, seven ablations, equal inputs and learning signals, resource ceilings, ordered decisions and stop rules before candidate implementation. The validator writes only a managed report and confirms that real frozen partitions remain closed.

[Done] Implemented the shared deterministic development generator and all four arms with one common bounded mistake update. Dynamic membrane/refractory state is isolated within each episode. On 250 development episodes, A and B are exactly equivalent at `0.720`; C reaches `0.912` by solving the timing and refractory families while remaining at `0.560` on branch conjunction; D reaches `1.000` and solves branch conjunction. Maximum event work is `6/6/20/22` and retained state is `2,970/2,970/19,373/26,240` bytes for A/B/C/D, within the frozen ceilings. This is development evidence only, not an accepted causal result.

[Done] Implemented all seven frozen interventions, deterministic replay digests, capacity-matched scalar state and per-seed ordered gains. The harness passes, but the development causal gate fails exactly one check: disabling refractory dynamics changes neither C predictions nor its refractory-family accuracy (`1.000`, delta `0.000`). Time shuffle drops aggregate C accuracy by `0.108`, state reset by `0.192`, spike-association shuffle by `0.144`, branch shuffle drops D branch-family accuracy by `0.400`, and outcome shuffle drops aggregate D accuracy by `0.528`. The capacity-matched scalar arm remains exactly equivalent to A despite retaining at least D's state bytes. The runner exits non-zero and held-out causal families remain unopened.

[Next] Retain v1 as a development negative result. Freeze a new, non-amending v2 protocol that separates temporal state without refractory dynamics from an otherwise matched refractory arm. Do not carry the current perfect development scores into acceptance thresholds, and do not consume the v1 held-out split.

## Factorial v2 result — 2026-09-16

[Done] Frozen v2 independently at SHA-256 `07afe1d0bd4389c1d71acecc30e360335ade36daa380579c68e8137d865a5494`, with five fresh seeds and namespace `event-unit-v2`. It compares compact events, temporal state without membrane/refractory dynamics, matched temporal state with refractory neurons, and temporal state with fixed branch structure.

[Done] The v2 harness passes but the development gate is negative. Aggregate accuracy is B `0.708`, C `0.900`, R `0.900`, D `1.000`. Temporal state adds `0.192`; time shuffle removes `0.084` and state reset removes `0.192`. Branch structure adds `0.100`, and branch shuffle removes `0.500` on the conjunction family. Refractory adds `0.000` in aggregate and is `-0.020/-0.020/0.000/-0.020/-0.020` across the five fresh seeds. Disabling refractory leaves aggregate predictions at `0.900`. Added capacity alone exactly preserves the compact-event predictions.

[Done] The v1 conclusion therefore replicates on fresh identities: local temporal pair state and fixed branch structure are useful in this synthetic scope; membrane/refractory dynamics provide no benefit and are slightly harmful in four of five seed-level comparisons. The v2 runner exits non-zero, and both v1/v2 held-out splits remain unopened.

[Done] Stopped the refractory line and retained compact event plus bounded local temporal-pair state as the smallest supported forward unit, with branch structure only for branch-specific conjunction. The requested beyond-pair development task was frozen and executed below.

## Beyond-pair development result — 2026-09-16

[Done] Frozen the beyond-pair protocol before implementation at SHA-256 `9a8005ed0d9e08adabadbf44249386dba332fdac9e758e1e366281adac6dd1af`. Five fresh seeds compare adjacent pair state, bounded triplet state and triplet-plus-fixed-branch state on balanced parity, nonadjacent match, three-event composition, branch conjunction and pair-sufficient controls.

[Done] The development harness passes but the causal gate fails. Pair-state ambiguous-family accuracy is `0.483`; triplet reaches `0.617`, a nominal `+0.133`, and history truncation removes `0.133`. However event-order shuffle reaches `0.622`, so the targeted order-control delta is `-0.006`, and seed-level triplet gains do not have a consistent positive sign. Fixed branch structure solves its conjunction family (`1.000` versus `0.483`) and branch shuffle removes `0.500`, but this does not rescue the failed triplet causal claim.

[Done] Retain this as a negative result: a categorical triplet key can increase development accuracy without demonstrating order-dependent computation. The runner exits non-zero; held-out identities remain unopened.

[Done] Did not increase lookup order; instead froze and executed the symbol-disjoint transition-rule task below.

## Symbol-disjoint transition-rule result — 2026-09-16

[Done] Frozen protocol `58c1959ced4f3ede26f2bc4c85066b826bb4c7d60cc5219ec29fcce1deb5393e` before implementation, with disjoint training symbols `0–5`, development symbols `10–15`, and five fresh seeds. Categorical lookup reaches `0.538` on relational families, relational transition reaches `0.575`, and relational composition reaches `0.758`.

[Done] The development gate fails. Relational transition gains only `0.038`, reaches neither the `0.85` accuracy gate nor the `0.20` gain gate, and relation shuffle removes only `0.083`. The task combines direction, equality and distance rules without an observable context cue, so identical relation observations can require conflicting labels. This is retained as a task-identifiability negative result, not evidence against relational transfer in a well-specified context.

[Done] The isolated composition result is positive but insufficient for overall adoption: composition accuracy improves from `0.350` to `0.900`, and composition reset removes `0.550`. Held-out remains unopened and the runner exits non-zero.

[Done] Preregistered the context-observable version below; implementation remains gated on its identifiability audit.

[Done] Frozen the context-observable v2 protocol before candidate implementation at SHA-256 `c79206dc8bde4d6ef8b0fb593a6a3da2982d718ffb70d89f2c492076da9d3082`. It uses five fresh seeds, disjoint symbols `20–25` versus `30–35`, and supplies the same bounded rule-context cue to categorical, relational and compositional arms.

[Done] Added materialization protocol `7681511e5b63d0a3fdf10d3e6bafa7129fd5b610431a80ab11459a9f13cbfffd` without modifying the parent. The pre-execution audit materializes 600 training and 300 development episodes and passes signature uniqueness, per-context/seed balance, v2 namespace, symbol disjointness and identity disjointness across 39 signatures. Held-out is not materialized; candidate execution is now authorized.

[Done] Implemented the contextual arms and controls below while keeping held-out closed.

[Done] Implemented the three contextual arms and all frozen development controls. Relational accuracy is categorical `0.796`, relational `0.900`, compositional `0.996`. Context shuffle removes `0.313`, relation shuffle `0.454`, and composition reset `0.383`; mechanism controls are strong. However categorical lookup exceeds its frozen `0.60` ceiling and relational gain is only `0.104` versus the required `0.20`.

[Done] The gate fails because development outcomes were observed online, allowing the categorical arm to memorize repeated development combinations. The v2 protocol did not freeze whether development updates were disabled, so this is retained as an evaluation-boundary negative result rather than repaired post hoc. The runner exits non-zero and held-out remains closed.

[Done] Froze and executed the zero-shot transfer protocol below; online adaptation remains intentionally unexecuted.

[Done] Frozen zero-shot protocol `5216b01ae192e82aad94c2bb8add001bb9ab42f79d00ae67ffd695189146efc5` with five fresh seeds and symbols. Development performs no learning; online adaptation is a separate, non-gating copy and was not executed.

[Done] Zero-shot relational transfer is strong: categorical lookup `0.500`, contextual relation `0.963`, contextual composition `1.000`. Relation gain is `0.463`; context and relation destruction remove `0.425` and `0.433`. However composition adds only `0.150` and composition reset removes `0.150`, below the frozen `0.20` gates. The overall runner exits non-zero, held-out remains closed, and thresholds are unchanged.

[Done] Retained the positive zero-shot relational result and negative composition gate separately, stopped composition tuning, and preregistered the independent replication below.

[Done] Frozen independent replication protocol `bc7ff8cb7891e02e3df57c5f15cd88ba3155494543837c82b76563ceea61f4cf` before generator implementation. It requires a new generator that cannot import prior transition generation, a separately implemented label evaluator, five fresh seeds, disjoint prime-valued symbol sets, unique zero-shot development pairs and train-only updates. Composition is explicitly non-gating.

[Done] The exhaustive pre-generator feasibility audit correctly blocks v1. All frozen training and development values are odd primes, so `same_parity` produces only label `1`; balanced generation is impossible in both splits. Ascending/equal, bounded jump and interval direction have both labels. No candidate or held-out data was generated, and the audit exits non-zero.

[Done] Preserved v1 unchanged and froze v2 at `b382a2fca20e8bf9a7eae010c88af8ec818048e64a15db222add30c780b41c5f`. Its fresh disjoint value sets contain both parities. The exhaustive pre-generator audit confirms both labels for every context and split, at least 12 unique examples per label where development requires eight, and keeps held-out closed.

[Done] Implemented a new generator without importing the prior transition generator, a separately coded evaluation oracle, and two zero-shot arms. Across five fresh seeds, categorical accuracy is `0.500` and contextual relational accuracy is `1.000`; every seed gains `0.500`. Context shuffle removes `0.200` and relation shuffle removes `0.509375`. Exact replay, capacity matching, event/state/feature ceilings, per-seed development uniqueness and zero development updates pass. This is independent synthetic development evidence, not held-out or production evidence.

[Done] Frozen independent held-out protocol `d79c3463fe74debc85d92348944a7425303d3aa9a221a5954684f97cc4c05e51` pins candidate source `388181f9…e343c5f`, uses fresh values and seeds, forbids candidate changes after materialization, and permits one execution only.

[Done] The one-shot held-out gate passes. Categorical/relational accuracy is `0.500`/`1.000`; every seed gains `0.500`. Context and relation destruction remove `0.209375` and `0.515625`. Exact replay, zero held-out updates, capacity matching and all resource ceilings pass. The immutable result digest is `8e8d298dead511a61807fcc4eaf690ced10245ea998d73651006787c68115581`.

[Done] Minimal-unit conclusion: this line supports sparse events plus explicit local relational state under a bounded local update. It does not support membrane potential, refractory dynamics or spike emission as the cause of the gain. The relation descriptors are supplied inductive structure, so this result is not evidence of autonomous relation discovery or deep credit assignment.

[Done] Frozen the two-stage Local Credit Packet protocol at `d7cdfc6550fa1e1b2bf2441a401442a50ca2ebfb9e8df709f37bd24b4e0bfaa5`. It fixes no-credit, broadcast, non-adoptable gradient-like and sparse branch-addressed arms; the seven packet fields; depth/TTL/fanout; equal forward/update budgets; eight interventions; leakage bans; five seeds; resource ceilings and stop rules. Candidate and development rows do not yet exist.

[Done] Frozen materialization supplement `49b53490c85cfeb9328289e38b61efe084bb36070d40dce9dfa61c668bc8a0bd` and audited 1,920 training/development templates. The row digest is `fc8b8395…11b91b`. Labels and scheduled actions are exactly balanced per seed/context, train/development identities are disjoint, the target rule is stable, and both counterfactual actions are present. Final-stage, route-alias, outcome-ID-bit and source-ID-bit accuracies are `0.500`, `0.500`, `0.516` and `0.504`, all below the frozen `0.55` leakage ceiling. Held-out remains absent.

[Done] Implemented a 21-byte seven-field packet and bounded depth-two eligibility path. Across five seeds, no-credit/broadcast/gradient-like/packet accuracy is `0.500/0.500/1.000/1.000`. Packet gain is `+0.500` over both adoptable controls. Route and sign destruction reduce accuracy to `0.000`; eligibility reset, TTL zero and depth one reduce it to `0.500`. Forward traces match across arms, peak eligibility is one entry, state is 2,272 bytes, and all replay/resource gates pass. Result digest: `e64d2e54…0b801`.

[Done] Replay-disabled remains exactly `1.000`, so replay is not credited for this short delayed task. The positive result isolates sparse branch-addressed delayed credit, not replay, autonomous structure discovery, real-event generalization or a backpropagation replacement claim.

[Done] Frozen the harder three-stage targeted-replay protocol at `ea61d37c8cf55d8c125d9b1be7c359fd659b9002f50fe775b449cf7cf9daf709` before materialization. All delays (`12/24/48`) exceed direct TTL `8`; at least four and at most eight eligibilities overlap. A targeted anchor is capped at eight 48-byte entries and one lookup per outcome; global history scans and label-bearing keys are forbidden. Five arms and eight replay/anchor controls are fixed.

[Done] Frozen materialization supplement `afbe5cb3eb5c2703bab3fb7ea17dce8b0f0bec9ef0934c0f8dd4fbc3767218df` and audited 2,880 rows (`dcde1cc4…c481a`). In 360 waves, exactly eight eligibilities overlap. Every outcome arrives after direct TTL and before anchor expiry; peak anchor occupancy is eight. Final-stage and route-digest majority probes are `0.500`; source/outcome ID low-bit probes are `0.506`/`0.510`. These bounded probes do not prove information-theoretic independence, but identifiers are constructed before target lookup and forced actions are balanced within each context.

[Done] Targeted-replay development passes the frozen gate on five seeds. No-credit/broadcast/direct-packet/replay/gradient-like accuracy is `0.500/0.500/0.500/1.000/1.000`. Replay-disabled, anchor-expired and depth-two controls fall to `0.500`; wrong route falls to `0.000`, wrong context to `0.400`. All direct traces expire, and each of 384 training outcomes per seed uses one anchor lookup and at most one backward event. The anchor wire payload is 33 bytes, the credit packet is 21 bytes, peak deep-counted Python state is 5,492 bytes, and the result digest is `3372c0d7…de712`.

[Done] Scope limit: the intermediate stage is a recorded route, not an independently trained circuit. Anchor replay reconstructs that route directly by source ID. Thus this is evidence for bounded targeted delayed credit under overlapping routes, not yet for learned credit passing through multiple trainable layers or for real-event generalization.

[Done] Froze the trainable multi-hop protocol `be5a941959c7919efb236b16495e604d8822716fbeeeef8fcab315a42aa38fb2` before implementation. Circuit A sees only its own cue and selects a branch; circuit B sees that branch plus a private cue and learns a local binary code from an explicitly disclosed auxiliary target. The global target is the XOR of the two private maps. Outcome→B and B→A are separate edges; A may receive only the seven-field packet. Broadcast, direct-anchor shortcut, no-credit and non-adoptable oracle arms share forced training actions and update opportunities.

[Done] Frozen materialization supplement `ef3cf6f26122ca1964c7bf49a53ef3bc0d7c4430b8ec7689dd05cddca86f1bc5` and audited 3,840 rows (`789b8a77…c71b1b`). Each cue pair has all four forced A/B action pairs twice; both private maps and global outcomes are balanced. Without B's private state, the direct shortcut and global outcome identify A's target at only `0.500`; with the B-local target, the bounded local inverse identifies it at `1.000`. Opaque-ID low-bit probes stay at or below `0.511`.

[Done] The synthetic development gate passes on five fresh seeds. No-A-credit/broadcast/direct-shortcut/packet/oracle global accuracy is `0.500/0.500/0.500/1.000/1.000`. B-map reset at packet interpretation falls to `0.500` while B forward local accuracy remains `1.000`; shuffled B-local teaching falls to `0.525`, and wrong route/sign to `0.000`. Packet delivery disable, A eligibility reset and global outcome shuffle fall to `0.500`. Packet size is 21 bytes, at most two backward events cross the two edges, A/B use 16/8 learned features, and peak deep-counted state is at most 3,232 bytes. The direct shortcut receives at least as much state allowance without changing its `0.500` predictions. Result digest: `bb3ce15e…cb6284`.

[Done] Scope limit: B receives an auxiliary local target, training actions are forced and balanced, and the task is a synthetic binary XOR decomposition. This establishes that a learned B-local code can turn a delayed outcome into a sparse A-directed packet under these conditions. It does not establish end-to-end learning without B supervision, general deep credit assignment, or real-event transfer.

[Done] Frozen independent two-circuit held-out protocol `5b542db726cf652c69158fa15cb007c3afa06ff615016a201bde9c809470402f` with candidate source `68e87e58…3cc61`, packet source `c675e688…91ec1`, five fresh seeds and disjoint A/B cue spaces. An independent SHA-rank map generator and separately coded oracle agree on all 3,840 rows (`52992344…cb4c`). The pre-execution audit passes complete action factorial, train/held-out identity isolation, balanced outcomes and 0.500 direct-shortcut/outcome-only probes. Control shuffles and state reserve were frozen in execution supplement `ed4c4f54…e992d` before scoring.

[Done] The one-shot held-out gate passes without candidate changes or threshold adjustment. No-A-credit/broadcast/direct-shortcut/packet/oracle accuracy is `0.500/0.500/0.500/1.000/1.000`; all five seed gains are positive. B-local map reset during packet interpretation falls to `0.500`, shuffled B-local teaching to `0.5625`, and wrong route/sign to `0.000`. Packet bytes, two backward events, 16/8 feature counts, ≤3,512-byte peak Python state, capacity allowance and exact replay pass. The exclusive result artifact was consumed once and has SHA-256 `6c4beca8…2ec87`; rerunning the held-out scorer is forbidden.

[Done] A separately preregistered five-seed development test reduced B-local target coverage by cue from eight to six, four, and zero. Packet global accuracy was respectively 100%, 87.5%, 77.5%, and 50%; at six cues the matched direct shortcut was 50%. Label-blind cue selection, capacity matching, B-map reset, route/sign interventions, exact replay and resource checks passed. Protocol digest: `cb48e2d6…7887`; result digest: `ea1b2763…61314`.

[Done] A further preregistered five-seed development test removed B calibration and delayed every exact B-local label by 0/16/64/256/512 training episodes. Packet accuracy was 100% through delay 256 and 50% at delay 512, when no labels were available before scoring. The delay-64 direct shortcut was 50%; B-map reset and wrong packet route/sign removed the gain. The maximum pending queue was 512 entries and maximum measured Python state was 7,709 bytes. Protocol digest: `78ee896b…40115f`; result digest: `3bc174fa…68410`.

[Next] Test noisy or misleading B-local evidence and a late-teaching schedule with insufficient post-teaching updates. This synthetic delay result does not prove long-delay credit in a general task: once exact B labels arrive, many later training episodes remain, and no retroactive A replay is attempted. Real-event transfer and physical-energy gains remain untested.

日本語: 2回路の局所creditは、新しいseedとcueを使った独立held-outでも成功しました。ただし、Bに補助的な教師信号を与えた合成課題に限られ、教師信号なしの深い学習と実イベントへの転移は未検証です。

简体中文：双回路局部信用分配在新的种子和线索构成的独立留出测试中也成功了，但仍限于给 B 提供辅助监督的合成任务；无辅助监督的深层学习和真实事件迁移尚未验证。

The causal comparison will use four arms:

1. **A — Scalar local learner:** no spike and no neuron dynamics.
2. **B — Compact event unit:** explicit sparse events without retained membrane or refractory state.
3. **C — Stateful spiking neuron:** retained membrane potential, spike and refractory dynamics.
4. **D — Dendritic / structural neuron:** branch-local state and structure, with structural changes disabled in the first comparison and enabled only in a separately registered follow-up.

Every arm must receive the same route identities, chronological examples, context memory, prediction head, outcome signal, update opportunities, accepted evidence and hard resource envelope. Capacity matching includes route count, learned scalar count, retained bytes, event work and tuning attempts. If an arm requires extra state, an explicit capacity-matched control receives an equal state allowance without the proposed mechanism.

The first synthetic task must independently vary four factors: order-sensitive timing, persistent state, refractory suppression and branch-specific conjunction. It must include time-shuffled, state-reset, spike-count-preserving and branch-shuffled controls. Five fixed seeds and a held-out causal-family split are required. BPI and Sepsis frozen partitions are not tuning data and must not be reopened for mechanism selection.

Primary outcomes are held-out task quality, calibration, causal-intervention sensitivity, deterministic replay, update count, event work, retained state bytes and CPU latency. A mechanism is credited only when its targeted ablation removes an advantage that survives scalar and capacity-matched controls. Exact prediction equivalence is evidence against a mechanism-specific quality claim, even if the implementation remains useful as an event representation.

Stop after the smallest successful arm. Do not implement `Local Credit Packet`, multi-layer backward messages or structural growth during this stage. Those belong to a later preregistration after the minimal forward/event unit is identified.

日本語: まず同じ routing・memory・budget・learning signal の下で、scalar、compact event、stateful spike、dendritic structure を比較します。実験runtimeは再現確認済みのPython 3.10に固定します。

简体中文：首先在完全相同的路由、记忆、预算与学习信号下比较标量、紧凑事件、有状态脉冲和树突结构。实验运行环境固定为已验证可复现的Python 3.10。

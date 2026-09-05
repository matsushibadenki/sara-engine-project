# Project Policy

SARA Engine is developed as a CPU-first, SNN-based AI project. New work must preserve the engineering constraints below and the repository's AGENTS.md instructions.

Roadmap entries cannot authorize exceptions to AGENTS.md. Design choices within those boundaries may change when evidence justifies them; record the reason, scope, and expected tradeoff. Offline ANN references must remain explicitly separate from the SNN candidate and are not evidence of autonomous local learning.

## Research Focus Reset — 2026-09-05

The immediate objective is a small CPU temporal learner whose predictions measurably improve through bounded local plasticity. The authoritative execution order is the Research Focus Reset in [ROADMAP_NEXT_LEVEL.md](ROADMAP_NEXT_LEVEL.md#research-focus-reset--2026-09-05). Existing phase records remain evidence history, not permission to start every unfinished mechanism.

- **Separate system usefulness from SNN contribution.** Retrieval, rules, exact caches, human-authored relations, and teacher-produced targets are valid engineering components. Report their contribution separately; do not attribute their answers to learned spikes without a causal ablation.
- **One closed learning loop first.** Select an existing sparse neuron/layer, define timestamped input, recurrent state, local eligibility, delayed outcome, bounded update, and minimal readout. Do not wait for anonymous concepts or recursive canopy topology to test elementary credit assignment.
- **Enforce budgets before work.** Cap input admission, fan-out, trace entries, new connections, replay, queues, and diagnostic bytes before allocation or traversal. A reported event count or a fallback after exceeding the budget is not a hard bound. Count ingestion, sorting, maintenance, verification, and serialization as well as spike propagation.
- **Establish necessity and sufficiency separately.** Compare intact learning with frozen weights, shuffled feedback, timing-destroyed inputs, and a comparable non-spiking temporal baseline. Preserve event counts and task information when constructing controls; include a task where timing should not matter.
- **Treat external validity as a separate result.** Repeated seeds and shuffled conditions do not create independent source examples. Split by source/session/generator and report independent-unit counts, uncertainty, per-task quality, abstention coverage, and errors. Development results cannot reopen a consumed held-out attempt.
- **Limit active scope.** Execute one causal-learning experiment at a time. Admit a new biological mechanism only when a documented failure predicts which metric it should improve over a simpler control. Keep negative results immutable and require fresh held-out data for a revised hypothesis.
- **Respect evidence tiers.** Test passage establishes covered behavior; synthetic success establishes limited mechanism evidence; independent task benefit establishes scoped usefulness. None alone establishes general intelligence or physical energy savings.

日本語: 当面は、局所学習で未知の時間系列に対する予測が改善する最小のSNN閉ループを実証する。検索・規則の成果とSNNの寄与を分け、処理前に資源上限を守る。

简体中文: 当前优先验证最小的 SNN 闭环，证明局部学习能够改善对未见时间序列的预测。分别报告检索、规则与 SNN 的贡献，并在执行前限制资源消耗。

## Core Design Rules

- Do not make correctness or normal runtime learning depend on global gradient backpropagation, end-to-end differentiation, or storage of a whole-network backward graph.
- Backward information is permitted and encouraged when bounded and auditable. Examples include branch-local activity history, bAP-like events, prediction-error or outcome events, global reward/modulatory scalars, replayed consequences, and source-linked correction signals.
- A backward signal must not be described as a gradient unless it is mathematically a derivative. Local eligibility, temporal proximity, branch contribution, novelty, reward, and modulation may assign credit without computing a global differential gradient.
- Prefer hierarchical local credit assignment: trace recently eligible branches, sub-branches, contacts, and synapses through explicit sparse structure, then apply local plasticity and structural-plasticity rules under fixed state/event budgets.
- Treat learning as an internal distributed capability of each bounded unit, not only as an external optimizer. A unit may compute, retain recent eligibility, receive feedback/modulation, update local parameters, and propose structural change through the same sparse event contract.
- Prefer algorithmic self-similarity over prescribed biological shape: the same bounded `local processing -> integration -> feedback -> plasticity` contract may recur at contact, branch, neuron, circuit, and module scales, while topology is allowed to emerge only through controlled growth and pruning.
- Do not make the SNN candidate depend on matrix calculations; use explicit sparse adjacency, scalar local state, and bounded event operations. Array storage or file decoding alone does not establish a matrix-computation dependency.
- Do not require GPUs for correctness or normal operation.
- Prefer sparse event routing, local plasticity, STDP-style updates, homeostasis, replay, and bounded memory structures.
- Prioritize accuracy and energy efficiency together; do not improve one by hiding unbounded cost in diagnostics or repair loops.
- Prefer Python plus Rust/PyO3 for practical performance, while keeping the public interface usable from Python.
- Code comments and logs should be written in English.

## Output Locations

Generated files must stay in managed directories:

- `data/raw/`: collected source data and exported chat JSONL files.
- `data/processed/`: cleaned corpora and finalized processed datasets.
- `data/interim/`: temporary preprocessing artifacts.
- `workspace/`: reports, summaries, operational runbooks, scratch files, and non-final artifacts.
- `models/`: final model artifacts.

Do not write generated artifacts to the repository root. Prefer `src/sara_engine/utils/project_paths.py` for new read/write paths.

## Release And Gate Policy

- Phase 3 completion requires `phase3_completion.passed == true`, `completion_score >= 1.0`, and no failed check entries.
- Phase 4 completion requires required metrics, threshold results, and numeric quality metrics to pass.
- Final production promotion should use the `extended` soak profile and `operational_readiness.py --strict-production`.
- Operational repair automation should be traceable through managed reports, runbooks, action manifests, and repair logs under `workspace/release/`.

## Research Intake Policy

Research ideas may be stored under `doc/idea/`. Promote them into active implementation only when they can be expressed as:

- A policy-compatible primitive.
- A benchmark or acceptance criterion.
- A bounded runtime feature.
- A managed artifact or release-gate signal.

Ideas that require GPU-first kernels, matrix training as the core mechanism, or global-gradient-backpropagation-dependent runtime learning must remain outside the policy-compatible SNN candidate. A roadmap cannot override the repository constraints. Bounded backward information and local credit propagation are policy-compatible when they satisfy the core rules above.

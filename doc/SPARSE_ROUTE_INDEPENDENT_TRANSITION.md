# Independent nonlearning scalar transition v1

🟢 [Done] Separately preregistered implementation computes fresh after-states from the existing 12 literal before/after templates × three roles. All 36 successful transitions match exact typed goldens: 1,296 changed and 14,364 unchanged fixture fields. Ten unit tests, 24 detected faults, exact fresh-process replay and 551 historical hashes pass. These are fixture computations, not actual learner updates or runtime engine equivalence.

## Preregistration and immutable scope

Selection SHA-256: `3c1a49e6e997d0a792fb2ce6cb9f43c2ebc5a23d4769331c6e74fb77c79cfb43`. It was written before implementation. Kernel, harness, runner, tests, path utility and CPython 3.10.22 were registered before the primary phase. The only parsed input is the existing `route_numeric_delta_preflight_selection_v1.json` (79,600 bytes); all twelve literal rows are hash-bound, unchanged and not regenerated from the implementation.

The public allocation and shuffled-source metadata are frozen at registration for ticks 0, 64 and 1023, seed 1201, candidate/permanent/shuffled_outcome roles. They use the existing five-input, width-one/two, three-output 45-route catalog, nine active credits and original two-half shuffle rule. This is offline metadata, not new sampled inputs or labels. All templates remain standalone examples, not complete feasible trained prefixes. The original scientific runs and thresholds are unchanged.

Limits: one saved input ≤262,144 bytes; 36 successful fixture transitions per phase; 36 changed /399 unchanged fields per success; 24 fixed faults; ≤262,144 bytes per artifact and ≤2 MiB aggregate. Each phase additionally contains 21 deliberately rejected transition attempts and three rejected after-state comparisons for those faults. Unit tests and proof verification reuse the same fixed templates; none are learning executions or task scoring.

## Independent computation and exact comparison

The kernel imports only `itertools.combinations` and `math.copysign/isfinite`. It does not import any production learner, old calculator, update helper, interface/ledger constructor or previous independent scalar kernel. Catalog construction, immutable state checking, public allocation checks and scalar arithmetic are implemented separately. Source import inspection and loaded-module checks enforce this scope; they are not security attestation.

`transition(owner, before, pending, delivery, diagnostic, consumed)` receives no expected after-state, golden error or externally supplied error increment. From the before masses it calculates the smoothed class probability, subtracts the credited binary side, squares the residual and multiplies by the public error weight. It increments only the credited mass, observation count, error sum and weight sum on each of nine routes. Every increment must yield an actual positive numeric change; rounded-away additions are refused. The returned immutable catalog and consumed set are fresh; original inputs and diagnostic state remain unchanged.

The harness constructs the expected after-state solely from the saved literal table, with a separately built catalog; it imports only the new transition function from the kernel. It compares all 435 addresses using exact types and finite float hex values, with no tolerance, then counts exactly 36 changed and 399 unchanged fields. Receiver/source IDs, owner and returned consumption state are also checked. The primary and fresh-process replay match complete canonical reports, not just totals.

## Identity, delay and faults

A pending receipt is an explicit trusted fixture input. Its receiving ID is the public tick; shuffled source ID is a distinct outcome address within the same phase, not a timestamp. Delivery must match both IDs and owner, have training purpose and arrive exactly one receiving tick later. Consumed receipts cannot be applied again. Fixed route/key/side credits are checked independently of the supplied pending tuple. Diagnostic state has a separate immutable namespace and is preserved.

All 24 fixed faults are detected: wrong receiving ID; wrong shuffled source; same-tick and late delivery; foreign owner; evaluation-purpose delivery; duplicate receipt; route/key/side drift; diagnostic alias; missing/duplicate field; bool/float mass; negative zero; NaN; absorbed increment; one-ULP after-error drift; inactive-field change; wrong role; source used as receiver; mutable fields; and nonfinite after-error. Rejected attempts leave original inputs unchanged. Tests also verify that the returned consumed ledger rejects a second delivery, fresh results do not alias before-state, APIs exclude expectations, and old helper/dynamic imports are rejected.

## Results, limitations and next priority

All 36 transitions pass. Result size is 17,864 bytes; the six artifacts total 102,097 bytes. They are saved under `data/processed/benchmark_fixtures/route_independent_transition_*_v1.json`: selection, registration, execution, result, replay_execution and replay. Result SHA-256: `75dc32baac9e0f8185adbe364ec7d618a8a55ef871ffa8211d23eda5e100e3b7`. Exclusive consumed markers refuse restarts before computation. Failures would be retained without retry, tolerance or condition adjustment.

**All twelve before templates have equal class masses and therefore probability 0.5.** This phase does not test asymmetric-count probability-to-transition integration. The earlier eleven-record stateless check includes asymmetric examples, but its success cannot be transferred to this new transition implementation. No extra row or revised golden was added after seeing this limitation.

Pending/delivery agreement does not authenticate actual runtime provenance: a mutually consistent fabricated pair can still be supplied. This phase does not execute a second local engine or establish same-role runtime equality. Counted changed fixture fields are not observed actual learning writes. Actual learner execution, learning updates, task prediction scores, acquisition and resource measurements remain zero; seals, original gates, old replay failure, recovery/retention negatives and causal uncertainty are preserved. Local scalar CPU execution meets project constraints but establishes no SNN-exclusive accuracy, energy or memory advantage, independent task usefulness, adaptive acceptance or promotion.

🟠 [Next] Preregister the smallest nonlearning asymmetric-count transition check using existing hand-record evidence, fixing literal after expectations and supported allocation cases before implementation. Preserve this 12-row result; do not retrofit or expand its protocol. Runtime provenance, same-role engine equality, independent task and SNN causal/resource work stay pending.

## 日本語

既存12組×3役割の36遷移を、既存ヘルパーを使わない独立実装で計算しました。変更1,296項目・無変更14,364項目が型と浮動小数の16進表現で完全一致し、10テスト・24故障・別プロセス再検証が成功、551件の既存ハッシュを維持しました。期待する更新後状態や誤差増分を実装に渡していません。受信ID・シャッフル元、固定1刻み遅延、重複配送、診断状態、丸めによる無変更を検証しています。ただし12組はすべて更新前確率0.5で、非対称カウントからの遷移は未検証です。次は既存の手書き証拠による最小の非対称遷移検証を別途事前登録します。実モデルの学習・採点・取得・計測はゼロで、旧失敗・負の結果・封印とSNN効果の未確定性を維持しています。

## 简体中文

使用不复用已有助手的独立实现计算了既有12组记录×3种角色的36次状态转移。1,296个变化字段和14,364个不变字段在类型及浮点十六进制表示上完全一致；10项测试、24项故障及独立进程复验通过，保留551项历史哈希。实现不接收期望后状态或期望误差增量，并验证接收ID与打乱来源、固定一拍延迟、重复投递、诊断隔离及舍入无变化。但12组前状态的概率均为0.5，尚未验证非对称计数的转移。下一步需另行预注册基于既有手写证据的最小非对称转移检查。未执行实际模型学习、任务评分、采集或测量；旧失败、负面结果、封存及SNN效果的不确定性均保持不变。

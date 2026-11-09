# Read-only capture logical-layout projection — 2026-11-09

🟢 [Done] Separately preregistered typed encoding validation and logical projection for the dormant capture candidate. Two hand predecessor/receipt pairs, 28 rejected faults, nine tests and exact fresh-process replay pass; 653 historical artifact hashes are preserved. Each phase projects to the existing 435 ordered learning addresses and 150 explicit route/key allocation flags. No candidate encoder, real model, host, learner, probability/error calculator or persistence adapter is executed.

Selection SHA-256: `3e033e493426068568660e010c0cca5b33d2da04afe8a998ab7889c613eb14ee`. Route catalog, hand recipes, two variants, invariants, fault roster and bounds were saved before implementation. Projection kernel/harness/runner/tests, unchanged candidate source, path utility and CPython 3.10.22 were registered before primary execution. Only the pinned candidate source and `route_capture_candidate_result_v1.json` are read as historical inputs. Six create-only artifacts use `route_capture_layout_*_v1.json`. Limits: 64 KiB historical inputs, 256 KiB encoded pair/artifact, 2 MiB aggregate artifacts, depth24 and 20,000 decoded nodes.

## Logical correspondence

The reader accepts immutable CapturePair/PhaseCapture/Input data records from the candidate format, not a model. It checks owner/epoch, separate even pre/receipt revisions differing by two, canonical encoded digest, bounded typed decoding and required logical fields. Tuples/lists, exact integers/Booleans and hexadecimal floats remain distinct; duplicate dictionary keys, unknown tags, invalid integer encodings and nonfinite floats reject. It does not call candidate._primitive or execute any model methods.

For each of 45 fixed one/two-input × three-output routes, enumerate binary keys lexicographically. Fifteen one-input routes have two keys and 30 two-input routes have four keys: 150 possible route/key allocations. Each contributes left/right mass fields (300), plus observations/error_sum/weight_sum for each route (135), totaling 435. Addresses match `(route, field_name, key_rank)` with -1 for route-level fields. The inactive route44 key(1,1) sentinel is retained at rank3 with literal masses7/9, preventing a collector from silently retaining only the active nine routes.

Missing table keys project to logical zero mass with allocation false. An allocated [0,0] key has the same two zero mass fields but allocation true. Two preregistered pairs differ only in the first active key's allocation and None versus .5 receipt probability; all 435 numeric fields are identical, while exactly one allocation flag differs. This is a representation distinction, not a new probability calculation or support claim.

Strict checks require nonnegative exact integer masses/observations, finite float errors/weights with 0≤error≤weight, complete fixed route/array layout and binary keys. This projection does **not** reuse the fixed-allocation transition validator or change its original thresholds/limits. It does not validate global mass/update conservation, true historical training support or after-learning increments. Separate downstream numerical contracts and original science gates still apply.

## Predecessor and receipt consistency

The predecessor must have no pending receipt. Both phases must have the admitted receiving step and unchanged update count. Receipt clock equals receiving ID and follows the pre clock. All logical learning fields and allocation flags must be exactly unchanged across begin; typed JSON comparison preserves signed-zero changes instead of Python float equality hiding them. Scheduler/pool/clock/pending changes outside the projected fields are not thereby forbidden or certified; they remain in the hash-bound raw state.

Receipt pending must be the five-part Reactivation shape: own tick, exact input events, nine distinct balanced route IDs, nine recorded probabilities and three recorded prediction slots. Derive only route/key addresses from the input events. Unallocated active key requires probability None; allocated key requires a finite float in [0,1]. Recorded probabilities are copied, never recomputed from counts. A test deliberately changes a valid allocated probability to .9 and shows that it is retained; this documents that arithmetic correctness/provenance is not being inferred from schema validity. The source/shuffle address stays in Input; it is not substituted as the receiving clock.

Full raw dictionaries are digest-bound, but only the registered logical subset is projected. Extra fields are retained in the raw encoding and are not authenticated or interpreted by this checker. A missing required logical field rejects; this does not prove that an arbitrary dictionary contains every field of a genuine FactorialMemory instance. Independent traversals of a live model, all-writer confinement and real owner authority remain untested.

## Hand scope, faults and validation

The two recipes build detached plain dictionaries and typed encodings independently of the candidate encoder. They are not model initializations, saved actual training trajectories or physically conserved learner histories. In particular lifecycle and count fields are hand inputs for schema checks; no mass/observation history is inferred from them. Existing result bits, tasks, thresholds, training counts and sealed periods are not used or changed.

Twenty-eight faults reject: digest/owner/epoch/revision mismatch; preexisting pending; wrong tick/events; partial/duplicate/unbalanced active routes; absent/allocated probability mismatch; Boolean/negative mass; Boolean observations; nonfinite weight; error above weight; missing field; wrong catalog; nonbinary key; duplicate encoded key; changed learning value/allocation/step/update; noncanonical integer; unknown tag; and byte limit. Nine tests also cover exact absent/zero distinction, inactive width-two addressing, Boolean route keys, signed-zero preservation, decoder bounds, readonly repeat and frozen input drift. The gate remains false and no real model module is imported by the tests.

Primary and separate-process replay agree; registered source and 653 historical hashes verify. Consumed primary/replay refuse before computation. The four selected roadmap gates still pass. All seven runtime/task/SNN evidence categories remain missing. No old failed attempt is resumed; original reproduction failure, recovery/retention/resource negatives, support/current-state/outcome conditions and test/source seals are unchanged. Generic schema projection is applicable to scalar systems and supplies no SNN-specific advantage.

🟠 [Next] Preregister a capture-generated acknowledgement bridge using only validated encoded pairs and model-free active-pause bindings. Derive digests, revisions and ordered route IDs from the validated pair rather than expected hand acknowledgement labels; require current receipt pause identity and distinguish binding validity from genuine numeric capture authority. No real model, producer, calculator, storage operation or runtime launch. Review the previous fixed-plan composition compatibility instead of simply relabeling hand proofs.

⭕️ [Pending] Real candidate encoding/capture, complete instance schema and independent live readback, external owner and input/outcome authority, full writer/alias confinement, independent scalar after-states, durable nine-route state/consumption, independent task and SNN causal/resource evidence. This projection does not unblock the dormant candidate.

## 日本語

起動禁止の採取候補が出す形式を、435学習フィールド・150割り当て欄へ写像する読み取り専用検証器を事前登録・実装しました。未割り当てと割り当て済みゼロ件数を区別し、非活性経路のキーも保持します。手作業2組、故障28件、9テスト、別プロセス再検証が成功し、653件の既存証跡を維持しました。受信時確率は再計算せず、実履歴・数値の正しさや権限は立証しません。候補エンコーダー・モデル・学習器・計算器・モデル保存処理は実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

预注册并实现将禁止启动的采集候选格式映射到435学习字段和150分配标志的只读验证器。区分未分配与已分配零计数，并保留非活跃路径的键。2组手工记录、28项故障、9项测试和独立进程复验通过，保留653项历史证据。不重算接收概率，不证明真实历史、数值正确性或权限。不执行候选编码器、模型、学习器、计算器或模型存储，旧失败、负面结果与封存保持不变。

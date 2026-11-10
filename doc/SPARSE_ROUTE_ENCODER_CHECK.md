# Model-free original encoder verification — 2026-11-11

🟢 [Done] Separately preregistered pure-format encoder execution on the two existing hand-layout recipes, without launching either capture candidate. Eight baseline helper calls per phase (two recipes × predecessor/receipt × original/reversed dictionary order) match the independent hand encoder exactly. Four logical pair projections pass; six malformed/bound cases reject. Seven unit tests and exact fresh-process replay pass, preserving 680 historical artifact hashes.

Selection SHA-256: `68ddc0f9f0510688c67e827a86f37fbf50daea8c6918b6d7f4d84c0e8edd8029`. Existing recipes, helper allowlist, order variants, call roster, independent oracle, bounds/stops and prohibited model boundaries were fixed before implementation. Harness/runner/tests, seven unchanged dependencies, path utility and CPython 3.10.22 were registered before primary execution. Only `route_capture_candidate_v2_result_v1.json` is parsed as historical input; pinned sources are hash-read only. Six create-only artifacts use `route_encoder_check_*_v1.json`.

## Narrow executable scope

This is a new model-free protocol after the source-only V2 study, not a restart of that study or an activation of its launch gate. It permits only the existing `_primitive` and `_encoded` format helpers on ordinary hand data, plus independent encoding/decoding/layout checks. V1 and V2 `LAUNCH_AUTHORIZED` remain false. Candidate constructors/capture methods and host creation are replaced with failure traps during verification and are never called. No learner class is imported, no model or host is created, and no prediction/error calculation, begin/observe, real snapshot traversal or model storage executes.

The old V2 report's encoder-execution count remains zero and is unchanged. Here the original pure encoder actually executes: eight baseline calls plus one negative byte-limit call per primary/replay/verification computation. Unit tests make their own small format-helper calls; phase counters are not presented as total-session call accounting. These are format operations, not local learning updates, predictions, score evaluations or SNN events.

## Independent comparison

Reuse the exact absent-first-key and allocated-zero-first-key recipes already registered by the layout study. No new training input, task, source, model seed or result label is selected. For each phase, compare original helper output and JSON bytes to the independent `hand_encoding` implementation. Recursively reversing dictionary insertion order must leave output/digests unchanged. Decode each result and re-encode with the independent reference to verify exact type/value preservation. Check the supplied plain input remains unchanged.

The two encoded phase records are then passed through the unchanged layout validator. All four variant/order pairs retain 435 learning fields, 150 explicit allocation flags and correct pending structure; absence versus allocated-zero remains distinct. Additional unit-level primitive type checks confirm None, Boolean false, integer zero, positive/negative floating zero and empty list/tuple tags are not conflated. The shared alias test verifies detached output while allowing repeated references that are not cycles. Matching a hand oracle does not validate a physically conserved learner history or the completeness of an actual model instance.

## Bounds and malformed inputs

A new verification-only wrapper accepts exact primitive/container types and checks recursion depth24, 20,000 nodes, cycles and finite floats before calling the unchanged encoder. Unsupported objects are rejected by type without invoking their iteration/repr hooks. Repeated references are counted again and allowed when not on the active recursion path. The encoded byte cap is 256 KiB after encoding. Historical input cap is 128 KiB, each artifact256 KiB and aggregate2 MiB.

Six registered failures: unsupported object, nonfinite float, cycle, excessive depth, excessive nodes and byte limit. The first five fail before the original helper call; the byte case executes the helper then refuses its oversized output. The wrapper is not integrated into either frozen candidate. Depth/node checks and an after-encoding byte cap do not constitute a bound on all temporary allocation or an untrusted-input service guarantee. String/integer payload size and generic serialization overhead are not resource measurements here. No CPU, energy or memory-performance advantage is inferred.

## Verification and remaining gaps

Seven tests cover registered comparison under traps/no new runtime imports; literal type tags; dictionary order; detached repeated aliases; hostile object refusal; cycle/depth refusal before encoder call; and frozen source/report/budget refusal. Primary/fresh-process replay agree exactly. Source pins and 680 preserved artifact hashes verify; consumed execute/replay refuse before computation. The four selected roadmap gate checks still pass.

This closes only original pure-format-helper equality on the existing hand examples. It does not close any of the seven missing runtime/task/SNN evidence categories. No actual model `vars` traversal, all-writer pause, live independent readback, authenticated owner/source, genuine outcome unavailability, scalar after-state execution or durable publication has been observed. The V2 body/bridge remains unexecuted. Original reproduction failure, earlier unit assertion failure/archived bytes, negative recovery/retention/resource outcomes, conditions and test/source seals are preserved.

⭕️ [Pending] Any model-bearing construction/begin/capture still needs a separate source-pinned bounded protocol and genuine owner/source eligibility, failure/cleanup and outcome-order grounds. Format-helper success cannot enable either launch flag, authenticate a hand record, open sealed periods or authorize old-attempt replay. Independent task, SNN-specific causal evidence, durable nine-route commit/consumption and resource work remain held. Generic serialization equality is equally useful to scalar implementations and supplies no SNN-exclusive benefit.

## 日本語

モデルを使わない別プロトコルを事前登録し、既存の手作業2組に限って元の形式変換ヘルパーを実行しました。辞書順を変えた8基準呼び出しが独立期待形式と完全一致し、4組の435欄・150フラグの写像、6種類の拒否、7テスト、別プロセス再検証が成功しました。680件の既存証跡を維持しています。過去のV2静的検証は再開せず、そのエンコーダー実行ゼロの記録も変更していません。候補の起動フラグはfalseのままで、モデル・ホスト・学習器・実採取・モデル保存処理は実行せず、旧失敗・負の結果・封印を維持しました。

## 简体中文

单独预注册无模型协议，仅在现有2组手工数据上执行原格式转换辅助函数。改变字典顺序后的8次基准调用与独立参考完全一致，4组435字段/150标志投影、6项拒绝、7项测试及独立进程复验通过，保留680项历史证据。未重启旧V2静态验证，也未改写其编码器执行为零的记录。两个候选的启动标志仍为false，不执行模型、主机、学习器、真实采集或模型存储，旧失败、负面结果与封存保持不变。

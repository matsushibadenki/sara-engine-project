# Saved-report serialization and equality preflight — 2026-10-26

🟢 [Done] Preregistered and implemented a strict codec preflight using only the saved [reactivation report and replay failure](SPARSE_ROUTE_REACTIVATION.md). All 19 registered checks pass, with 35 passing tests and an exact fresh-process **static preflight** replay. No learner is imported or executed. The original full learner replay remains failed, numerical trace equality remains unknown, and the original component result remains negative.

## Registered inputs and bounds

The [selection](../data/processed/benchmark_fixtures/reactivation_codec_selection_v1.json) was written at **12:59:21 JST**, before implementation. [Code/tests registration](../data/processed/benchmark_fixtures/reactivation_codec_registration_v1.json) at **13:02:23 JST** preceded the retained static execution; its separate-process replay followed at **13:02:24 JST**. This is a retrospective engineering check on known report shapes, not a new predictive experiment or independent replication.

| Parsed input | Bytes | Scope |
| --- | ---: | --- |
| Saved primary report | 464,131 | Existing numeric values, field types and layout only |
| Saved replay-failure receipt | 1,177 | Failed replay and unknown numeric equality remain unchanged |
| Total | **465,308** | Under the registered 1 MiB aggregate input limit |

Each input is limited to 2 MiB, accepted JSON trees to depth 32 and 100,000 nodes, each new artifact to 128 KiB, and all new artifacts to 1 MiB. Six new artifacts total **66,786 bytes**. Verification preserves **412 file hashes**: the inherited 399 artifacts, six completed/failed reactivation artifacts, six registered source/test files and the post-result failure-regression test. Other preserved files are read only for hashes, not parsed as new evidence. No dataset/model contents or sealed periods are semantically read.

## Comparison contract

The [pure codec](../src/sara_engine/evaluation/reactivation_codec.py) derives an exact typed-layout template from the pinned saved report, then checks both supplied values against it. The template is copied to prevent later caller mutation. This is a deliberately version-specific contract; it does not silently accept arbitrary future schemas or variable-length arrays.

- Only `/cases/{index}/arms/{arm}/initial_pool` may interchange tuple/list when the saved field is a non-null list. Reconstructing those **80** tuple shapes from saved lists eliminates their Python representation mismatch. Round-robin's 20 null pools remain null; tuples elsewhere fail. Reconstruction is explicitly labeled saved-metadata reconstruction, never a runtime report.
- Object-key order is ignored. Array order and lengths, all object fields and primitive types remain exact. Missing/extra fields, bool/int interchange and int/float interchange fail rather than being discarded or coerced.
- Finite floats compare by their canonical JSON bytes, with no tolerance or rounding. A one-ULP float change is detected; positive and negative floating zero remain distinct. Signed 64-bit integers remain integers. No Unicode normalization, string conversion or metric recomputation occurs.
- Parsing rejects duplicate keys (also when nested), nonfinite constants, overflow-to-infinity floats, invalid UTF-8 and out-of-range integers. Node/depth/byte bounds reject oversized inputs. Encoding rejects non-string keys and non-JSON objects. Limits bound accepted representations; they are not a measurement or proof of parser peak memory.

Canonical bytes use sorted string keys, preserved array order and strict JSON primitives. The complete supplied representations are compared, rather than only their digests or selected successful metrics. The saved report's in-memory canonical form is **223,520 bytes**, SHA-256 `fa0436a4d757f56b5be32c19d05ae79a07a69e626280a5cd1bcc5e10391e2dd2`. That compact representation is not a new report observation or a saved model, and no storage/resource advantage is inferred from its size.

The [standalone runner](../scripts/eval/reactivation_codec.py) exposes only registration, static execution and static replay. It verifies fixed input/source hashes before and after the check, uses exclusive phase files and rejects any success receipt appearing in the old failed learner-replay namespace. The frozen old runner/source and consumed markers are untouched. No subprocess, learner, data acquisition, resource probe or scoring path is available through the new implementation.

## Results and limits

The 19 preregistered checks cover saved round-trip, whitelisted tuple reconstruction, object-key order, one-ULP drift, boolean/integer/float distinction, array order, missing/extra fields, forbidden tuples, NaN/infinity, duplicate keys, non-string keys, large integers, signed zero and input/node/depth limits. Counterexamples alter only saved-report metadata or malformed JSON fragments; they do not generate predictions or re-score data.

**35 tests pass under Python 3.10.22.** Tests additionally exercise nested duplicate keys, invalid numeric/UTF-8 payloads, hashes/pool changes, template isolation, forbidden tuple locations and consumed-attempt refusal. An import check confirms that the learner module is absent from the test process. Tests run only this codec and managed-path checks; they do not run the earlier learner-mechanics suite. The static result is reproduced exactly in a fresh process.

| Retained field | Result |
| --- | --- |
| `codec_preflight_passed` | `true` |
| `learner_executions`, new scored predictions / measurements / acquisitions | `0` |
| `original_component_passed` | `false` |
| `full_replay_verified` | `false` |
| `numeric_trace_equality` | `null` / unknown |
| Study restart, condition relaxation, seal opening, promotion | Not authorized |

A passing codec proves that the known tuple/list shape difference can be handled without suppressing the registered counterexamples. **It cannot show that this was the only difference in the discarded failed learner replay.** Comparing a reconstructed report with its own saved source is a representation test, not independent numerical replication. The 256-tick recovery failures, exact-dependency-retention failures and other original negatives remain unchanged.

| Artifact | SHA-256 |
| --- | --- |
| Selection | `a634fecb6c348f4ca0dfecc4fbdf54efb00d0208fbc22ea0a936a0d8f3f4e255` |
| Code registration | `3fbc9ea9cd03a0603bcde319ed22025fea0ce8ed4c06094306ec16f0c021afd9` |
| [Static result](../data/processed/benchmark_fixtures/reactivation_codec_result_v1.json) | `c21a53edc09bf27a0675a0ef11f55fd2cdc2397c32c218caa007f3f4aa1162bd` |
| [Static replay receipt](../data/processed/benchmark_fixtures/reactivation_codec_replay_v1.json) | `ff8c01b0958114b3405768744be7fd2f096327bc4e6a4fd49f5bf971b3802c7a` |

```sh
# These commands never execute the old learner or its consumed replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python scripts/eval/reactivation_codec.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. workspace/.venv310/bin/python -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_reactivation_codec.py tests/test_project_paths.py
```

⭕️ [Pending] Any future numerical replication needs a separately justified/versioned protocol and explicit authorization; this preflight does not reopen the consumed run. No threshold/seed/window adjustment, automatic learner retry or further codec-audit loop is queued. Nonstationary promotion, independent-task qualification, branch-local causal efficacy, resource/energy work and production retain their existing holds. This work supplies a trustworthy comparison primitive, not an SNN capability or energy result.

日本語: 保存済み報告と失敗証跡だけを使う事前検証を実装前に登録し、19チェック・35テスト・別プロセスでの静的結果の完全再現が成功しました。tuple/listの同一視を既知の80か所に限定し、数値の最小単位の差、型差、配列順序、欠落・追加を検出します。412ファイルのハッシュ、既存の負の結果・再現失敗・条件・封印を保持しました。学習器の実行・採点・取得・計測はゼロであり、元の完全な数値一致は未確認のままです。

简体中文：在实现前注册仅使用已保存报告和失败证据的序列化与等价比较预检。19项检查、35项测试及独立进程的静态结果完全重放通过。tuple/list等价仅限已知80处，检测最小浮点差、类型差、数组顺序及缺失/新增字段。保留412个文件哈希、负结果、重放失败、原条件和封存。学习器执行、评分、获取、测量均为零，原完整数值一致性仍未知。

🟢 [Done] Separately authorized follow-up: [fixed-condition numerical reproduction v2](SPARSE_ROUTE_NUMERIC_V2.md) completes new primary/fresh-process replication using this unchanged codec. Numerical values match while all original scientific negatives and the discarded old replay failure remain unchanged. No old phase was resumed.

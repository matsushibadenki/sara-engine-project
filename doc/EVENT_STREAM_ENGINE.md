# Generic Bounded Event-Stream Engine

## Milestone — 2026-09-16

[Done] `BoundedEventStreamEngine` joins the bounded event-route encoder and `BoundedNormalizedHybrid` into one serializable runtime. It owns activity/lifecycle/gap/prefix route identity, optional calendar routes, transition contexts, local weights, compact/explicit/scalar backend configuration and prediction sequence.

The engine rejects serialization while an outcome receipt is pending. State loading revalidates schemas, sorted bounds, route order, route/context/weight caps, labels and numeric limits. A model artifact contains both the encoder mapping and learner state, preventing route-ID drift after restart. Atomic files are restricted to `models/event_stream_engine/`.

[Done] Both accepted real workloads were replayed with a real checkpoint boundary immediately before the frozen partition. The BPI pre-test checkpoint contains 1,018 routes, 5,181 weights and 107 transition contexts; after its private online frozen replay it contains 1,024/5,260/107 and exactly reproduces 32,694 accepted predictions. The Sepsis pre-test checkpoint contains 995 routes, 3,820 weights and 102 contexts; after replay it contains 1,021/3,997/105 and reproduces 2,035 accepted predictions. Checkpoint and prediction digests are recorded in `workspace/evaluation/event_stream_engine_replay_v1.json` and the formal package manifests.

[Done] Future real event-stream experiments should use this engine instead of duplicating dataset-specific route and learner implementations. Immutable historical evaluators remain unchanged as evidence.

[Done] Added artifact envelope `sara-bounded-event-stream-artifact-v1`. It binds the canonical engine payload to SHA-256, rejects corruption and unknown schemas, and retains an explicit legacy loader for the raw v1 engine state. Pending predictions remain non-serializable.

[Done] Added generation-checked publication under an owner-only advisory lock. Four equal-generation processes admit exactly one writer and advance generation once. Artifacts and locks are owner-only, files are capped at 32 MB, and file plus parent directory are synced around atomic replacement. The durable gate also verifies checksum, permission and oversize rejection.

[Done] Materialized generation-1 research checkpoints at the pre-test boundary for BPI and Sepsis under `models/event_stream_engine/`. `research-checkpoints-v1-manifest.json` pins the whole-artifact SHA-256, canonical payload SHA-256, accepted-result SHA-256, accepted prediction trace and runtime configuration digest. Publication refuses an existing generation instead of silently replacing research evidence.

[Done] A separate load-only verification process checked every pinned identity before opening the frozen partitions. It reproduced the accepted BPI trace across 32,694 predictions and the accepted Sepsis trace across 2,035 predictions. Frozen replay updates only the private in-memory copy and leaves the pre-test artifact unchanged. The report is `workspace/evaluation/research_checkpoint_release_v1.json`; `production_authorized` remains `false`.

Build the research-only artifacts once, then verify them without refitting:

```bash
python3 scripts/eval/research_checkpoint_release.py build
python3 scripts/eval/research_checkpoint_release.py verify
```

The build command is intentionally fail-closed when the generation-1 files or release manifest already exist. Delete or replace published research evidence only through a separately reviewed release procedure.

### Formal dataset packages

[Done] The research release is also materialized as two self-contained dataset packages:

```text
models/
├── bpi2012/
│   ├── sara-bpi2012-pretest-v1.sara
│   ├── SHA256SUMS
│   └── manifest.json
└── sepsis/
    ├── sara-sepsis-pretest-v1.sara
    ├── SHA256SUMS
    └── manifest.json
```

Each manifest binds the algorithm configuration, processed source manifest, source digest, chronological split counts, pre-test checkpoint identity and resource counts, accepted result, evaluation procedure, prediction digest, and known negative results. `SHA256SUMS` independently binds the `.sara` artifact and manifest. Verification checks those identities before loading, performs no checkpoint refit, updates only a private in-memory replay state, and confirms that the artifact hash remains unchanged afterward.

```bash
python3 scripts/eval/research_checkpoint_release.py package
python3 scripts/eval/research_checkpoint_release.py verify-package
python3 scripts/eval/research_checkpoint_release.py verify-package --dataset bpi2012
python3 scripts/eval/research_checkpoint_release.py verify-package --dataset sepsis
```

The formal package verification passes all thirteen identity/replay checks for each dataset. Its machine-readable report is `workspace/evaluation/research_package_verification_v1.json`.

日本語: event符号化表と局所学習状態を一つのartifactへ統合し、BPIとSepsisで学習後checkpointを復元して受理済み予測列を完全再現しました。今後の実イベント研究はこのengineを共通基盤にします。

简体中文：事件编码表与局部学习状态现已合并为一个模型工件，并在BPI和Sepsis上从训练后检查点恢复后完整复现已接受的预测序列。后续真实事件流研究将使用该通用引擎。

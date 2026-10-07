# Temporal capability boundary v1

## English

[Done] Completed the separate preregistration, baseline identity check, bounded reference implementation and exhaustive input-only audit. Protocol SHA-256: `7574d2d1ea3ddec03af52229b7db247d53728e10bca425ea518a10017eb2c789`.

The saved shared-observer protocol, immutable result receipt and final research checkpoint are hash-pinned without modification or retraining. This run verifies their identities; it does not repeat the old checkpoint prediction replay.

All 276 unordered pairs of distinct routes are checked in each of three families. Opposite modes share histograms and total duration. Middle-order modes also share every first/last/count record; gap modes share the entire ordinal sequence. Observers receive only routes and nonnegative relative intervals, never a mode label or outcome.

| Reference | First-order distinct pairs | Middle-order distinct pairs | Gap distinct pairs |
| --- | ---: | ---: | ---: |
| Histogram | 0/276 | 0/276 | 0/276 |
| First/last/count | 276/276 | 0/276 | 0/276 |
| Full ordinal sequence | 276/276 | 276/276 | 0/276 |
| Full sequence and gaps | 276/276 | 276/276 | 276/276 |

The registered audit passes all 828 pairs. This demonstrates representation information boundaries, not learned accuracy. These are newly registered synthetic templates, not independent real-event confirmation. No spike advantage, physical-energy benefit or production promotion follows.

The non-spiking sequence reference admits at most 95 events over 24 routes. Gaps are integers from 1 to 255 after an initial zero. Validation rejects invalid input and overflow before mutation. Snapshots are immutable. The maximum wire format is 194 bytes; the audited short templates use at most 16 bytes. Wire size excludes Python object overhead. Full-order retention costs one stored route per event; interval retention adds one byte per event. Every future learner must account for its own observation, projection, live Python state and readout costs. Both references visit each arriving event once; this audit visits 7,728 events overall and does not measure CPU or energy advantages.

Run once with `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 scripts/eval/temporal_capability_boundary.py`. The runner checks protocol/baseline hashes before executing and exclusively creates `workspace/evaluation/temporal_capability_boundary_v1.json`; an existing report is never overwritten. Tests may repeat the pure input audit. Related regression suite: 36 tests passed on Python 3.10.18.

[Next] Freeze a separate matched learning protocol before scoring: readout-visible middle-order/gap encoding, delayed local feedback, fresh training/evaluation identities, negative controls and complete resource accounting. The retained full sequence is a strong non-spiking information reference, not yet a predictive readout.

[Later] Independent real-event confirmation, deep credit and production review. Completed shared-observer and electricity hypotheses remain closed.

## 日本語

[Done] 最小研究構成の保存物をハッシュ固定し、別の事前登録・有界参照実装・828組の入力監査まで完了しました。最初／最後／回数では途中の順序を失い、全順序だけでは時間間隔を失う境界を確認しました。最大95イベント、通信形式194byte。Pythonオブジェクトの実メモリはこのbyte数に含みません。関連36テストが通過しました。

[Next] 学習前に、readoutが実際に参照できる情報、遅延局所更新、新しい学習／評価分割、非スパイク対照、全状態／処理予算を固定します。今回の成功は情報識別の監査であり、学習精度や実データでの効果ではありません。

[Later] 実イベントでの独立確認・深いcredit・本番審査。完了した電力実験は再開しません。

## 简体中文

[Done] 已固定最小研究组件的保存文件哈希，完成单独预注册、有界参考实现和828组输入审计。首次／末次／计数会丢失中间顺序，完整顺序仍会丢失时间间隔。最多95个事件，最大线格式194byte，不包含Python对象内存。相关36项测试通过。

[Next] 在评分前固定readout可见信息、延迟局部反馈、新训练／评估身份、非脉冲对照和完整状态／计算预算。本次通过仅表示信息可区分，不代表学习准确率或真实数据收益。

[Later] 独立真实事件确认、深层credit和生产审核。保持已完成的电力实验关闭。

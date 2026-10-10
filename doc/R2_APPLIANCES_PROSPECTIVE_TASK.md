# Prospective appliance-demand task: source and support gate

🟢 [Done] Fresh source acquisition, frozen outcome/split contract, causal extraction,
sufficient non-spiking input controls and train/development support audit.
🟢 [Done] Registered and executed the [delayed learner comparison](R2_APPLIANCES_DELAYED_COMPARISON.md); actual updates and replay pass, SNN increment fails. The task is consumed; do not retune it.
⭕️ [Pending] Independently useful SNN effect and physical energy measurements.

## Source and frozen task

The official [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
contains measured appliance energy at ten-minute intervals. Attribution:
Candanedo, L. (2017), DOI [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G),
CC BY 4.0. The raw CSV SHA-256 is
`2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`.
The reviewed protocol was written and hash-pinned before numeric inspection:
`2ccb958df09bc6de92744e6a5fbc7c23454b428b2bee951f7cf4436411418caa`.

At every admitted ten-minute cutoff, predict whether the measurement exactly
30 minutes later is at least 20 Wh above the current measurement. Inputs are
36 chronological appliance measurements ending at the cutoff (350 minutes
between first and last timestamps). This is an observed future measurement,
not a terminal-row classification or an inferred absence of an outcome.
All intermediate future timestamps must exist. Incomplete follow-up abstains.
Training starts January 11, 2016 and ends before March 1; development is March;
April 1 onward remains sealed. Entire histories and outcome horizons must fit
within their respective split. No history crosses a split boundary.

Only historical `Appliances` values are input features. Future values are
evaluator-only labels. Date is read before numeric cells, and sealed numeric
cells are never indexed. Source byte hashing reads bytes, without decoding
sealed numeric values. Airport weather includes interpolation in the source
description; all weather and random-variable columns are excluded.

## Retained result

| Split | Positive | Negative | Admitted days |
| --- | ---: | ---: | ---: |
| Training | 1,665 | 5,395 | 50 |
| Development | 1,021 | 3,405 | 31 |

The fixed support gate requires 500 examples per training class, 100 per
development class and at least 30/14 admitted days, respectively. It passes.
11,562 numeric observations were read; 8,173 sealed rows were inspected only
for timestamps. The compact receipt retains source, protocol and causal-example
hashes. No model was trained, no learner metric was computed, and no sealed
label or score was exposed. Twenty related tests pass, including sealed-cell
spy access, changed-future/invariant-input checks, split purging, incomplete
follow-up, invalid values and exact reconstruction from change positions.

## Next comparison contract

Execution update: the following requirements were frozen and implemented in
the [comparison protocol and negative result](R2_APPLIANCES_DELAYED_COMPARISON.md).
This section records the original forward contract, not authorization to rerun.

Before learner scoring, freeze candidate routes, fresh seeds, probability
mapping, delayed-feedback handling, actual local-update budgets, quality and
resource thresholds, and a development-only decision rule. Include independently
deployable current-value, histogram, full-ordinal and transition-gap controls.
The supplied change positions retain enough information to reconstruct all
36 samples, so an apparent gain from shared input information cannot be
attributed to spikes. Charge history storage, projection and update work to
every arm; release the 30-minute outcome only after prediction, and prohibit
development updates unless explicitly preregistered as a separate arm.

Examples overlap and are serially correlated. Aggregate uncertainty by day,
not by treating 11,486 examples as independent trials. This is a single-house,
regularly sampled stream, not evidence for general irregular-event timing or
cross-house generalization. Change-event gaps are derived from observed value
transitions. Timestamps have no verified timezone; do not claim physical elapsed
time or DST correctness. The class-support gate is a screening gate, not a
demonstration that temporal information improves predictive accuracy.

## Reproduction

Run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`:

1. `python3 scripts/eval/acquire_appliances_source.py` downloads the official
   archive, verifies the pinned CSV checksum and writes only to `data/raw/`.
   It refuses to overwrite an existing source.
2. `python3 scripts/eval/appliances_prospective_audit.py` verifies protocol/source
   identities and writes the compact receipt under `data/processed/`. It refuses
   to overwrite an existing receipt. Add `--verify` to replay extraction and
   compare every retained field without replacing the evidence.
3. `python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_appliances_prospective.py tests/test_sealed_temporal_csv.py tests/test_prospective_event_contract.py`.

日本語: 最優先の新規実データ課題について、30分後の実測値を予測する契約と漏洩防止抽出を実装しました。学習・開発の件数条件は通過し、テスト期間の数値は封印しています。性能比較は未実施で、次に局所更新回数・遅延結果・十分な非スパイク比較条件を事前登録します。

简体中文: 已为优先的新真实数据任务实现预测30分钟后实测值的固定协议和防泄漏提取。训练及开发集通过样本支持门槛，测试期数值仍封存。尚未进行模型评分；下一步预注册局部更新次数、延迟反馈和充分的非脉冲对照条件。

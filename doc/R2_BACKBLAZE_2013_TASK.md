# R2 Backblaze 2013 Drive-Failure Task

## Decision

Backblaze Drive Stats 2013 is the next real delayed-outcome source for the Local Credit Packet transfer study. It contains daily drive observations and a terminal failure flag across many independently identified drives. This selection does not authorize feature design, candidate scoring, or held-out SMART access.

## Frozen Source

- Official archive: `data/raw/r2_backblaze_2013/data_2013.zip`
- SHA-256: `7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe`
- Source: Backblaze Drive Stats
- Required attribution and use conditions are recorded in the V1 protocol.

The source audit reads only `date`, anonymous `serial_number`, `model`, `capacity_bytes`, and `failure`. It does not read SMART values, materialize held-out features, train a learner, or score a candidate.

## Audit Results

V1 is retained as a negative audit (`388d074c…45e8`). Its quantitative support gates passed, but a stronger assumed invariant failed: 180 observations occur after a drive's failure row.

V2 froze the adjudication rule before rerunning the audit (`protocol 26af53b5…f7f4`, `result 0c02ce96…e2eb`): the first `failure=1` row is the terminal event and every later row for that serial is recorded and excluded from all future features, labels, and controls. V2 passes:

- 266 daily files and 5,091,501 rows
- 29,072 distinct drives and 40 models
- 724 failed drives; 597 have at least 30 observations through the failure day
- failures by chronological split: 400 training, 193 development, 131 held-out
- one failure row per failed drive, no duplicate drive/day rows, and no filename/date mismatches

## Research Boundary

This source is suitable for a drive-level feasibility study, not yet for a performance claim. The next protocol must freeze case construction before SMART values are opened. It must:

- use the first failure day as the terminal cutoff and exclude all later observations;
- forbid serial number, calendar identity, and model identity as predictive shortcuts;
- keep failed-drive and nonfailed-control identities disjoint across train, development, and held-out partitions;
- define censoring and matched controls without using held-out outcomes to tune windows;
- require at least 30 prior daily observations for eligible failure cases;
- compare identical routing, memory, budget, and learning signals across minimal event-unit controls;
- keep the held-out SMART feature rows sealed until development gates pass.

The archive provides observational fleet data. Drive removal is not equivalent to failure, model and calendar distributions may shift, and class imbalance is substantial. These limitations must remain explicit in every later result.

## Case Manifest and Development Result

The frozen drive-level manifest gate passed before SMART access. Exact-model, one-to-one deterministic matching produced 249/249 training, 33/33 development, and 130/130 sealed held-out cases. All 824 case IDs are unique; each has at least 30 observations through its terminal date. Serial, case, model, capacity, and calendar identity are prohibited candidate features.

Only the 498 training and 66 development cases were then materialized. Six preregistered SMART raw fields were converted into bounded onset, persistence, and positive-delta events over the final 30 observed rows. The output contains no held-out case or held-out SMART row.

The one-shot development comparison is negative (`14b83f87…20c9`):

| Arm | Balanced accuracy |
| --- | ---: |
| Scalar local learner | 0.5000 |
| Compact event unit | 0.5000 |
| Temporal state | 0.5000 |
| Dendritic temporal unit | 0.4545 |

The branch-shuffled dendritic control reached 0.4394, while the outcome-shuffled control reached 0.5455. The minimum quality, gain-over-compact, and targeted-control-drop gates failed. Scalar and compact predictions were exactly equivalent. The sealed 260-case held-out set was not opened and must not be used for this consumed hypothesis.

A non-scoring post-hoc coverage report shows no empty episode, but event counts frequently approach the 96-event ceiling and several routes are nearly universal. This suggests weak selectivity or saturation as a future training-only diagnostic question; it does not rescue the failed gate and does not authorize threshold or encoding retuning on this development identity.

## Training-only Sparse Selectivity Diagnostic

A separate protocol was frozen before reopening the 498 training cases. Development and held-out cases were forbidden. Five deterministic folds compared four Compact Event encodings without adding temporal, spiking, refractory, or dendritic mechanisms.

| Encoding | Five-fold balanced accuracy | Median events | Event-cap fraction | Dominant-route fraction |
| --- | ---: | ---: | ---: | ---: |
| V1 presence/onset/increase | 0.5622 | 67.5 | 0.2912 | 1.0000 |
| Transition only | 0.5542 | 2 | 0 | 1.0000 |
| Final log snapshot | 0.6044 | 6 | 0 | 1.0000 |
| Sparse signed log delta | 0.7410 | 31 | 0 | 0.9558 |

The sparse signed-log-delta encoding exceeded 0.696 balanced accuracy in every fold and improved substantially over V1. Nevertheless, the registered diagnostic is negative (`794ef9f4…9464`): its dominant-route fraction is 0.9558, above the frozen 0.95 ceiling. The ceiling is not relaxed after inspection. This result supports **event selectivity before neuron complexity** as the next hypothesis, but does not authorize candidate promotion or reuse of the consumed development set.

## 日本語要約

Backblaze 2013 は、独立した多数のドライブと遅延故障結果を持つため候補に採用しました。824件の均衡manifestとtrain/devイベント化は通過しましたが、4系統の開発比較は陰性でした。training-only診断では符号付きlog差分が0.741まで改善しましたが、支配route比率の凍結上限を超えたため総合判定は陰性です。developmentの再利用とheldout 260件の開封は行っていません。

## 简体中文摘要

Backblaze 2013 含有大量独立硬盘和延迟故障结果，因此被选为候选数据源。824个平衡案例的清单与训练/开发事件化通过，但四类单元的开发比较为负面结果。仅训练诊断中，稀疏有符号对数差分达到0.741，但主导路由比例超过冻结上限，因此总门控仍为负面。未重复使用开发集，也未开放260个保留案例。

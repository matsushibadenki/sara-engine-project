# Prospective UCI Incident Closure Task

## Source and attribution

The [UCI Incident management process enriched event log](https://archive.ics.uci.edu/dataset/498/incident%2Bmanagement%2Bprocess%2Benriched%2Bevent%2Blog) is an anonymized real ServiceNow incident log by Amaral, Fantinato, and Peres (2018), DOI [10.24432/C57S4H](https://doi.org/10.24432/C57S4H), licensed CC BY 4.0. The official archive is `data/raw/r2_uci_incident/incident_management.zip`, SHA-256 `6294e29a…25d1d6`.

The source-only audit (`4b0b71c4…ee688`) confirms 141,712 rows and 24,918 incident identities. Reading only case ID, state and timing/outcome metadata, it found 6,128 eligible future seven-day closures and 9,084 eligible nonclosures. The source includes `resolved_at` and `closed_at` values on early rows; these are retrospective outcome fields and must never become candidate inputs.

## Frozen prospective identity

The case-manifest protocol (`c2f95eb5…b421d`) was frozen before feature access. An incident is eligible only if it has two distinct nonterminal updates after opening and before resolution or closure. The decision time is the second such update. The label is whether closure occurs strictly after decision time and within seven days. If closure is later, the case is negative; missing/conflicting endpoints and insufficient histories are excluded.

Cases are split by opening month: March 2016 training, April development, and May held-out. Within each split, cases are balanced deterministically by a protocol-bound hash, and no incident appears twice. The metadata-only manifest passed (`50b8d213…fcb8`):

| Split | Cases per class | Total |
| --- | ---: | ---: |
| Training | 1,386 | 2,772 |
| Development | 1,901 | 3,802 |
| Held-out | 1,679 | 3,358 |

The full manifest SHA-256 is `bc2f4b0f…98c54`. It contains hashed case identities, split, label, opening time, feature cutoff, outcome-window end, and preterminal update count. At this manifest stage no descriptive feature had yet been read or candidate scored; later work is documented below.

## Remaining methodological risks

- The dataset records final closure timestamps on every event row. Candidate extraction must use an explicit allowlist and never pass `resolved_at`, `closed_at`, `closed_code`, future states, or raw incident IDs into learning.
- The first two updates may occur shortly before closure. This is a prospective seven-day closure task, but not yet an established long-lead early-warning task. Report the cutoff-to-closure lead-time distribution before making an early-warning claim.
- Outcome ascertainment relies on a known final closure time. Cases without a trustworthy endpoint are excluded, not silently labeled negative.
- Calendar month can correlate with workload and outcomes. Month identity is a split key, not a feature; event-unit comparisons need matched simple baselines and an outcome-shuffle control.
- This is not a Local Credit Packet benchmark yet. First show that a Compact Event control has useful prospective signal under source, feature, and resource gates.

## Compact Event development gate: negative result

The one-shot protocol (`68ad7c04…befab`) froze a seven-column descriptive allowlist, nine tokens from the first two nonterminal updates, training-only route construction, simple lookup controls, outcome shuffle, budgets, and acceptance thresholds. Materialization passed for all 6,574 training/development cases (`1931bb03…c56841`): each second update equaled the manifest cutoff, and all admitted event times were at or before it. An initial implementation attempt used the wrong protocol ID for hashed-case matching; its failed diagnostic is retained as `workspace/evaluation/uci_incident_compact_materialize_attempt0_v1.json`. The corrected extraction was rerun **before** model scoring.

The frozen development result is `workspace/evaluation/uci_incident_compact_development_v1.json`. Balanced accuracy was 0.4982 for Compact Event versus 0.5445 for the last-state table and 0.5452 for the cutoff snapshot table. The training-outcome-shuffle control scored 0.5118. Thus Compact Event failed the 0.60 absolute threshold, +0.01 improvement over the strongest simple control, and 0.05 shuffle drop. Resource gates passed: 6,054 state bytes, 32 features, at most nine events per case, and 0.003% unknown development tokens. This is a negative result for this fixed encoding and local update rule, not evidence that event computation cannot solve incident prediction.

An access-boundary review found a protocol violation after scoring: the original materializer read descriptive fields transiently on *all* CSV rows before selecting training/development membership. It did not save, fit, or score held-out rows, but the strict promise that May descriptive values were never **read** was false. The [first boundary amendment](../workspace/evaluation/uci_incident_compact_boundary_amendment_v1.json) preserves this finding and pins the original result hash. A further review found that selected training/development rows after their case-specific cutoff also had descriptive cells read transiently before their event time was checked. The [second amendment](../workspace/evaluation/uci_incident_compact_boundary_amendment_v2.json) records this separate violation. Neither kind of row was admitted to the feature artifact. The materializer now checks case membership and time before descriptive access, backed by a reusable [sealed temporal CSV guard](SEALED_TEMPORAL_CSV.md); the candidate was **not** rescored. Accordingly the development result is exploratory, not a clean confirmatory gate, and May cannot be claimed as a pristine held-out for this protocol.

## Fresh-boundary feasibility

A metadata-only audit of unused whole opening months was frozen (`6608591d…7820`) and executed (`c9aac38d…da7f`). The existing minimum of 700 cases **per class** requires at least 1,400 eligible cases in a new month, even before checking class balance. The largest unused month has only 172 eligible cases. No unused month can meet this necessary support condition. This audit read no raw CSV row or descriptive feature and scored no candidate. The decision is to stop a confirmatory extension on this incident identity and seek a new independent source. The state-transition and relative-timing contrast remains a future hypothesis; do not tune it on the consumed April labels or present May as pristine.

## 日本語要約

UCIの実インシデントログについて、判断時点と将来7日間の閉鎖結果を分離しました。2016年3月を学習、4月を開発、5月をheldoutとし、各クラス1,386 / 1,901 / 1,679件の均衡ケースを凍結しました。`closed_at` や `resolved_at` は特徴に使っていません。学習・開発6,574件の特徴抽出は時刻検査に合格しましたが、Compact Eventの開発balanced accuracyは0.4982で単純な状態表の0.5445/0.5452を下回りました。事後監査で、初回抽出器がheldoutの記述列に加え、選択済みケースの判断時点後の記述列も一時的に読んだ境界違反を発見しました。これらの行は特徴成果物へ採用・保存・採点されていませんが、5月を完全未読のheldoutとは主張できません。未使用月は最大172件で必要な1,400件に届かないため、このソースでの確認実験を止め、結果は探索的な負例として保持します。抽出器と再利用可能な読み取りガードを修正しましたが、再採点はしていません。

## 简体中文摘要

针对UCI真实事件日志，研究已把决策时点与随后七天的关闭结果分开。2016年三月用于训练、四月用于开发、五月保留；每类分别有1,386、1,901、1,679个平衡案例。`closed_at` 和 `resolved_at` 没有作为特征。训练和开发集的6,574个案例通过时点检查，但Compact Event的开发集平衡准确率仅为0.4982，低于简单状态表的0.5445/0.5452。事后发现初版提取器短暂读取了保留集的描述性字段，以及已选案例在决策截止时间之后事件的描述性字段；这些行没有进入特征产物，也未用于评分，但五月数据不能再称为完全未读。已修复提取器并加入可复用的读取防护，没有重新评分。未用月份最多仅有172例，低于所需的1,400例，因此停止用该数据源作确认性扩展，并将负面结果仅作为探索性证据。

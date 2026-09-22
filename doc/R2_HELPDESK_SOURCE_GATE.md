# Helpdesk prospective source gate

## Source

The [Mendeley Data “Hepdesk anonymized” version 1](https://data.mendeley.com/datasets/nm9xkzhpm4/1), by Ilya Verenich, DOI [10.17632/nm9xkzhpm4.1](https://doi.org/10.17632/nm9xkzhpm4.1), states a CC BY 4.0 license. The official CSV was obtained from the dataset's file link into `data/raw/r2_helpdesk/helpdesk_anonymized.csv`, SHA-256 `a25beda5…fcffdc`. Attribution and version identity are required; the source is not interchangeable with differently processed Helpdesk logs.

## Metadata-only source audit

The audit protocol was frozen at `ece79c05…7bc260` before full parsing. Only `Case ID`, `Activity` and `Complete Timestamp` were accessed. The result (`0152ff7b…f78511`) passed: 22,748 timestamped events, 4,167 cases, all with at least two distinct timestamps, and 47 opening months from January 2010 through November 2013. Every case has a terminal `Closed` event; there are 4,405 `Resolve ticket` events. There are 4,251 repeated same-case timestamps. These numbers describe this exact CSV, not counts quoted for other Helpdesk exports. No non-audit column, candidate or held-out feature was opened.

## Frozen prospective feasibility: negative gate

Before model work, protocol `d9636942…97119` fixed a decision after the second distinct nonterminal update strictly after the case's first event, a future 30-day `Closed` outcome, and chronological opening-year splits: 2010–11 training, 2012 development, 2013 held-out. Each class needed at least 500/100/100 cases respectively before balancing. The retained result (`1b65ccf1…3dea46`) failed:

| Split | Closure within 30 days | Later closure | Required per class |
| --- | ---: | ---: | ---: |
| Training | 195 | 567 | 500 |
| Development | 95 | 210 | 100 |
| Held-out | 186 | 137 | 100 |

Another 2,777 cases lack two qualifying preterminal updates. Do not relax the class minimums or switch the cutoff on this consumed feasibility identity. No candidate was trained or scored, and no descriptive feature was read.

The observed cutoff-to-close ranges are approximately 15–30 days for positives and 30–60 days for negatives. A separate metadata-only timestamp audit (`8115ec49…24ad6`) tested whether `Closed` was *exactly* 15 days after the latest preceding `Resolve ticket`: 0/4,167 cases matched within one second, so that narrow mechanical-lag hypothesis did not pass. Rounded lags cluster near 15 days, however, and the source's closure-time semantics remain uncertain. This is an additional caution, not a proof of synthetic labels.

## Decision

Stop this prospective Helpdesk task at the source/feasibility gate. Seek a larger independent timestamped event source and freeze a case-time access test before reading its descriptive fields. Retain this negative result without post-hoc threshold changes.

## 日本語要約

公式CC BY 4.0のHelpdesk CSVを取得し、ハッシュを固定しました。この版には22,748イベント、4,167ケースがあります。時刻・活動名だけによる30日閉鎖タスクの事前ゲートでは、学習陽性195件（最低500件）と開発陽性95件（最低100件）が不足しました。`Resolve ticket`から`Closed`までの正確な15日固定も確認できず、閉鎖時刻の意味には不確実性が残ります。特徴抽出やモデル採点には進みません。

## 简体中文摘要

已取得官方CC BY 4.0的Helpdesk CSV并固定哈希；该版本含22,748个事件和4,167个案例。仅用时间与活动名称进行的预先规定的30天关闭任务检查未通过：训练集阳性195例（最低500例），开发集阳性95例（最低100例）。`Resolve ticket`到`Closed`严格相隔15天的假设未获支持，关闭时间的含义仍有不确定性。未提取描述性特征或评估模型。

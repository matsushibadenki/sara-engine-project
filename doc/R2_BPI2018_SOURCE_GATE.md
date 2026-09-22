# BPI Challenge 2018 source and prospective payment gate

## Official source and integrity

The [4TU.ResearchData BPI Challenge 2018, version 1](https://data.4tu.nl/articles/dataset/BPI_Challenge_2018/12688355/1), by van Dongen and Borchert, DOI [10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972), is marked CC0 on its landing page. The official compressed XES is `data/raw/r2_bpi2018/BPI_Challenge_2018.xes.gz`, 158,270,992 bytes. Its MD5 matches the publisher's `66e54dda…f7a5d4`, gzip integrity passes, and SHA-256 is `7055a531…2b30a9`. The [acquisition record](../data/processed/benchmark_fixtures/bpi2018_source_acquisition_v1.json) pins the URL and hashes.

An earlier BPI Challenge 2019 candidate had a clear CC BY 4.0 official page, but its uncompressed 728,558,522-byte XES was too slow to obtain reliably here. The incomplete 2.4 MB scratch download was deleted; no BPI 2019 event was parsed or scored. It is not treated as a source-pinned dataset.

## Metadata-only audit

Protocol `2702eb51…3564c3` froze access to only case `concept:name`, event `concept:name`, and event `time:timestamp`. The [audit result](../workspace/evaluation/bpi2018_source_audit_v1.json), SHA-256 `bf661b05…ddc9c87`, passed: 43,809 unique cases, 2,514,266 timestamped events, and three or more distinct timestamps in every case. Opening years have 14,746 cases in 2015, 14,550 in 2016, and 14,507 in 2017; six opened in 2014. The most frequent last activity is `finish payment` (34,830 cases), but many cases end on other activities. No descriptive attribute value was accessed, no outcome task or held-out set was selected, and no candidate was scored at this stage.

## Frozen 180-day payment feasibility: negative result

Before evaluating the candidate task, protocol `654efe4f…18d707` fixed the third distinct event timestamp as decision time. A case is positive only if its first `finish payment` is strictly after decision and within 180 days. Cases already paid at decision are excluded. A negative has no payment within the horizon and a source-wide observation end at or after the horizon; this is only a global follow-up proxy and does not prove individual-case ascertainment. Opening years 2015/2016/2017 were assigned to training/development/held-out, with at least 300 cases required **per class** in each.

The retained [feasibility result](../workspace/evaluation/bpi2018_payment_feasibility_v1.json), SHA-256 `32f18d55…8f62`, failed:

| Opening-year split | Payment within 180 days | No payment within 180 days | Required per class |
| --- | ---: | ---: | ---: |
| Training, 2015 | 1 | 14,745 | 300 |
| Development, 2016 | 605 | 13,945 | 300 |
| Held-out, 2017 | 1 | 14,505 | 300 |

The pronounced year dependence makes this fixed horizon unsuitable for the intended event-unit comparison. Do not select a longer horizon after seeing these class counts and present it as the same preregistered experiment. The source remains useful for other *new* hypotheses, but this payment task is closed. No descriptive features or model scores were produced.

## 日本語要約

4TU公式のCC0 BPI 2018を取得し、配布MD5・gzip完全性・SHA-256を確認しました。時刻・活動名・ケースIDのみの監査では43,809ケース、2,514,266イベントが確認できました。しかし事前固定した「3番目の時刻から180日以内の支払完了」は、2015年/2016年/2017年で陽性1/605/1件となり、各年300件の基準に失敗しました。結果を見て期間を延ばすことはせず、このタスクを停止します。記述属性の抽出もモデル採点も行っていません。

## 简体中文摘要

已获取4TU官方CC0的BPI 2018数据，并验证发布方MD5、gzip完整性及SHA-256。仅使用案例ID、活动名和时间的审计确认43,809个案例及2,514,266个事件。但预先固定的“第三个不同时间点之后180天内完成付款”任务，在2015/2016/2017年分别只有1/605/1个阳性案例，未达到每年每类300个案例的门槛。不会在查看结果后延长时间窗并当作同一预注册实验；未提取描述性属性或评估模型。

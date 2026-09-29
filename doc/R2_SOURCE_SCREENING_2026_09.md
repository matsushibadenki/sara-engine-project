# R2 source screening after negative prospective gates

This is a read-only source-catalog screen, not a model experiment or a new task identity. The criterion is stronger than total event count: the source must support a decision cutoff, an independently observable later outcome, sufficient positive and negative cases across chronological splits, and case-level evidence of follow-up for negatives.

| Candidate | Official-source observation | Current decision |
| --- | --- | --- |
| [BPI Challenge 2019](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1) | CC BY 4.0; the official XES is 728,558,522 bytes and uncompressed. | Acquisition deferred after slow partial transfer; incomplete scratch removed, no source audit. |
| [Road Traffic Fine Management Process](https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249/1) | Small XES.gz, but landing page names **4TU General Terms of Use**, not a simple Creative Commons license. | Do not assume CC reuse rights; review terms and outcome/censoring semantics before acquisition. |
| [Hospital Billing – Event Log](https://data.4tu.nl/articles/dataset/Hospital_Billing_-_Event_Log/12705113/1) | Small XES.gz; landing page also names **4TU General Terms of Use**. | Same license/semantics review required. |
| [PhysioNet/CinC 2019 sepsis](https://physionet.org/content/challenge-2019/1.0.0/) | Official CC BY 4.0; all 40,336 patient files acquired and integrity checked. The fixed hour-12/future-12-hour metadata gate yielded only 107/29/78 positives. | [Negative feasibility result](R2_PHYSIONET2019_SOURCE_PREFLIGHT.md). Stop this task without changing cutoff/horizon; no clinical features or candidate were scored. |
| [NASA PCoE C-MAPSS turbofan degradation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | The official PCoE repository now exposes a 12,429,152-byte ZIP. A read-only HEAD check on 2026-09-29 returned HTTP 200, `Last-Modified: 2022-09-18`, and single-part ETag `a83e8f128c59fc5614a4ca2e42a2e30c`. The corresponding NASA Open Data resource remains marked `License not specified`. | Availability block resolved; reuse-rights gate remains blocked. Do not acquire, parse or score until NASA states applicable data-use terms. Distinguish this downloadable degradation dataset from the unavailable C-MAPSS simulator/software entries. |
| [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricity) | CC BY 4.0; 370 client series, 140,256 quarter-hour timestamps, no reported missing values, and directly observable future consumption. The catalog declares DST zero/aggregation artifacts and pre-activation client zeros. | Selected for an acquisition-only gate under a frozen 2011–2012/2013/2014 split. Exclude declared artifacts and require client-level two-class support before feature materialization or candidate scoring. See [source contract](R2_ELECTRICITY_LOAD_SOURCE_GATE.md). |

The prior [Helpdesk](R2_HELPDESK_SOURCE_GATE.md) and [BPI 2018](R2_BPI2018_SOURCE_GATE.md) source audits passed, but their frozen delayed-outcome count gates failed. BPI 2018 also exposed the weakness of using a log-wide final timestamp as negative follow-up evidence. The new [case-level contract](PROSPECTIVE_EVENT_CASE_CONTRACT.md) must be satisfied before another candidate is scored. None of the candidates above has been scored in this screen.

## 日本語要約

総イベント数だけでなく、判断時点・将来結果・時系列分割ごとの両クラス件数・陰性のケース単位追跡証拠をソース選定条件にしました。BPI 2019は公式ファイルが大きく取得を保留し、交通反則金と病院請求ログは4TU一般利用規約と結果の意味を先に確認する必要があります。PhysioNet 2019の固定条件は陽性数不足で停止しました。NASA C-MAPSSの公式ZIPは現在取得可能ですが、配布リソースのライセンス表示が未指定のため、取得・解析はまだ行いません。いずれも新たなモデル採点はしていません。

## 简体中文摘要

筛选标准不只看事件总量，还要求明确的决策时刻、未来结果、各时间分组的两类样本数，以及负例的逐案例随访证据。BPI 2019的官方文件体积较大，暂缓获取；交通罚款和医院账单日志需先核查4TU通用使用条款及结果语义。PhysioNet 2019已因固定条件下阳性样本不足而停止。NASA C-MAPSS官方ZIP目前可访问，但资源许可证仍标为未指定，因此尚未下载或分析。这里没有进行新的模型评估。

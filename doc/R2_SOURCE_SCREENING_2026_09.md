# R2 source screening after negative prospective gates

This is a read-only source-catalog screen, not a model experiment or a new task identity. The criterion is stronger than total event count: the source must support a decision cutoff, an independently observable later outcome, sufficient positive and negative cases across chronological splits, and case-level evidence of follow-up for negatives.

| Candidate | Official-source observation | Current decision |
| --- | --- | --- |
| [BPI Challenge 2019](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1) | CC BY 4.0; the official XES is 728,558,522 bytes and uncompressed. | Acquisition deferred after slow partial transfer; incomplete scratch removed, no source audit. |
| [Road Traffic Fine Management Process](https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249/1) | Small XES.gz, but landing page names **4TU General Terms of Use**, not a simple Creative Commons license. | Do not assume CC reuse rights; review terms and outcome/censoring semantics before acquisition. |
| [Hospital Billing – Event Log](https://data.4tu.nl/articles/dataset/Hospital_Billing_-_Event_Log/12705113/1) | Small XES.gz; landing page also names **4TU General Terms of Use**. | Same license/semantics review required. |
| [PhysioNet/CinC 2019 sepsis](https://physionet.org/content/challenge-2019/1.0.0/) | Official CC BY 4.0; 40,336 subjects in two sets; one hourly ICU file per patient; sepsis labels shifted six hours early. Official S3 object-inventory gate passed at 20,336/20,000 files. | [Conditional separate task identity](R2_PHYSIONET2019_SOURCE_PREFLIGHT.md). No verified inter-patient calendar chronology or case-level follow-up audit yet; patient bodies not acquired or scored. |

The prior [Helpdesk](R2_HELPDESK_SOURCE_GATE.md) and [BPI 2018](R2_BPI2018_SOURCE_GATE.md) source audits passed, but their frozen delayed-outcome count gates failed. BPI 2018 also exposed the weakness of using a log-wide final timestamp as negative follow-up evidence. The new [case-level contract](PROSPECTIVE_EVENT_CASE_CONTRACT.md) must be satisfied before another candidate is scored. None of the candidates above has been scored in this screen.

## 日本語要約

総イベント数だけでなく、判断時点・将来結果・時系列分割ごとの両クラス件数・陰性のケース単位追跡証拠をソース選定条件にしました。BPI 2019は公式ファイルが大きく取得を保留し、交通反則金と病院請求ログは4TU一般利用規約と結果の意味を先に確認する必要があります。PhysioNet 2019は患者数と公開ライセンスが有望ですが、患者間の暦時刻順を確認できないため別タスク候補に留めます。いずれもモデル採点はしていません。

## 简体中文摘要

筛选标准不只看事件总量，还要求明确的决策时刻、未来结果、各时间分组的两类样本数，以及负例的逐案例随访证据。BPI 2019的官方文件体积较大，暂缓获取；交通罚款和医院账单日志需先核查4TU通用使用条款及结果语义。PhysioNet 2019的患者规模和开放许可有潜力，但尚未确认患者间的日历时间顺序，只能作为独立任务候选。此处没有评估任何模型。

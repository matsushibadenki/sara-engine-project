# R2 Independent Sepsis Event Task

## Selection — 2026-09-16

[Done] The independent replication task is the real-life Sepsis Cases hospital event log from 4TU.ResearchData. The publisher reports anonymized hospital ERP events and states that absolute timestamps were randomized while intervals within each case were preserved. Source DOI: `10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`.

The downloaded file matches publisher MD5 `b5671166ac71eb20680d3c74616c43d2`; SHA-256 is `709c52340306415952811b9b9c5dc6bcc8f8d47d583eba39df9a538459dc543a`. Audit found 1,050 cases, 15,214 events, 16 activities and 4,652 distinct within-case intervals, with no negative intervals. Case length ranges from 3 to 185 events.

The target is the next activity after each non-terminal event. Complete cases are ordered by start timestamp and case identifier, then split into 735 training, 157 development and 158 sealed test cases. Absolute time is forbidden as a semantic feature because the publisher randomized it. Only elapsed time within a case may be used.

[Done] The manifest fixes 10,083/2,046/2,035 training/development/sealed-test predictions. Development top-1 is 22.87% majority, 53.18% first-order, 55.87% second-order, 54.20% timing-aware and 53.42% constant-gap. Test metrics remain unopened.

[Done] The inherited hybrid development gate improved top-1 55.87%→56.06% and macro-F1 37.79%→41.26%, but worsened Brier 0.5706→0.5756 and failed timing/shuffled controls. Its override rate rose to 21.41%. Frozen test remains sealed. See [result](R2_SEPSIS_DEVELOPMENT.md).

[Done] The outcome-blind normalized-router diagnostic fixed a 10% override threshold from training scores without Sepsis development labels. Seven of eight original gates passed; the absolute shuffled-gap result remains recorded as negative.

[Done] The subsequent source-normalized final gate passed on all 2,035 sealed test predictions: top-1 55.97%→56.81%, macro-F1 35.43%→40.31% and Brier 0.5643→0.5625. See [final result](R2_SEPSIS_NORMALIZED_FINAL.md).

[Next] Consolidate the replicated mechanism into a generic bounded runtime and compare explicit-neuron and scalar execution costs on both real streams.

日本語: 第二の独立課題として、実際の病院ERPから得られたSepsis Casesを選びました。絶対時刻は匿名化されていますが症例内間隔は保存されています。症例単位の時系列分割を固定し、BPI由来の方式を調整せず再検証します。

简体中文：第二个独立任务采用真实医院ERP的Sepsis Cases日志。绝对时间已匿名化，但病例内事件间隔得到保留。我们固定按病例划分的时间顺序分区，并在不针对Sepsis调参的情况下复验BPI方案。

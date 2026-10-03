# Electricity Load R2 Source Gate

## Executed train/development support gate

[Done] The frozen feasibility audit passed. Training has 333 active clients and 225 qualifying clients; development has 349 active and 214 qualifying clients. A qualifying client has at least 100 rise and 100 fall observations within that split; the frozen minimum is 100 qualifying clients and 200 active clients. Training rise/fall counts are 81,482/83,483; development counts are 53,941/60,721. Exact ties excluded: 10,367/8,458 respectively.

The report is `workspace/evaluation/r2_electricity_load_support_v1.json`. The forward-only audit parsed 38,929,550 train/development cells and held at most 100 rows. All 35,041 rows outside numeric access remained opaque, including the entire 2014 split and the final 2015 row. No features were materialized and no candidate was scored. Fourteen related tests passed, covering sealed access, exact decimals, activation, invalid observations, ties, DST exclusions, split crossings and timestamp gaps. This result establishes support only, not predictive utility or independence. [Next] Freeze a matched learner comparison and calendar-block evaluation before any candidate scoring; held-out access remains unauthorized.

日本語: [Done] 固定済み支持数gateを通過しました。活動client／増減それぞれ100件以上あるclientは、学習333／225、開発349／214です。保持上限は100行、2014年の数値解析は0件です。特徴量生成・候補評価は行っていません。関連テスト14件成功。[Next] 同条件のlearner比較と暦ブロック評価を事前登録します。支持数は性能や独立標本数を意味しません。

简体中文：[Done] 固定支持量gate通过。活跃客户／增减各至少100次的客户为训练333／225、开发349／214。缓冲最多100行，2014年数值解析为零。未生成特征或评估候选，相关14项测试通过。[Next] 预注册同条件learner比较与日历分块评估。支持量不代表性能或独立样本量。

## Frozen feasibility access boundary

[Done] Feasibility protocol v1 is frozen at SHA256 `a2220cf5d49ac68fd74a71f923b7c43701f190975291d4dc7fd4f2b6ed4e9157`. Numeric support comparisons are authorized only in 2011–2013, without changing the historical source protocol. The reader checks the timestamp year before splitting load tokens. Sealed 2014 and out-of-scope 2015 payloads remain opaque: physical decompression still occurs, but numeric interpretation does not. No real load cells were numerically parsed in this step.

Daily noon cutoffs use 96 observations including the cutoff and a four-interval future outcome. Exact ties are excluded; activation is established forward in time. Windows intersecting an entire registered DST transition day are conservatively excluded. Each split requires 200 active clients and 100 clients with at least 100 observations of each direction. These are support thresholds, not independent sample sizes or performance results. [Next] Implement the streaming eligibility/count audit. Feature materialization and candidate scoring remain unauthorized.

日本語: [Done] 数値アクセス前にfeasibility protocolを凍結しました。2011–2013年の支持数だけを許可し、年を確認してから負荷トークンを解析します。2014年は封印、2015年は対象外です。日次正午cutoff、96観測の履歴、1時間後の増減、同値除外、前向きの活動開始判定、DST日の保守的除外を固定しました。実データの数値解析は未実行です。[Next] 有界ストリーミングの適格性・支持数監査を実装します。支持数は独立標本数ではありません。

简体中文：[Done] 在数值访问前冻结了feasibility协议。仅允许2011–2013年的支持量比较，先检查年份再解析负荷；2014年保持封存，2015年不在范围内。固定每日中午cutoff、96次观测历史、一小时后的增减、排除相等值、前向激活判定和保守的夏令时日期排除。尚未解析真实负荷数值。[Next] 实现有界流式资格与支持量审计。支持量不等于独立样本量。

## English

UCI ElectricityLoadDiagrams20112014 is selected for an acquisition-only source gate. The frozen protocol SHA-256 is `9828a9e359c5ebfb3e6fc88f1a399b562e31a99c3d27b07c5805773101509434`. The official catalog reports 370 client series, 140,256 quarter-hour timestamps, no missing values, Portuguese local time, and CC BY 4.0 licensing. It is a real consumption stream with directly observable future load, not a failure-event dataset.

The frozen protocol separates 2011–2012 training, 2013 development, and sealed 2014 held-out time. The proposed outcome is each client's load direction one hour after a cutoff, using at most 96 prior quarter-hours. This is only a proposed identity until source support passes.

The source explicitly encodes pre-activation client periods as zero and modifies daylight-saving transition hours. Those rows are not ordinary consumption observations. Client pre-activation prefixes, March DST zero hours, October aggregated hours, non-contiguous windows, and non-finite values must be excluded before support counts. At least 200 active clients and 100 clients with both outcome classes are required in every split.

The official archive acquisition is complete. The managed raw artifact is 261,335,609 bytes with SHA-256 `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`; its primary member is 710,998,915 bytes and every ZIP member passes CRC validation. The acquisition receipt SHA-256 is `ed194308adb6ac1d0bbca248f9fb0181ada64ab0bd9970d5ea45740e288baeb7`. The CSV body has not been extracted or parsed. Candidate scoring and held-out load-value access remain unauthorized. The next step is the frozen shape/support gate.

## Shape gate result — 2026-10-03

The streaming timestamp/shape audit passes all six checks: 370 distinct client columns, 140,256 rows, valid row shape, strict ordering and continuous 15-minute intervals. The first timestamp is `2011-01-01 00:15:00`; the last is `2015-01-01 00:00:00`. The single 2015 boundary row must be assigned by its timestamp in any future split.

Zero load cells were numerically parsed. Class support, activation and finite-value eligibility remain untested. The original source protocol permits only timestamp and finite/zero status access, which cannot establish whether loads increase or decrease. A separate feasibility protocol must explicitly register outcome comparisons and held-out access before execution. Shared calendar/grid effects also mean 370 client series cannot be assumed statistically independent.

日本語: shape gateは370列・140,256行・時刻順序・15分連続性の全条件を通過しました。最終行は2015年1月1日であり、今後のsplitは行時刻で決定します。負荷セルの数値変換は0件です。増減二クラスのsupportを評価するには、元protocolの有限／ゼロ状態の許可を越えるため、別のfeasibility protocolが必要です。同じ電力網・暦条件を共有するclientの統計的独立性は仮定しません。

简体中文：shape gate通过了370列、140,256行、时间顺序和连续15分钟间隔的全部检查。最后一行属于2015年1月1日，后续分组必须依据行时间。数值解析的负荷单元为零。原协议只允许有限性／零值状态访问，无法据此计算增减两类的支持量；执行前需要单独冻结feasibility协议。共享电网和日历条件的客户不能直接视为统计独立样本。

## 日本語

UCI ElectricityLoadDiagrams20112014を取得専用のsource gate候補に選定しました。公式カタログでは、370 client、15分間隔の140,256時点、欠損なし、ポルトガル現地時刻、CC BY 4.0とされています。故障イベントではなく、将来の消費電力を直接観測できる実系列です。

凍結protocolでは2011–2012をtraining、2013をdevelopment、2014を未開封held-outとします。最大96区間の履歴から1時間後のclient別負荷方向を予測する案ですが、source support gateを通るまでは正式なtask identityではありません。

client開始前のゼロ、3月DSTの人工ゼロ、10月DSTの集約値、非連続window、非有限値を除外します。全splitでactive client 200以上、両方のoutcome classを持つclient 100以上を要求します。公式archiveは261,335,609バイト、SHA-256 `f6c4d0e0…c8d4e1`として取得し、全memberのCRC検査を通過しました。CSV本文は未解析です。次はshape/support gateだけを実行し、candidate採点とheld-out値の参照は許可しません。

## 简体中文

已将UCI ElectricityLoadDiagrams20112014选为仅限获取阶段的source gate候选。官方目录说明其包含370个客户序列、140,256个15分钟时点、无缺失值、葡萄牙当地时间，并采用CC BY 4.0许可。它是真实用电序列，可直接观察未来负荷，但不是故障事件数据集。

冻结协议将2011–2012年作为训练、2013年作为开发、2014年作为封存保留期。拟议目标是利用最多96个历史区间预测一小时后的客户负荷方向；在source support gate通过前，这还不是正式任务身份。

必须排除客户启用前的零值前缀、三月夏令时人工零值、十月聚合小时、不连续窗口和非有限值。每个分组至少需要200个活跃客户，以及100个同时具有两类结果的客户。官方压缩包已获取，大小为261,335,609字节，SHA-256为`f6c4d0e0…c8d4e1`，所有成员均通过CRC检查；CSV正文尚未解析。下一步仅执行shape/support gate，不允许候选模型评分或读取保留期负荷值。

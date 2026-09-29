# Electricity Load R2 Source Gate

## English

UCI ElectricityLoadDiagrams20112014 is selected for an acquisition-only source gate. The frozen protocol SHA-256 is `9828a9e359c5ebfb3e6fc88f1a399b562e31a99c3d27b07c5805773101509434`. The official catalog reports 370 client series, 140,256 quarter-hour timestamps, no missing values, Portuguese local time, and CC BY 4.0 licensing. It is a real consumption stream with directly observable future load, not a failure-event dataset.

The frozen protocol separates 2011–2012 training, 2013 development, and sealed 2014 held-out time. The proposed outcome is each client's load direction one hour after a cutoff, using at most 96 prior quarter-hours. This is only a proposed identity until source support passes.

The source explicitly encodes pre-activation client periods as zero and modifies daylight-saving transition hours. Those rows are not ordinary consumption observations. Client pre-activation prefixes, March DST zero hours, October aggregated hours, non-contiguous windows, and non-finite values must be excluded before support counts. At least 200 active clients and 100 clients with both outcome classes are required in every split.

The official archive acquisition is complete. The managed raw artifact is 261,335,609 bytes with SHA-256 `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`; its primary member is 710,998,915 bytes and every ZIP member passes CRC validation. The acquisition receipt SHA-256 is `ed194308adb6ac1d0bbca248f9fb0181ada64ab0bd9970d5ea45740e288baeb7`. The CSV body has not been extracted or parsed. Candidate scoring and held-out load-value access remain unauthorized. The next step is the frozen shape/support gate.

## 日本語

UCI ElectricityLoadDiagrams20112014を取得専用のsource gate候補に選定しました。公式カタログでは、370 client、15分間隔の140,256時点、欠損なし、ポルトガル現地時刻、CC BY 4.0とされています。故障イベントではなく、将来の消費電力を直接観測できる実系列です。

凍結protocolでは2011–2012をtraining、2013をdevelopment、2014を未開封held-outとします。最大96区間の履歴から1時間後のclient別負荷方向を予測する案ですが、source support gateを通るまでは正式なtask identityではありません。

client開始前のゼロ、3月DSTの人工ゼロ、10月DSTの集約値、非連続window、非有限値を除外します。全splitでactive client 200以上、両方のoutcome classを持つclient 100以上を要求します。公式archiveは261,335,609バイト、SHA-256 `f6c4d0e0…c8d4e1`として取得し、全memberのCRC検査を通過しました。CSV本文は未解析です。次はshape/support gateだけを実行し、candidate採点とheld-out値の参照は許可しません。

## 简体中文

已将UCI ElectricityLoadDiagrams20112014选为仅限获取阶段的source gate候选。官方目录说明其包含370个客户序列、140,256个15分钟时点、无缺失值、葡萄牙当地时间，并采用CC BY 4.0许可。它是真实用电序列，可直接观察未来负荷，但不是故障事件数据集。

冻结协议将2011–2012年作为训练、2013年作为开发、2014年作为封存保留期。拟议目标是利用最多96个历史区间预测一小时后的客户负荷方向；在source support gate通过前，这还不是正式任务身份。

必须排除客户启用前的零值前缀、三月夏令时人工零值、十月聚合小时、不连续窗口和非有限值。每个分组至少需要200个活跃客户，以及100个同时具有两类结果的客户。官方压缩包已获取，大小为261,335,609字节，SHA-256为`f6c4d0e0…c8d4e1`，所有成员均通过CRC检查；CSV正文尚未解析。下一步仅执行shape/support gate，不允许候选模型评分或读取保留期负荷值。

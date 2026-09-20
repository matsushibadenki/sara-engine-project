# C-MAPSS Source Gate

## Decision

NASA C-MAPSS is a technically suitable future benchmark but is not selected yet. The current official metadata describes four simulated run-to-failure fleets with 708 training and 708 test trajectories in total. Engines are independent time series; training trajectories reach failure and test trajectories terminate before failure with supplied remaining-useful-life outcomes.

Two source conditions block selection:

1. The NASA catalog marks the resource license as **not specified**.
2. The official ZIP endpoint returned no bytes during a request lasting more than 60 seconds, and a separate 20-second HEAD request timed out.

No mirror was substituted, no file was retained, no sensor value was read, and no task or candidate was scored. The evidence record is `workspace/evaluation/cmapss_source_availability_v1.json`.

C-MAPSS is also simulated rather than field-observed. If the official artifact becomes available and its research use is reviewed, it should be positioned as a controlled temporal-mechanism benchmark, not as real-world predictive-maintenance validation.

## 日本語要約

C-MAPSSは独立engine系列と明確なrun-to-failure結果を持つため技術的には適しています。しかし公式配布URLが応答せず、NASAカタログ上のlicenseも未指定です。非公式mirrorへの切替は行わず、現時点では選定を保留しました。またシミュレーションデータであり、実設備での有効性の証拠にはしません。

## 简体中文摘要

C-MAPSS具有独立发动机序列和明确的运行至故障结果，在技术上适合作为基准。但NASA官方文件地址当前无响应，目录中的许可证也未注明。因此未改用非官方镜像，并暂停选择。该数据是模拟数据，不能作为真实设备有效性的证据。

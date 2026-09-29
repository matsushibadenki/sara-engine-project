# C-MAPSS Source Gate

## Decision

NASA C-MAPSS is a technically suitable future benchmark but is not selected yet. The current official metadata describes four simulated run-to-failure fleets with 708 training and 708 test trajectories in total. Engines are independent time series; training trajectories reach failure and test trajectories terminate before failure with supplied remaining-useful-life outcomes.

The official-source availability was rechecked on 2026-09-29. NASA's PCoE repository now links an official S3 ZIP, and a read-only HEAD request returned HTTP 200 with `Content-Length: 12429152`, `Last-Modified: Sun, 18 Sep 2022 03:42:08 GMT`, and the single-part ETag `a83e8f128c59fc5614a4ca2e42a2e30c`. The earlier availability block is resolved.

One source condition still blocks selection:

1. The NASA catalog marks the resource license as **not specified**.

Public HTTP availability is not treated as permission to reuse the data. No archive body was downloaded, no mirror was substituted, no sensor value was read, and no task or candidate was scored. The retained original failure is `workspace/evaluation/cmapss_source_availability_v1.json`; the availability-only recheck is `workspace/evaluation/cmapss_source_availability_v2.json` (SHA-256 `551c9089…1d8714`).

C-MAPSS is also simulated rather than field-observed. If the applicable data-use terms become explicit and are reviewed, it should be positioned as a controlled temporal-mechanism benchmark, not as real-world predictive-maintenance validation.

## 日本語要約

C-MAPSSは独立engine系列と明確なrun-to-failure結果を持つため技術的には適しています。2026-09-29の再確認では公式ZIPがHTTP 200で応答し、12,429,152バイトと単一ETagを確認できました。しかしNASAカタログ上のlicenseは未指定です。公開URLへのアクセス可否を再利用許可とは見なさず、データ本体は取得していません。シミュレーションデータであり、実設備での有効性の証拠にもなりません。

## 简体中文摘要

C-MAPSS具有独立发动机序列和明确的运行至故障结果，在技术上适合作为基准。2026-09-29复查时，官方ZIP返回HTTP 200，并确认大小为12,429,152字节及单段ETag；但NASA目录中的许可证仍未注明。公开URL可访问不等于获得再使用许可，因此没有下载数据正文。该数据是模拟数据，也不能作为真实设备有效性的证据。

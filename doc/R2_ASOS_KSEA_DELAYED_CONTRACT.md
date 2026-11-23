# KSEA delayed extinction outcome contract V1

## Preregistered scope

Use KSEA / WBAN 24233 DSI-6405 page 1 only. For any separately authorized acquisition, allow exactly the six registered January–June 2024 station-month files in order, one HTTPS GET each from the NCEI inventory origin, with no redirects, retry, alternate product or fallback station. Require a complete identity-encoded response and cap each body at its calendar minutes × 101 bytes (99 record bytes plus CRLF), 26,470,080 bytes total. Stop on missing/truncated response or cap violation. Retain July exclusion and August seal. This protocol implementation uses normalized, synthetic in-memory observations only. Future acquisition and byte decoding require a separate entry point and verification of first-sensor provenance; EPA documents variable column placement. [NCEI layout](https://www.ncei.noaa.gov/data/automated-surface-observing-system-one-minute-pg1/doc/asos-1min-pg1_documentation.pdf) · [EPA format guide](https://www.epa.gov/sites/default/files/2020-10/documents/aerminute_userguide.pdf)

Each normalized minute supplies local standard time, first-sensor extinction in exact nonnegative thousandths of the archive coefficient, and D/N. Missing, undocumented or malformed readings become an unavailable minute at the normalization boundary. Valid normalized rows must have a naive minute-aligned timestamp inside the development split, in strictly increasing order. A missing row or unavailable minute clears the history. Use at most 13 valid contiguous minutes in memory.

At prediction minute t, the exact current key is the coefficient, D/N, weekday, hour and quarter-hour. The history contains eight minutes ending at t. The history diagnostic is the sign of the difference between the last and first coefficient, with falling, flat and rising values. This gives a finite contrast without coarsening current coefficient. The result is the first-sensor coefficient at t+5, positive at or above 0.100 archive units and negative below it. This threshold defines an extinction proxy and has no airport visibility-distance interpretation. The one-minute outcome window starts at t+4, wholly after prediction. Require continuity through delivery, without filling gaps. Emit the delayed result only on its due minute. Prediction inputs contain only observations available by t. This diagnostic makes no claim that the listed current key is sufficient for the final strongest-control comparison.

Verify exact threshold equality, strictly delayed delivery, opposite histories at the same exact current key, missing rows, invalid normalized values, ordering and sealed dates. Keep both classes, recurrence, adequate opposing-history support, existing evidence and matched actual local writes as gates before a learner comparison. Original Kitchen ON, CASAS, negative results and seals retain their own definitions and thresholds. A synthetic pass establishes the contract only.

## Verification

🟢 [Done] A fixed acquisition manifest and streaming delayed-outcome extractor are implemented. Synthetic verification covers same-key opposing histories, both threshold sides, delivery time, gaps, malformed normalized rows and split violations. 42 related tests passed. No station payload was acquired or opened.

🟢 [Done] The [first-sensor normalization follow-up](R2_ASOS_KSEA_FIRST_SENSOR_NORMALIZATION.md) verifies the example-shaped layout with synthetic raw records. Real development coverage and support are pending. Independent testing and causal SNN or resource claims remain gated.

日本語　2024年1〜6月の取得上限と5分遅延の消散係数代理結果を固定し、同じ現在状態・異なる履歴の識別を合成入力で検証した。観測データ本体は取得しておらず、8月は封印を維持する。

简体中文　固定了2024年1至6月的获取上限与延迟5分钟的消光代理结果，并以合成输入验证同一当前状态下的历史区分。未获取观测数据，8月继续封存。

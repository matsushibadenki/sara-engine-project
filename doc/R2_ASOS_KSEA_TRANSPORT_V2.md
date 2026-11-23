# KSEA development transport V2

## Public metadata audit

🟢 [Done] On 2026-11-23 JST, six direct HTTPS HEAD requests with `Accept-Encoding: identity` returned HTTP 200 with the lengths and strong ETags below. No observation response body was requested. The NCEI monthly directory indexes identify the same station-month filenames. [NCEI 2024 archive indexes](https://www.ncei.noaa.gov/data/automated-surface-observing-system-one-minute-pg1/access/2024/)

| Month | HEAD Content-Length (bytes) | V1 cap (bytes) | Difference (bytes) | ETag (inside quotes) |
| --- | ---: | ---: | ---: | --- |
| 2024-01 | 4,668,708 | 4,508,640 | +160,068 | `473d24-6106bbb1c40c0` |
| 2024-02 | 4,389,485 | 4,217,760 | +171,725 | `42fa6d-61276d2d6d740` |
| 2024-03 | 4,689,387 | 4,508,640 | +180,747 | `478deb-61522065b9300` |
| 2024-04 | 4,809,958 | 4,363,200 | +446,758 | `4964e6-6180a66574980` |
| 2024-05 | 4,194,108 | 4,508,640 | −314,532 | `3fff3c-619ed408a1a40` |
| 2024-06 | 4,837,756 | 4,363,200 | +474,556 | `49d17c-621f14e414800` |
| Total | 27,589,402 | 26,470,080 | +1,119,322 | |

The January HEAD exceeds the V1 cap by 160,068 bytes, reproducing the reason the size gate rejects the currently published representation. V1 did not retain the original GET length or validator, so its exact header cannot be reconstructed. Five months exceed the maximum implied by one 99-byte record plus up to two line-ending bytes per scheduled minute. NCEI describes a 99-character format and space padding for truncated records. EPA notes variable field positions and garbled records. The metadata cannot distinguish extra rows, longer physical records, other framing bytes or a combination of causes. A smaller file also does not establish valid coverage. [NCEI DSI-6405 §2](https://www.ncei.noaa.gov/data/automated-surface-observing-system-one-minute-pg1/doc/asos-1min-pg1_documentation.pdf) · [EPA AERMINUTE Figure 1](https://www.epa.gov/sites/default/files/2020-10/documents/aerminute_userguide.pdf)

## Preregistered transport and framing diagnostic

🟠 [Next] Transport V2 is registered and unexecuted. Use the same six URLs in order, one direct HTTPS GET per URL, with no redirects, retries or additions. Keep the 60-second socket timeout and 300-second total deadline. Send identity encoding and the corresponding quoted ETag in `If-Match`. Before reading each body, require HTTP 200, the matching ETag and one exact Content-Length as listed above, identity or absent Content-Encoding, and no Transfer-Encoding. Reserve the listed size before reading. Stop on mismatch, truncation or timeout. The fixed aggregate body ceiling is 27,589,402 bytes. Record lengths and SHA-256 digests of completed bodies. The ETags are HTTP validators. The SHA-256 values identify the acquired bytes.

The revision changes transport admission while preserving the existing decoder. Before visibility normalization, run the unchanged V1 framing and timestamp parser for each acquired month, using empty streams for the other months. If a structural check fails, record aggregate LF-delimited length frequencies (excluding LF, retaining CR), final-LF presence and the failing stage for that complete body, then stop before requesting the next month. Publish no payload or example timestamps. Only six fully validated months permit the unchanged development summary. A framing failure remains inconclusive. A new decoder would require separate evidence and registration.

The V1 stop, the 180 eligible development dates, exact-current recurrence, history and class support, original evidence thresholds and matched actual-write conditions remain in force. No learner, scores or resource comparison are authorized by this contract. July is excluded and August remains sealed.

日本語　HEADで6ファイルの正確なサイズを確認した。1月は旧上限を160,068バイト超過。物理形式の差の内訳は未確認。本文を取得せず、別版の取得・停止条件を登録した。

简体中文　通过HEAD核对六个文件的精确大小。1月比旧上限多160,068字节，物理记录差异仍待确认。未获取正文，已登记新版传输和停止条件。

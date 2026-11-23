# KSEA January 112-byte line format review

🟢 [Done] Reviewed the public NCEI specification, EPA's field map and the Iowa Environmental Mesonet archive parser against the saved V2 aggregate. The aggregate records 41,316 lines of 112 bytes excluding LF, retaining any CR. January's response body was not retained, so the aggregate establishes neither the content nor the location of bytes beyond the documented width. No observation body or preview was retrieved in this review. [V2 receipt](R2_ASOS_KSEA_TRANSPORT_V2.md#execution-result)

| Source | What it establishes | Remaining gap |
| --- | --- | --- |
| NCEI DSI-6405 §2 | Describes a 99-character format, space padding for truncated records, and an example in which the first and second extinction/illumination pairs follow the station and time header. | Provides no byte map for the KSEA 112-byte physical line. |
| EPA AERMINUTE Figure 1 | Lists one-based columns 31–41 as an observed range for the first extinction coefficient and illumination, and labels the second sensor position variable. The table qualifies the ranges as empirical and the guide describes garbled records. | No exact first-sensor slice or missing-first-slot rule for KSEA January 2024 follows from the range. |
| Iowa Environmental Mesonet Page 1 parser | Reads the timestamp from columns 14–29, then tokenizes the payload from column 31 and numbers the coefficient pairs it finds in order. | The source code provides no interpretation of the extra bytes. Enumerating present pairs cannot by itself prove that a pair belongs to sensor 1 when that sensor's field is absent. |

[NCEI DSI-6405](https://www.ncei.noaa.gov/data/automated-surface-observing-system-one-minute-pg1/doc/asos-1min-pg1_documentation.pdf) · [EPA AERMINUTE guide](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P1018NUM.TXT) · [Iowa Environmental Mesonet parser](https://github.com/akrherz/iem/blob/main/scripts/ingestors/asos_1minute/parse_ncei_asos1minute.py)

⭕️ [Pending] The extra bytes' meaning, possible CR contribution, and first-sensor position in the delivered representation remain unverified. Hold a new decoder until a publisher layout or attributable format examples establish the placement and identify the first slot even when it is missing and a later sensor is present. Synthetic records would then test that documented contract. No parser revision was registered or implemented. The V2 framing stop, existing task conditions, negative results, July exclusion and August test seal remain in force. No learning or scoring was run.

日本語　公開仕様で第一視程センサーのおおよその位置は確認できた。KSEAの112バイト行の追加部分と、第一センサーが欠測した場合の欄位置は確定できず、別版パーサーを保留した。

简体中文　公开资料给出了第一能见度传感器的大致列范围，但无法确认KSEA的112字节行中额外字节的含义及第一传感器缺测时的字段位置，暂缓新版解析器。

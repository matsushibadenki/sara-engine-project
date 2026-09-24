# PhysioNet/CinC 2019 sepsis source preflight

## Official-source facts

The [PhysioNet/CinC Challenge 2019, version 1.0.0](https://physionet.org/content/challenge-2019/1.0.0/) is an openly accessible, CC BY 4.0 research dataset (DOI [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)). Its official description reports 40,336 subjects across training sets A and B, with one pipe-separated file per subject and hourly ICU measurements. The outcome column, `SepsisLabel`, is shifted six hours *before* the sepsis time for sepsis patients. The source describes `ICULOS` as hours since ICU admission.

## Official object-inventory gate: passed

Before acquiring any patient body, a fixed [inventory protocol](../data/processed/benchmark_fixtures/physionet2019_s3_inventory_protocol_v1.json) (SHA-256 `20347146…6a1`) enumerated only public S3 keys, sizes and ETags under the official `physionet-open/challenge-2019/1.0.0/training/` prefix. The [retained result](../workspace/evaluation/physionet2019_s3_inventory_v1.json) (SHA-256 `7aea0182…3141d2`) passed the publisher counts: set A has 20,336 patient files/131,388,704 bytes and B has 20,000/123,668,780 bytes. The sorted object-metadata digest is `4e77c4d1…95eb37`. No patient file body, label, or descriptive value was read; no candidate was scored. The official versioned HTTP path for an alleged `training_setA.zip` returned 404, so future acquisition should use the verified individual-object inventory rather than assume that ZIP exists.

## Research decision: conditional, separate identity

This is a promising patient-level prospective sequence source, but **not** a direct replacement for the current calendar-chronological event-log gate. The published description does not establish comparable patient admission dates for an inter-patient chronological split. Sets A and B may support a cross-hospital/domain comparison, but their independence, hospital membership, prevalence, and observation-horizon coverage must be audited from the official files. Do not combine this identity with the already used Sepsis Cases hospital-process log or claim a prospective medical benchmark from the catalog alone.

Before any descriptive value or model score is read, a new protocol must freeze:

1. Download only objects matching the frozen official inventory, verify each body's size and checksum, and audit one patient identity per file with no cross-set duplicates. The S3 ETag is recorded as source metadata, not assumed to be a content checksum without verification.
2. A patient-isolated train/development/held-out allocation that honestly states whether it is cross-domain or randomized, **not calendar-chronological** unless admission dates can actually be verified.
3. The decision hour and label semantics. The shifted `SepsisLabel` must be used only as outcome metadata, never as an input. An onset-time proxy recovered from its first `1` would itself be six hours after that row; this derivation must be checked against the source definition and never treated as an observed onset timestamp without qualification.
4. A finite prediction horizon and case-level observation through that horizon for negatives. The last recorded ICU hour can establish within-record coverage only; it does not prove absence of sepsis after discharge. Exclude censored windows rather than using a global data endpoint.
5. Minimum positive and negative patients in every fixed split, feature-missingness handling, last-state and no-time controls, bounded event work, and a clinical-non-deployment statement.

Only a metadata/label/time audit is authorized as the next source step. If its fixed support or follow-up gate fails, stop before feature materialization. A patient-hour score is not a patient-level independent sample count.

## Acquisition and fixed feasibility gate: negative

The resumable acquisition protocol (`ff96e8e…e0ca5`) re-enumerated the frozen source before transfer and accepted only single-part MD5 ETags. All 40,336 official objects and 255,057,484 bytes passed source-size, MD5 and local SHA-256 verification. The retained acquisition result is `workspace/evaluation/physionet2019_acquisition_v1.json` (SHA-256 `3427bd69…788db`), with content-manifest digest `d1d59b98…0eed7f`. It parsed no clinical column or label.

Before the feasibility protocol was frozen, a manual column-position check displayed the final two `HospAdmTime` and `SepsisLabel` values for `training_setA/p000001.psv`. The [inspection record](../workspace/evaluation/physionet2019_preprotocol_inspection_v1.json) preserves the error; that patient was excluded from all subsequent counts and any future task use.

The fixed metadata-only protocol (`36023484…16d52`) then selected ICU hour 12 as decision time and hours 13–24 as the future official-label horizon. A negative required a continuous one-based `ICULOS` sequence observed through hour 24. Set A was deterministically divided 4:1 into training/development; set B was held out as a domain split, explicitly **not** a calendar split. Minimum positive/negative support was fixed at 500/500, 100/100 and 300/300.

The [result](../workspace/evaluation/physionet2019_feasibility_v1.json) (`31a985c1…aee11`) failed every split's positive-support gate:

| Split | Positive | Negative | Required per class |
| --- | ---: | ---: | ---: |
| Training | 107 | 7,996 | 500 |
| Development | 29 | 2,041 | 100 |
| Held-out domain | 78 | 13,843 | 300 |

In addition, 8,793 patients failed the preregistered one-based `ICULOS` requirement, 6,625 lacked observation through hour 24, and 823 were already positive by the decision hour. These exclusion rules and thresholds were not relaxed after inspection. No clinical feature was interpreted or materialized and no candidate was scored. This exact task is stopped; a different cutoff or horizon would require a separately motivated new identity, not a post hoc repair.

## 日本語要約

PhysioNet/CinC 2019は公式にCC BY 4.0、患者約4万例のICU時間列として公開されています。公式S3の40,336ファイルを取得し、255,057,484バイトすべてのMD5とSHA-256を検証しました。事前固定した「ICU 12時間時点から将来12時間の公式陽性ラベル」は、学習/開発/heldoutで陽性107/29/78件となり、必要な500/100/300件を満たしませんでした。時刻列不適格、追跡不足、判断時点以前の陽性も固定条件で除外しました。事前確認で患者1人の管理時刻とラベルを誤って表示したため、その患者は全工程から除外しました。閾値を緩めず、このタスクを停止します。臨床特徴の抽出とモデル採点はしていません。

## 简体中文摘要

PhysioNet/CinC 2019官方以CC BY 4.0发布约四万名患者的ICU逐小时序列。已获取官方S3的40,336个文件，并验证全部255,057,484字节的MD5和SHA-256。预先固定的“ICU第12小时后未来12小时官方阳性标签”在训练/开发/保留域仅有107/29/78个阳性案例，低于500/100/300的门槛。时间序列不合格、随访不足和决策前已阳性的患者也按固定规则排除。因协议冻结前误看了一名患者的管理时间与标签，该患者已从全部后续流程中排除。没有放宽门槛；停止此任务，未提取临床特征，也未评估模型。

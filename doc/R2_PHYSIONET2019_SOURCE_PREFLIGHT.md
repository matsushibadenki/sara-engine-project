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

## 日本語要約

PhysioNet/CinC 2019は公式にCC BY 4.0、患者約4万例のICU時間列として公開されています。公式S3のオブジェクト一覧監査では、A/Bの20,336/20,000ファイルと合計255,057,484バイトを確認しました。ただし患者ファイル本体は未取得・未読です。患者間の暦時刻順も確認できず、陽性ラベルは発症6時間前に前倒しされています。現行の暦時刻分割タスクと同一視せず、取得後まず患者ID、分割、判断時刻、ラベル、陰性のケース単位追跡期間だけを監査します。採点はしていません。

## 简体中文摘要

PhysioNet/CinC 2019官方以CC BY 4.0发布约四万名患者的ICU逐小时序列。官方S3对象清单审计确认A/B分别有20,336/20,000个文件、共255,057,484字节；尚未获取或读取患者文件内容。官方说明没有证实患者之间可按日历时间排序，阳性标签还被提前到发病前六小时，因此不能直接替代现有的日历时间分组任务。取得并固定文件后，应先只审计患者身份、分组、决策时刻、标签及负例的逐案例观察期；若样本支持不足，应在读取描述性特征前停止。没有评估模型。

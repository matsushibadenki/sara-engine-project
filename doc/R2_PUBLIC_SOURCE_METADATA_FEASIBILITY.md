# Public source acquisition feasibility — 2026-10-18

🟢 [Done] A preregistered **TIHM → SPHERE 100 Homes → IDEAL → ECO** metadata review finds **no acquisition-ready source under the unchanged conditions**. TIHM's documented duration is insufficient. IDEAL's documented high-rate stream is incompatible and its full household bundle exceeds the budget. SPHERE and ECO remain unresolved, rather than excluded by an invented zero count. No dataset payload, row preview, prior sealed interval or learner is opened.

## Registration and scope

Candidate names, order, conservative method and aggregate budgets were fixed at `2026-10-18T03:29:33Z`, before external searches. Selection SHA-256 is `ae07845150a6338fd648e5974a02bb267deb59add7f950b216090fc6ba6ad822`; it preserves **338** prior JSON/gzip artifacts, including all CASAS exclusions and learner negatives. These four source families do not occur in the earlier local source protocols; actual household non-overlap and independent predictive usefulness remain future prerequisites.

The scalar implementation, evidence and calculation cases were pinned at `03:35:23Z`, **after metadata was known and before result calculation**. Code registration SHA-256 is `ff88eff2af9fef096f9355e1b0d6b41a37601965ff1bf5ff90a389fdab9acde3`. This is a public-metadata feasibility audit, not blind scoring or a new performance experiment.

| Review budget | Registered | Used |
| --- | ---: | ---: |
| Source families | 4 | 4 |
| Search queries | 8 | 8 |
| Primary metadata page attempts, including failures | 16 | 16 |
| Retained evidence UTF-8 bytes | 1,048,576 | 8,792 |
| Maximum retained bytes per page | 65,536 | Within cap |
| Dataset payload bytes / preview rows | 0 / 0 | 0 / 0 |
| Actual learning writes | 0 | 0 |

The retained-byte count covers the saved evidence JSON, not unobservable browser HTTP/TLS traffic. Search results may mention other datasets; none are added to the roster or investigated. Page records include inaccessible pages and CAPTCHA/redirect failures. Facts are manually transcribed with primary URLs and section references; the evidence hash pins that transcription, not the publisher's entire page or a downloaded dataset. Browser failures establish neither source impossibility nor absence of data.

Any later payload proposal retains the existing ceilings: **192 MiB requested HTTP body bytes**, **2 GiB expanded across passes**, **512 MiB per member per pass**, **5,000,000 training rows per identity** / 15,000,000 aggregate training timestamp parses, 36,000,000 metadata timestamp parses, 263,295 qualified outcomes, 789,885 changed writes per arm and 100,000 routes. No payload is authorized by this review; these are future planning ceilings, not a newly acquired source.

The original prospective target also remains unchanged: the first observed binary notification 300–600 seconds later is positive when its raw sensor name starts with `kitchen` and its message is `ON`. Exact current sensor/message and clock inputs, `log(80)` evidence, class/day support and sufficient controls remain required. This count-only review does not verify that another source can supply that target; entrance motion, occupancy or clinical alerts are not substituted for it.

## What the public metadata establishes

| Source | Period and density evidence | Review outcome |
| --- | --- | --- |
| TIHM | Activity schema dates lie in a 91-day global envelope | Exclude this documented release on the 180-recorded-date prerequisite |
| SPHERE 100 Homes | Year-long installations; binary PIR history packets with overlap | Potential duration/density, but compatible event semantics, usable dates and sized access manifest unresolved |
| IDEAL | Long installations; 1 Hz electrical and 12-second contextual measurements | Documented high-rate stream is not native binary events; full household bundle over budget; compatible subset unverified |
| ECO | Eight-month campaign; 1 Hz electricity; PIR in some homes | Campaign permits at most 163 training dates; per-home PIR coverage/cadence/counts and bounded access remain unresolved |

TIHM's author [Activity schema](https://github.com/PBarnaghi/TIHM-Dataset/blob/main/README.md?plain=1) places the release between April 1 and June 30, 2019. This bounds any contained identity to at most **91** recorded dates; the mean participant duration is not used as a maximum. The [release catalog](https://zenodo.org/records/7622128) lists about 7 MB and research/non-commercial terms. Small size does not rescue duration. This is an exclusion based on documented scope, not a timestamp-validated certificate.

The [SPHERE study page](https://www.bristol.ac.uk/research/groups/digital-health/research/sphere/the-100-homes-study/) supports long installations. Its [technical guide, sections 1 and 4.3.2](https://arxiv.org/html/1805.11907v2) specifies overlapping PIR histories: packet cadence cannot be treated as independent binary-event cadence. Public per-household coverage and byte manifests were not established within this review. Neither packet expansion nor retrospective delivery of historical samples is authorized.

IDEAL's [data descriptor](https://www.nature.com/articles/s41597-021-00921-y) documents high-frequency physical measurements and installation spans; these do not establish recurring native binary messages. The [publisher catalog](https://datashare.ed.ac.uk/items/15e82917-56ed-4282-99ef-7f211bf87f9d/full) lists `household_sensors.zip` at **14.77 GB**, beyond 192 MiB. This rejects the documented full-bundle route, not every conceivable subset. No range/subset delivery was verified and no power thresholding or downsampling is introduced.

The [ECO project](https://vs.inf.ethz.ch/res/show.html?what=eco-data) distinguishes electrical sampling from PIR. The [author paper, section 5](https://vs.inf.ethz.ch/publ/papers/beckel-2014-nilm.pdf) bounds the campaign to June 2012–January 2013. Its electrical total cannot be assigned to a household's PIR stream. The inclusive month envelope has at most **245** dates, without certifying that any household records all of them. The project offers an archive link and a credentialed dashboard; a bounded public PIR package and usable event schema remain unverified. No author messages or access requests were sent.

## Duration, density and the unchanged split

The original split remains `train=min(180,2*N//3)` and `development=min(60,N//6)` over valid recorded dates after clock exclusions. Thus **180 source dates give 120 training dates**; **270** are needed for the full 180 training dates. ECO's documented envelope permits at most **163**, not 180, training dates. Installing a sensor for a year is not proof of 270 usable dates.

The fixed 70,212 target corresponds to about **390.07 qualified outcomes per training date** only when there are 180 training dates. Public whole-campaign averages, electrical sampling frequencies, participant-days, household totals and timestamp resolution are not training outcome counts. A maximum of 1,440 spaced cutoffs per recorded date is a loose necessary capacity bound, not observed yield or a sufficient feasibility gate.

To verify the arithmetic, the audit freezes conditional regular-grid cases at 120/163/180 training days and ten cadences. Every case assumes exactly one **native binary event** at each grid time, no missing/duplicate records, eight-event history, <=300-second gaps, the original >=60-second greedy clock and the first result in inclusive 300–600 seconds. These are timing witnesses, not training data, a new interface or actual source estimates.

| Conditional timing witness | Raw rows | Qualified arrivals | Within 5M rows and existing outcome/write caps? |
| --- | ---: | ---: | --- |
| 180 days, 1 second | 15,552,000 | 259,194 | **No**, row ceiling |
| 180 days, 20 seconds | 777,600 | 259,192 | Yes |
| 180 days, 60 seconds | 259,200 | 259,188 | Yes |
| 180 days, 180 seconds | 86,400 | 86,391 | Yes |
| 180 days, 240 seconds | 64,800 | 64,791 | Yes, but below target |
| 163 days, 60 seconds | 234,720 | 234,708 | Yes |
| 120 days, 60 seconds | 172,800 | 172,788 | Yes |

This shows that the target is arithmetically possible without a full 180-day fit, and that denser data can exceed the input budget. It does **not** show that SPHERE packets, ECO PIR or another source supplies such a grid. No source is downsampled, quantized, pooled across households or assumed complete. The hypothetical changed-write count is exactly three times arrivals; actual writes remain zero.

The formula charges the original warm-up clock: early cutoffs are consumed before eight history events exist, and terminal outcomes outside the half-open training window are unavailable. Sixteen one-day cadence cases agree exactly with the unchanged original extractor, including non-divisors of 60 and the 300/301-second history boundary. The all-binary formula is **not** an upper bound for arbitrary filtering; the earlier greedy-phase counterexample and safe timestamp method remain unchanged.

## Verification and decision

All **215** related tests pass. They verify scalar/extractor equality, original split arithmetic, insufficient duration without household pooling, dense-stream row failure, refusal to substitute electrical totals for PIR evidence, source/evidence identity checks, frozen page/search/evidence/payload budgets and preservation of all 338 earlier evidence artifacts and registered code. Fresh-process scalar/result replay matches exactly and performs no browser or dataset access. Missing facts remain unknown; no quality, class, recurrence, opposing-history or independent-validation gate is waived.

Fixtures are `data/processed/benchmark_fixtures/source_metadata_feasibility_*_v1.json`: selection, transcribed evidence, assessment inputs, code registration, execution marker, result and replay receipt. Replay is idempotent and preserves the receipt:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/eval/source_metadata_feasibility.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_source_metadata_feasibility.py
```

The finite review closes at its budget with `no_acquisition_ready_source_within_registered_metadata_review`. No CASAS candidates are added. Original thresholds, exact current inputs, support/class/day/evidence conditions, earlier negatives and all independent-period seals remain unchanged. This audit supplies neither a learner negative nor evidence of SNN advantage.

🟢 [Done] The separately preregistered [SPHERE/ECO contract audit](R2_SPHERE_ECO_METADATA_AUDIT.md) finds both documented streams incompatible with the unchanged raw Kitchen ON target. It preserves unknown household PIR coverage/density/package/terms, every earlier result and all seals, and closes without payload acquisition or release substitution. This earlier feasibility result is not overwritten.

⭕️ [Pending] New dataset acquisition requires a compatible justified source/task contract, scoped identity/access/coverage/size metadata and all original causal/quality/resource gates. No automatic source or CASAS expansion or metadata retry follows these closed exclusions; metadata capacity is not positive SNN evidence.

日本語: 取得前に4候補と調査総予算を登録しました。TIHMは公表期間不足、IDEALの確認済み高速データは形式・全体サイズが条件に合わず、SPHEREとECOは世帯別の有効期間・PIR密度・取得サイズが未確定です。本体取得へ進める候補は確定していません。元の分割では訓練180日に270有効記録日が必要で、ECOは公表範囲から訓練最大163日です。条件付きの密度計算を元の抽出器と照合し、215テストと別プロセス再現が成功しました。本体取得・学習・CASAS追加・封印解除は0で、既存の負の結果を維持します。

简体中文：在采集前注册四个候选及调研总预算。TIHM公开期间不足；IDEAL已确认的高速流在格式和整体体积上不符；SPHERE、ECO的逐户有效期间、PIR密度和采集体积尚未确定，没有可直接采集的候选。原划分中180个训练日需要270个有效记录日，ECO公开范围最多允许163个训练日。条件密度计算与原提取器一致，215项测试及新进程重放通过。本体采集、学习、CASAS扩充和解除封存均为0，保留既有负结果。

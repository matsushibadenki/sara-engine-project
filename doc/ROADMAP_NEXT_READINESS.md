# Independent task and publisher pilot readiness

## Independent delayed-outcome task

The REFIT Smart Home dataset (DOI 10.17028/rd.lboro.2070091.v1) is a candidate for a new task. The catalogue identifies a 2013–2015 study of 20 UK households with motion and door sensors and measurements before and after installation. The project span does not establish continuous coverage for any household. [University catalogue](https://www.research.ed.ac.uk/en/datasets/refit-smart-home-dataset/)

A hypothesis to review is whether the order of recent motion events predicts a later opening on a separately recorded door channel when the current sensor state is the same. First verify that the released files contain synchronized motion and door channels with stable household and sensor identities, explicit coverage and missingness, and usable timestamps. The repository README documents the join from time-series variable IDs to the building survey XML. It places smart-device installation in June or October 2014 and collection end in April 2015. Even granting the first day of each installation month and the last day of April, the inclusive calendar ceilings are 334 and 212 days respectively. These are ceilings on calendar span, with no guarantee of overlap, continuity or usable development days. Earlier building survey dates cannot extend coverage of these smart devices.

The deposit lists CC BY 4.0 and a 98,567,363-byte compressed time-series archive. The README reports 25,312,397 rows across all variables. Neither figure bounds expanded bytes or establishes motion arrivals or door coverage for an individual household. This review retrieved only public metadata and the README. At that stage the survey XML and observation archive were unopened. [Loughborough deposit](https://repository.lboro.ac.uk/articles/dataset/REFIT_Smart_Home_dataset/2070091)

**Decision:** retain REFIT as a candidate. The next bounded check is the household-to-sensor mapping and paired coverage, using source metadata first. Where continuity cannot be established, an absent door opening cannot be classified as a negative outcome. No fixed household roster or archive acquisition is justified by the aggregate figures alone.

**Survey inventory:** The [bounded stdin checker](../scripts/eval/refit_metadata_inventory.py) was verified against the published 1,755,414-byte survey XML. It reports 90 Motion variables, 56 Open/Closed variables, and 15 of 20 buildings containing both types. Open/Closed may describe a window or a door. The counts establish co-location at building level. The output keeps coverage unverified because variable presence does not establish synchronized uptime or distinguish silence from outages. The observation archive stayed unopened. A follow-up structural check found all 56 contact variables under building-level sensors, with no opening-type link on those sensor records. All 146 motion and contact variables lack an explicit missing-data flag. The public survey therefore leaves door identity and continuous paired coverage unresolved. Keep REFIT on hold until a source-provided door mapping and coverage evidence are available. Prioritize an alternative with documented outcome identity and observation coverage if that evidence cannot be obtained.

Before acquisition, fix the household roster, acquisition and state budgets, chronological development split and outcome horizon. Require coverage through the entire horizon for both classes. Keep the established current-state definition and support thresholds, and verify in development that identical current states recur across distinct histories with adequate support and both outcomes. Compare against the strongest current-state control and an event-driven scalar with the same delayed outcomes and actual local writes. Stop before scoring if any gate fails. The proposed door outcome is a separate hypothesis. The original Kitchen ON outcome and its 70,212-arrival gate retain their recorded status. [Decision gate](R2_MECHANISM_DECISION_GATE.md) · [Prior task evidence](ROADMAP_HISTORY.md#existing-research-queue)

## REFIT alternatives — 2026-11-14

This screen covers two independent residential releases and uses only the authors' papers and deposit listings. Advance a candidate only with a physical outcome mapping and evidence of paired channel observation throughout each eligible outcome window. Keep the 180-day development requirement, the existing current-state and support gates, and the established scalar comparison at matched actual local writes. The original Kitchen ON definition and its 70,212-arrival gate still apply to that original task.

| Candidate | Verified metadata | Decision |
| --- | --- | --- |
| Winterberger et al., Zenodo v1 | A separate reed contact at the main entrance of some homes records open or closed. PIR reports motion events. The study spans 359 days across 62 installations, with no event sensors in Switzerland. The authors report server, gateway and sensor outages and staggered installation. Event archive is 6.8 MB as listed. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12145531/) · [Deposit](https://zenodo.org/records/14243471) | First choice for a coverage follow-up. Published global extent and aggregate counts give no 180-day paired development guarantee. Require per-installation deployment dates and timestamped availability for both channels. Verify reuse terms and fix the acquisition budget before obtaining observations. |
| Residential homecare, Data 11(9), 233 | Authors document door-contact streams and per-participant spans, including a wholly absent contact stream and a roughly 11-day interruption on another. Table 4 reports a maximum actual span of 34.0 days across ten participants. [Paper, Table 4](https://www.mdpi.com/2306-5729/11/9/233#table4-data-11-00233) | Exclude at the duration gate. Combining several households cannot satisfy the single-household development requirement. |

No candidate qualified. A periodic reading on the separate environmental pipeline cannot certify availability of the PIR or reed contact, and gaps between contact events cannot establish either outage or a no-opening outcome. If adequate availability evidence is obtained, fix the development roster and delayed outcome window before checking exact current-state recurrence, history support and both classes. No archive was acquired. Existing negative results and sealed test periods remain in force.

日本語：玄関ドアの結果が明記された欧州4か国データを追跡候補に選定。世帯別の連続観測と180日の開発期間は未確認。期間が最長34日の在宅ケアデータは除外。

简体中文：将明确记录入户门开闭的四国数据列为首选跟进对象。各户成对观测覆盖及180天开发期尚未确认。最长34天的居家护理数据未达到时长门槛。

### Four-country coverage follow-up

Reviewed the authors' data format, acquisition and validation sections, limitations, and the v1 deposit file listing. The deposit lists only event and periodic observation archives. No separate public installation schedule or availability log was identified in these materials. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12145531/) · [Deposit v1](https://zenodo.org/records/14243471)

- **180-day development period — unverified.** The 359-day project extent and the anonymized installation IDs do not give installation or removal dates for the paired PIR and entrance contact at any household. The authors report staggered installation and no event sensors at Swiss sites. No household can be certified from this evidence.
- **Coverage of outcome windows — unverified.** The documented event schema contains timestamp, installation, country, room, sensor type and value. It has no availability field. The authors report interruptions at the devices, gateways and server without providing dated intervals by installation. The PIR spot checks used care visit diaries for a subset of households and do not establish continuous PIR or door coverage. An event confirms a reported transition at its timestamp. It does not resolve subsequent silence. Environmental samples come through a separate path and cannot certify the event path.

**Decision:** keep this candidate on hold. Reassess only with a source-provided installation schedule keyed to anonymized IDs and dated availability evidence covering both devices and their delivery path. Intersect the documented intervals with the development split and require coverage for the full history and outcome windows. Treat unverifiable windows as missing. Both archives and their previews stayed unopened. No threshold, negative finding or test seal changed.

日本語：公開資料では世帯別設置期間と両センサーの稼働区間を確認できず、180日の開発期間と欠測を区別できる観測範囲は未確認。

简体中文：公开资料未证实各户安装时间和双传感器可用区间，180天开发期及可区分缺测的观测覆盖仍未确认。

## Independent source screen with household uptime

**UK-DALE, January 2015 release.** Prioritized because Table 1 gives household observation dates and mains uptime separately. House 1 first recorded on 2012-11-09, finished meter installation on 2013-04-12 and last recorded on 2015-01-05. The paper reports 786 days of total span and 655 days of aggregate mains uptime. The other houses report 140, 36, 155 and 131 days of mains uptime. Only House 1 clears a necessary aggregate-uptime screen for a mains-based 180-day task. The published mean dropout rate excludes large gaps. The figures establish neither 180 eligible days in a frozen development split nor coverage of a particular appliance channel during each history and outcome window. [Original paper, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4432654/) · [Release notes](https://jack-kelly.com/data/index.html)

**Outcome decision.** A dedicated appliance submeter could support a separate future electrical-activation hypothesis if its channel, activation rule and missingness semantics were fixed before acquisition. The original paper documents electricity meters, so this source cannot establish the PIR-to-entrance-opening outcome or the original Kitchen ON event. Aggregate mains uptime cannot certify a submeter outcome, and an unobserved sample cannot be counted as an inactive appliance. UK-DALE stays unqualified under the existing task conditions.

**Outcome-compatible lead.** The SHAPES household deposit describes motion and magnetic sensors, but its public listing does not provide household deployment dates, dated channel availability or a mapping that distinguishes the entrance from other doors or windows. The listing alone cannot establish 180 development days or the delayed outcome. The public listing was checked without opening its archive or preview. [SHAPES deposit v1](https://zenodo.org/records/14832143)

Neither candidate advanced. Require a published household-level channel map and dated paired coverage before fixing a development roster. Recurring current states, contrasting histories, both development classes and the existing support thresholds remain required. No observations were acquired. The 70,212-arrival target for the original Kitchen ON task, previous negative results and test seals remain in force.

日本語：UK-DALEは世帯別の期間と主電力計の稼働日数を確認できるが、玄関ドア結果は観測対象外。SHAPESは動体・開閉センサーを含むが、世帯別の設置期間と稼働範囲は未確認。

简体中文：UK-DALE 公开了各户记录时段及总电表运行天数，但未记录入户门结果。SHAPES 包含运动和磁性传感器，仍缺少逐户安装日期及可用区间的公开证据。

## SHAPES follow-up and HOMESENSE priority

**SHAPES remains on hold.** The [household deposit](https://zenodo.org/records/14832143) lists an observation archive without a separate deployment or uptime supplement. The related [Smart Mirror validation paper](https://mdpi-res.com/d_attachment/sensors/sensors-21-07938/article_deploy/sensors-21-07938.pdf) reports a two-month validation in March–April 2021 and combines door and window counts. That validation falls short of 180 days, and the reviewed sources do not link its installation to a specific household in the deposit. The later [anomaly-detection paper](https://pure.qub.ac.uk/en/publications/unsupervised-convolutional-neural-networks-using-home-presence-se/) and [MIRATAR release](https://zenodo.org/records/15800926) concern a separate 50-home collection and synthetic anomaly labels. They supply no basis for certifying the SHAPES deposit's per-household deployment dates, entrance mapping or paired PIR/contact uptime.

**Prioritize HOMESENSE for metadata follow-up.** The independent Dutch release documents a 366-day publication grid across 11 dwellings and per-value flags that distinguish observed, interpolated, reset-imputed and missing values. Its [release preparation code](https://zenodo.org/records/21687970/files/preprocess_release.py?download=1) documents aggregation of categorical states onto the 10-minute grid. The grid length is a calendar ceiling. Public materials reviewed did not establish a specific home's front-door mapping, 180 development days with paired observed coverage, or whether an opening and closing within one bin survive aggregation. An observed flag proves support for the published value. It does not certify every instant in the bin or show that no transition occurred. [Data descriptor](https://www.nature.com/articles/s41597-026-08321-w) · [Deposit](https://zenodo.org/records/21687970)

Before advancing, require a published channel map and development-only paired coverage summary for the full history and outcome windows. If exact opening events cannot be recovered, any grid-defined outcome needs a separate hypothesis and registration under the existing gates. No observation archive, CSV or data preview was opened. The 180-day gate, current-state and support conditions, Kitchen ON target, negative results and test seals retain their recorded status.

日本語：SHAPESは根拠不足で保留。欠測フラグが公開されているHOMESENSEを優先するが、世帯別の180日と玄関開放の識別は未確認。

简体中文：SHAPES 证据不足，继续暂缓。优先核查公开质量标记的 HOMESENSE，各户180天覆盖和入户门开启识别仍待确认。

### HOMESENSE release-code audit

Reviewed the v1.0.0 deposit listing, data descriptor and public preprocessing script (SHA-256 `f0103df987df4b39ec8dd979b41dca76fced18d4f24d12cc0bc0654e3266740f`). The script was inspected as text. Household CSVs and previews stayed unopened. [Deposit](https://zenodo.org/records/21687970) · [Descriptor](https://www.nature.com/articles/s41597-026-08321-w) · [Preprocessing code](https://zenodo.org/records/21687970/files/preprocess_release.py?download=1)

- **Entrance mapping — unverified.** The code uses a location from the input manifest and identifies the reed-contact channel. The reviewed public documentation gives no household-by-household correspondence between those channels and physical entrance doors.
- **180 paired development days — unverified.** The common 366-day grid includes missing and filled bins. The reviewed material lacks a dated household-level intersection of observed PIR and entrance-contact coverage. AM307 PIR uses maximum reported status without gap fill. WS202 can fill eligible internal gaps with zero. Such zeros are distinct from observed reports.
- **Within-bin opening — lost in general.** The digital contact retains the last valid state per 10-minute bin. The current contact gap limit cannot fill a whole empty bin. QC therefore distinguishes a bin with a valid contact report from a missing bin, but a closed-only report sequence and an open-then-closed sequence produce the same observed closed output. Within-bin order and event time are lost, and an observed bin does not establish uninterrupted coverage.

**Decision:** hold HOMESENSE for the exact entrance-opening hypothesis. A separately registered grid-state hypothesis would need its own outcome definition plus a source-backed entrance map and paired development coverage. Keep the 180-day requirement, current-state and support rules, original Kitchen ON target, negative results and test seals.

日本語：実測と欠測は区別できるが、10分枠内で開いて閉じた事象は失われる。玄関の対応と180日の実測重複も未確認のため保留。

简体中文：质量标记可区分实测和缺测，但同一十分钟内先开后关的事件会丢失。各户入户门映射及180天共同实测覆盖尚未确认。

## Native-timestamp entrance sensor screen

**Prioritize Örebro E-care@home v1 for a coverage follow-up.** Table 3 maps D01 to the entrance reed contact and M01–M06 to room PIR sensors at the same residence (location 711). The release describes native one-second timestamped state samples, including explicit open and closed observations. Consecutive valid samples can bracket a sampled closed-to-open transition. They cannot recover transitions that open and close between samples. An absent sample is unobserved and cannot be classified as closed. This source supplies native samples, with no separate exact transition timestamp established. [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7758366/) · [Deposit](https://data.mendeley.com/datasets/t9n68ykfk3/1)

The campaign ran from 2020-02-26 through 2020-08-26, a ceiling of 183 calendar dates including both endpoints. Sensors were installed on a rolling basis. Figure 2 summarizes recorded days by sensor. Separate day totals cannot establish their dated intersection, within-day gaps, or the portion available before a sealed test boundary. No dated paired PIR/D01 availability record was verified in the materials reviewed. The 180-day development gate and coverage of complete history and outcome windows remain unverified.

**UMass Smart* 2013 Home A does not advance.** Table 2 describes timestamped motion and contact events, but the original deployment identifies the documented contacts as refrigerator and freezer doors. The release listing does not establish an entrance channel or a dated paired-uptime record. [Original paper](https://www.ecs.umass.edu/irwin/sustkdd12.pdf) · [Release listing](https://traces.cs.umass.edu/docs/traces/smartstar/)

No source qualified for the existing exact entrance-opening task. Reassess E-care@home only with dated development-only coverage for D01 and the required PIR channels and evidence that the outcome timing meets the registered definition. A sampled-state transition would require a separate hypothesis if exact events are required. No data files or previews were opened. Existing thresholds, negative results and test seals remain in force.

日本語：玄関センサーを特定できるE-care@homeを優先。公開資料では180日の同時稼働と正確な開閉イベント時刻は未確認。

简体中文：优先跟进可识别入户门传感器的 E-care@home。公开资料尚不能确认180天共同覆盖及精确开闭时间。

### E-care@home supplement follow-up

The original article XML indexes Figure 2 as a sensor-level day-count figure and contains no separate supplementary-material entry. The v1 deposit and article did not establish the D01 or PIR installation dates, dated outage intervals, or shared recording dates. Aggregate counts cannot certify paired coverage within development. The campaign ceiling remains 183 calendar dates, and the native one-second state samples bound a detected transition to the interval between reports. Exact opening time and openings entirely between samples remain unobservable under this format. Keep E-care@home on hold. [Article and Figure 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7758366/#fig0002) · [Deposit](https://data.mendeley.com/datasets/t9n68ykfk3/1)

The alternative screen prioritized releases with timestamped contacts. Mi4A-HomeData documents timestamped contact and PIR states, but its release dates (3–16 April 2024) cap development at 14 calendar dates. The independent DomoCare study documents entrance open/close events, PIR and sensor-day validity methods, but access to original observations requires ethics approval and its public aggregates cannot certify dated paired coverage. Neither advances. Prioritize a fresh independent release that publicly documents a dated household-level coverage intersection and original entrance-event timestamps. No observation files or previews were opened. The 180-day requirement, outcome definition, other eligibility conditions, previous negative results and test seals stay fixed. [Mi4A deposit](https://zenodo.org/records/19182995) · [DomoCare study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9381599/)

日本語：E-care@homeは同時稼働と正確な開放時刻の根拠不足で保留。代替2候補も成立条件を満たす公開証拠がない。

简体中文：E-care@home 暂缓，缺少共同覆盖和精确开门时间的证据。两项替代来源也未通过公开证据筛查。

### ORCATECH and Orange4Home screen

Prioritized ORCATECH/CART because its public protocol documents room PIR and egress-door contacts, timestamped hub records and monitoring for outages. An ORCATECH-based protocol plans up to 40 weeks per participant. That planned duration cannot certify 180 eligible development days at any household. The reviewed public materials do not give a dwelling-level main-entrance channel map, confirm that exported timestamps preserve original contact-event times, or provide dated paired uptime and outage intervals. OHSU lists a request process for research data. Public evidence leaves the delayed outcome and duration gates unverified. [Study protocol](https://formative.jmir.org/2024/1/e53192) · [OHSU data resources](https://www.ohsu.edu/oregon-center-for-aging-and-technology/data-resources)

Orange4Home independently fails the duration screen. The authors' release dates, January 30 through February 24, 2017, allow at most 26 calendar dates even before accounting for recording hours or a sealed split. [Author's release documentation](https://github.com/JCumin/jcumin.github.io/blob/main/datasets.md)

No source advances. Keep ORCATECH/CART on hold pending public evidence that ties a household's entrance channel and original event times to dated paired coverage. Silence in an outcome window needs coverage evidence before it can be counted as no opening. No observation files or previews were opened. The 180-day gate, existing eligibility conditions, negative results and test seals remain in force.

日本語：ORCATECH/CARTは世帯別の同時稼働・欠測と元時刻を確認できず保留。Orange4Homeは最大26暦日で期間条件を満たさない。

简体中文：ORCATECH/CART 缺少逐户共同运行、缺测区间及原始时间戳的证据。Orange4Home 最多覆盖26个日历日，未达到时长门槛。

### CART dictionary follow-up and TIHM screen

**CART remains on hold.** The February 2020 sensor activity dictionary covers watch, computer-use and scale fields. It does not supply the dwelling-to-entrance mapping or dated PIR/contact availability. OHSU's home guide identifies a front-door use case, while the technical protocol describes internal outage monitoring. Neither publishes dwelling-level paired operating days or outage intervals. The NYCE technical specification documents state changes and two-hour heartbeat reports. It explicitly dates both PIR and contact reports at hub arrival. That timestamp has no documented bound to the physical transition in the reviewed materials. The device-origin timestamp specification for the TimerCap applies to pillbox doors. The reviewed public evidence cannot certify 180 development days or exact opening timestamps. [CART dictionary](https://www.ohsu.edu/sites/default/files/2020-02/CART%20Study%20-%20Sensor%20Activity%20Data%20Dictionary%20-%20Feb%202020.pdf) · [CART Home](https://www.ohsu.edu/collaborative-aging-research-using-technology/cart-home) · [Technical protocol](https://www.ohsu.edu/sites/default/files/2019-11/CART-VA-Protocol-Template.pdf) · [NYCE specification](https://api.orcatech.org/orcatech/v0.9/docs/items.html#nyce)

**Prioritize TIHM for metadata follow-up.** The independent release documents a main-entrance sensor, hallway and living-room PIR, a Front Door location in the Activity schema and second-resolution event timestamps. Figure 2 identifies a collection-server failure on 2019-06-14. The reported mean of about 50 days per participant is insufficient to determine the longest eligible household span. The public materials reviewed provide no household-level paired coverage calendar or complete outage intervals. They also do not establish that the exported front-door records preserve separate opening and closing transitions at source event times. The next check requires those three items before any development roster can be fixed. [TIHM descriptor](https://www.nature.com/articles/s41597-023-02519-y) · [TIHM deposit](https://zenodo.org/records/7622128)

No source qualified. Observation files and previews stayed unopened. The existing 180-day requirement, outcome definition, current-state and support gates, negative findings and test seals retain their recorded status.

日本語：CARTの玄関センサー時刻はハブ受信時刻。同時稼働は未確認。次の公開資料確認はTIHMを優先する。

简体中文：CART 的门传感器采用网关接收时间，共同运行区间未确认。优先核查 TIHM 的公开元数据。

### TIHM release follow-up

**Exclude TIHM at the duration gate.** The authors' Activity schema dates the release from April 1 through June 30, 2019, inclusive. This caps the entire release at 91 calendar dates before any development split or coverage exclusions. The descriptor identifies a collection server failure on June 14 but gives no dated household-level intersection of PIR and entrance-contact operation or complete outage intervals. [Release schema](https://github.com/PBarnaghi/TIHM-Dataset/blob/main/README.md#activity) · [Descriptor](https://www.nature.com/articles/s41597-023-02519-y)

The Activity schema documents participant, location and date fields. The descriptor identifies the entrance location and second-resolution event times. Neither document establishes a direction field separating open from close or specifies whether event time originates at the sensor or is assigned downstream. A front-door record alone therefore cannot certify a delayed opening label. The outcome remains unverified independently of the duration failure.

**Alternative screened.** Prioritized the independent LACE House release because its public sensor map distinguishes the entry contact from the fridge contact and documents room multisensors. The companion study reports 31 days of readings, so it also fails the 180-day gate. Continue the search for a source with sufficient release span, a directional entrance event schema and dated paired coverage. [LACE release](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lace-house-domotic-sensors-dataset/) · [Companion study](https://www.research.unipd.it/bitstream/11577/3494789/4/Fernandez-Carmona2024.pdf)

No observation files or previews were accessed. Existing eligibility rules, negative results and test seals remain in force.

日本語：TIHM公開版は最大91暦日、LACE Houseは31日で期間条件を満たさない。TIHMの開閉方向・時刻由来・同時稼働範囲は未確認。

简体中文：TIHM 公开版本最多覆盖91个日历日，LACE House 为31天，均未达到时长门槛。TIHM 的开闭方向、时间来源和共同运行范围尚未确认。

### TigerPlace and Tokyo smart-room screen

**TigerPlace remains unqualified.** Prioritized this independent deployment because researchers documented monitoring over years and sensors at the front door and in rooms. The multi-year research description does not identify a released household trace with directional contact events or define where its timestamps originate. A separate TigerPlace paper reports 3–48 months for its Kinect cohort. Those durations concern depth-sensor recordings and cannot verify PIR and entrance-contact coverage. The reviewed public material gives no dwelling-level intersection of operating days or dated outage intervals for those two channels. Neither 180 eligible development days nor a delayed entrance-opening outcome can be certified. [Deployment paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3001103/) · [Longitudinal research overview](https://c2ship.missouri.edu/wp-content/uploads/2024/01/TigerPlace-An-Innovative-Educational-and-Research-Environment.pdf) · [Kinect cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC5272917/)

**Tokyo smart rooms fail the duration gate.** The data descriptor documents PIR and open/closed sensors but reports individual recording stays from one week to two months. Combining stays from different residents cannot meet the same-household development requirement. The duration failure rules out advancement even if channel direction, time provenance and coverage are documented separately. [Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC9898591/)

Keep the independent task pending. Advance a release only with a source-backed residential entrance channel map, directional opening and closing timestamps with documented provenance, and dated paired availability and outage intervals sufficient for 180 development dates and complete history and outcome windows. No data files or previews were opened. Existing eligibility rules, negative findings and sealed test periods remain in force.

日本語：TigerPlaceでは年単位の研究を確認したが、玄関接点とPIRの同時稼働や開閉時刻の由来は未確認。東京のスマートルームは個人あたり最長2か月で除外。

简体中文：TigerPlace 有多年研究记录，但入户门与 PIR 的共同覆盖及事件时间来源仍未确认。东京智能房间每位住户最长记录两个月，未达时长门槛。

### TigerPlace deployment follow-up

The TigerPlace sensor-network paper documents a ceiling-mounted PIR near the entrance to infer departures. It explains that residents often left entrance doors ajar, making magnetic contacts unreliable for that departure measure. Its typical floor plan does not link dwelling IDs to physical entrance contact IDs. The published motion density maps include different months for the same resident, but provide no dated paired PIR and entrance-contact coverage or outage intervals. The reviewed technical papers and linked public material provide no contact-event schema establishing separate open and close labels and the origin of their timestamps. Keep TigerPlace on hold. Neither 180 eligible development days nor the exact delayed entrance-opening outcome is established. [Sensor-network paper](https://c2ship.missouri.edu/wp-content/uploads/2024/01/Using-Sensor-Networks-to-Detect-Urinary-Tract.pdf) · [Motion density study](https://c2ship.missouri.edu/wp-content/uploads/2024/01/Density_Map_Visualization_from_Motion_Sensors_IEEE-1.pdf)

Prioritized an independent residential release with binary door sensing, the original van Kasteren home study. Its authors document 28 days of sensor data, so the release fails the 180-day gate even if every day were usable. Continue screening independent releases with a documented span of at least 180 development days before investing in event provenance and outage checks. [Original study](https://dare.uva.nl/id/1d2afcf6-c9a0-4a65-bb28-2646924685a6)

No observation files or previews were accessed. Existing eligibility rules, negative findings and sealed test periods remain in force.

日本語：TigerPlaceは玄関付近のPIRを確認したが、開閉イベントと同時稼働の証跡が不足。van Kasterenの原公開版は28日で除外。

简体中文：TigerPlace 证实入口附近装有 PIR，但缺少门开闭事件及共同覆盖证据。van Kasteren 原版仅28天，未达时长门槛。

### Longitudinal ambient sensing and ARAS screen

**Fritz and Cook longitudinal study.** Prioritized this lead because the authors report at least one year of collection per participant using room PIR and magnetic contacts on exterior doors and cabinets. The reported study span is compatible with a 180-day development period in principle. The authors removed unreliable sensors and excluded dates when all sensors failed to report. The public methods do not identify a released household trace with a physical entrance map, separate open and close values, original event timestamp provenance, or dated paired coverage and outage intervals. The publication does not establish a release independent of the existing CASAS evidence. Keep the study as an unqualified lead. Its deployment duration cannot certify 180 eligible development days or a delayed opening outcome. [Original study](https://nursing.jmir.org/2025/1/e69052)

**ARAS independent release.** The authors report a month per dwelling across two homes. Each dwelling fails the duration gate on its own. The combined duration cannot qualify a household and no channel or timestamp assumption can change that bound. [Original dataset paper](https://open.metu.edu.tr/bitstream/handle/11511/36652/index.pdf)

No candidate advanced. Require a documented independent release, a dwelling-level entrance channel map, directional opening and closing records with timestamp provenance, and dated paired availability sufficient for the existing 180-day gate and full history and outcome windows. No observation files or previews were accessed. The existing eligibility rules, negative results and test seals remain in force.

日本語：年単位の住宅観測研究では公開データの独立性・開閉時刻・同時稼働が未確認。ARASは各世帯約1か月で期間条件を満たさない。

简体中文：年度住宅观测研究未证实公开数据的独立性、开闭时间来源及共同覆盖。ARAS 每户约一个月，未达到时长门槛。

### Fritz and Cook supplement and provenance follow-up

**Keep the source on hold.** The publisher's article exposes no linked appendix with a deployment ledger, sensor dictionary or outage calendar. The article and the CASAS free-living deposit acknowledge the same two NIH awards, R01NR016732 and R01EB009675. This establishes a shared funding lineage. Household-level overlap remains unresolved because the reviewed publications provide no cohort-to-release crosswalk. The CASAS deposit's generic contact schema cannot establish how the study's physical entrance channels were mapped, how open and close were encoded, or where original timestamps were assigned. [Fritz and Cook article](https://nursing.jmir.org/2025/1/e69052) · [CASAS deposit metadata](https://zenodo.org/records/15708568)

The article reports monitoring for at least one year and describes exclusion of unreliable sensors and dates without sensor reports. Those statements provide no dwelling-specific dates or intervals for joint PIR and entrance-contact coverage. They cannot establish 180 eligible development days or distinguish a missing outcome window from a window with no entrance opening. Eligibility requires an independent provenance record, a directional entrance-event schema with timestamp provenance, and dated paired coverage through the full history and outcome windows.

**Independent alternative screened.** The Grenoble LIG VocADom@DOMUS release documents synchronized home automation recordings in an instrumented apartment. Its public duration table reports 14 hours, 51 minutes and 30 seconds across the listed participants, so it fails the 180-day gate. Continue to prioritize independent residential releases with a documented long deployment span. [Project corpus documentation](https://vocadom.imag.fr/index4df3.html?page_id=290)

Observation files and previews stayed unopened. Existing eligibility rules, negative findings and sealed test periods remain in force.

日本語：Fritz・Cook研究とCASASは助成番号が共通し、世帯単位の独立性と同時稼働範囲は未確認。VocADom@DOMUSは期間条件で除外。

简体中文：Fritz 和 Cook 的研究与 CASAS 共用资助编号，各户独立性与共同观测范围仍未确认。VocADom@DOMUS 未达到时长门槛。

### Minneapolis homedata release screen

The independent LinkNLearn release documents a single residence and distinguishes its front and back doors. Its README reports four months of door open/close data alongside multiyear environmental logs. The door stream therefore falls short of the 180-day development requirement even before reserving any test dates. Temperature and humidity coverage cannot extend the door stream. [Publisher README](https://github.com/LinkNLearn/homedata#included-within-dataset)

The public README establishes the physical front-door location and describes open/close data. It does not establish the released direction encoding or whether the timestamp is assigned at the sensor, gateway or database. It does not document a paired PIR stream, joint operating dates or dated outages. An interval with no reported opening cannot be classified as a negative example on that evidence. Exclude the release at the duration gate and record the other checks as unverified. [Publisher README](https://github.com/LinkNLearn/homedata)

No qualifying source was identified in this search. Observation files and previews stayed unopened. Existing eligibility rules, negative results and sealed test periods remain in force.

日本語：homedataは玄関・裏口の開閉を記録するが、対象期間は4か月で除外。元時刻とPIRとの同時稼働は未確認。

简体中文：homedata 区分前门与后门，开闭记录仅四个月，未达时长门槛。原始时间来源与 PIR 共同覆盖仍未确认。

### Overijssel IEBB dwelling screen

Prioritized the Overijssel IEBB study because it documents one year of monitoring in each of three dwellings, from December 2021 through November 2022. Section 2.1 reports recording door and window openings and closings on action. Table 1 documents 12-month availability in all three dwellings for living-room movement/light and the doors from the living room to the hall, storage and staircase. This supports a calendar span longer than 180 days for those channels. The table provides month-level availability and does not certify dated development coverage. [SHE2024 proceedings, section 2, Figure 1 and Table 1](https://pure.tue.nl/ws/portalfiles/portal/355569775/SHE2024_Book_of_Proceedings_Second_Edition_1.pdf)

**Hold at the entrance-outcome and coverage gates.** The published floor plan shows door and movement/light sensor placements, but the availability table covers bedrooms and living rooms and does not map a released channel to the physical entrance for any dwelling. The paper describes event-triggered collection without specifying the released open/close encoding, timestamp assignment point or original event resolution. It does not identify PIR hardware for the movement/light channel or provide a dated intersection of PIR and entrance-contact uptime and gaps. No public observation release or separate coverage manifest was verified. An eligible 180-day development period and a delayed opening label therefore remain unverified. Reassess only with those source records.

No observations or previews were accessed. Existing eligibility gates, negative results and sealed test periods remain in force.

日本語：IEBBは3世帯で1年の観測を確認。玄関イベントの元時刻、PIRとの共通稼働日と欠測区間が未確認で保留。

简体中文：IEBB 记录了三户一年的观测，入户门原始事件时间、与 PIR 的共同运行日期及缺测区间尚未确认。

### IEBB appendix follow-up

TNO report 2025 R11086, Appendix A, names the same three dwellings (HKJH, QIMU and WEJH) and confirms the 12-month campaign. The appendix reports window-opening analyses using environmental, movement/light and window-contact measurements. It supplies no dwelling-level entrance contact ID, PIR device specification, original door timestamp definition, or dated joint uptime and outage records. The existing proceedings describe action-triggered opening and closing but document interior doors in the availability table. Taken together, the public materials establish the campaign span but leave 180 eligible development days and complete delayed entrance-opening windows unverified. Keep IEBB on hold. [TNO report, Appendix A](https://publications.tno.nl/publication/34644624/pPpnNquM/TNO-2025-R11086.pdf#page=21) · [Proceedings, section 2](https://pure.tue.nl/ws/portalfiles/portal/355569775/SHE2024_Book_of_Proceedings_Second_Edition_1.pdf)

Screened the independent Home Monitoring System release because its public schema names a front-door signal and hallway motion. Its single 21-day period fails the duration gate. The six-minute rows also provide no verified original directional entrance events. Prioritize independent releases with a documented longer span and dated paired coverage. [Publisher README](https://github.com/MBJamshidi/HomeMonitoringSystemDataSet)

Observation files and previews stayed unopened. Existing eligibility rules, negative findings and test seals remain in force.

日本語：TNO付録で同じ3世帯の12か月観測を確認。玄関イベントの元時刻とPIRとの同時稼働は未確認。代替候補は21日で除外。

简体中文：TNO 附录确认同三户的12个月监测。入户门原始事件时间及与 PIR 的共同覆盖仍未确认。替代来源仅21天。

## Independent HTTPS publisher pilot

Use the existing authoritative evidence V2 runner. Before scheduling the pilot, record the independent operator and scope owner, approved HTTPS endpoint and publisher ID, TLS trust and certificate rollover plan, and whether the sequence store is single-host or backed by a reviewed transactional authority. Confirm with the operator how complete revisions and withdrawals are produced and agree on the expected cadence and outage behavior. Keep the approved configuration and trust material in the managed workspace.

### Live pilot acceptance procedure

Before connecting, have the scope owner confirm the exact HTTPS URL, publisher ID, scope, permitted trial questions and independent operator through an established contact channel. Record the trust source, expected hostname, and who approves certificate changes. For a private CA, compare the bundle fingerprint against the value supplied through that channel. Run the offline check with the planned configuration and report path. Reserve one sequence file for the approved publisher and scope, and reuse it across the trial launches. Agree on the trial window, expected availability, and any latency or cost limits before running. Record missing limits as unassessed.

Choose two disjoint trial questions, one for a record to revise and withdraw and one for an unaffected record. The CLI reports response kinds in configuration question order alongside the aggregate counts. Use both questions in one configuration to attribute the two outcomes to the same snapshot. Have the operator provide a dated list of expected sequences and record changes. Compare the actual approved answer with the anticipated value through a restricted operator check because the CLI report contains counts rather than answer text.

| Step | Evidence required for a pass |
| --- | --- |
| Initial publication and revision | The operator supplies complete snapshots with increasing sequences. The CLI reports `published` and the expected new sequence. The approved answer check confirms the revised value and the unaffected record. |
| Conditional refresh | The operator observes the expected validator and returns an empty page-zero 304. The CLI reports `publisher_not_modified` and keeps the sequence. Use a bounded multi-cycle run with a loaded snapshot to check answer continuity. A new process retains the sequence but does not reload the prior snapshot contents. |
| Withdrawal | The operator lists the removal and publishes a higher complete snapshot omitting the withdrawn record while retaining the unaffected record. Check `published`, the higher sequence, verified abstention for the withdrawn record and a verified answer for the unaffected record. Check both against the same published revision. |
| Replay and recovery | In the agreed trial scope, serve an earlier sequence as HTTP 200. After a fresh CLI launch, verify `publisher_rollback` and the previous on-disk value. Serve a higher valid snapshot on the next launch and verify `published`, with the accepted sequence matching the on-disk value. |
| Availability and cost | Record attempted and successful refreshes during the agreed window, request counts, response-body bytes and refresh latency. Compare measurements with the agreed limits. Record publisher-side work and transport overhead separately if evaluating complete-path cost. |

For the withdrawal check, use three refresh cycles in one CLI invocation. Coordinate the publisher to serve the complete pre-withdrawal snapshot first, the higher withdrawal revision second, and an empty conditional 304 third. On the third cycle, require the same per-question response kinds as the withdrawal cycle and the same accepted sequence in the report and on disk.

Mark the live pilot **pass** only when the trust review and every required row have matching operator and client evidence. Mark an observed mismatch **fail** and preserve its report and sequence state before investigating. Mark missing independent evidence or unassessed required measures **pending**. A local success establishes the client behavior exercised by the fixture. The local TLS rehearsal also covered abstention for an omitted record and a verified answer for the retained record on separate launches. Withdrawal completeness and independent operational evidence require the live checks above.

日本語：接続先と信頼元を事前確認し、改訂・撤回・304・再起動を公開者側の記録と照合する。証拠が不足する項目は保留とする。

简体中文：预先核实端点与信任来源，并将修订、撤回、304 和重启结果与发布方记录核对。证据不足的项目保持待定。

Use the following placeholder configuration as a starting point. Replace the values with approved settings and save it to `workspace/config/authoritative_external_pilot.json`.

```json
{
  "schema": "sara-authoritative-external-pilot-config-v1",
  "endpoint": "https://publisher.example/evidence",
  "publisher_id": "replace-with-reviewed-publisher-id",
  "scope": "replace-with-reviewed-scope",
  "language": "en",
  "aliases": [
    ["tank temperature", "tank:temperature"]
  ],
  "markers": ["temperature"],
  "ca_file": null,
  "timeout_seconds": 5.0,
  "max_bytes": 262144,
  "max_snapshot_lifetime_seconds": 300,
  "max_future_skew_seconds": 5,
  "max_fetch_seconds": 8,
  "sequence_state_path": "data/interim/authoritative/publisher-sequence.json",
  "questions": [
    "Report tank temperature."
  ]
}
```

🟢 [Done] Validate the operator configuration locally before attempting a live cycle:

```text
workspace/.venv310/bin/python scripts/eval/authoritative_external_pilot.py --config workspace/config/authoritative_external_pilot.json --check-config
```

This mode checks the same schema, limits, HTTPS endpoint, CA bundle and question routing as a live run. It validates the managed sequence and lock paths without opening or creating state files. Both CLI modes reject a report path that overlaps the configuration, sequence state or lock before connecting. Existing watermark contents and write access are checked during the live run. It prints only validation status and a question count. The mode exits before runtime initialization, HTTPS access or report creation. Publisher identity and trust still require the independent operator review.

The live CLI opens a temporary report in the destination directory before contacting the publisher. It replaces the previous report after the new JSON is fully written. A run or serialization exception leaves the prior report available. Publisher sequence persistence is independent of report completion.

Each refresh records the accepted sequence before and after the request and the deadline of any snapshot held in memory at refresh completion. Use the deadline to check that a warm 304 preserves expiry. A cold 304 can report a persisted sequence with a null snapshot deadline. Compare those values with the decision to distinguish a published revision from a 304 that preserves the accepted sequence. On a restart, the first cycle starts from the loaded sequence. Use the publisher's revision and withdrawal records to assess completeness.

🟢 [Done] The scripted lifecycle check applies sequence 1, preserves it on 304, starts a new runtime against the same on-disk watermark, rejects a replay of sequence 1 as rollback, and applies sequence 2. The restarted client uses sequence 1 as its conditional validator for both requests. This exercise uses an in-process transport and a new runtime instance. The independent HTTPS pilot remains pending.

🟢 [Done] A localhost HTTPS integration test ran five separate CLI processes against a simulated publisher and a shared sequence file. The CLI published sequences 2 and 3, preserved 3 on a conditional 304, rejected sequence 2 as rollback, and published 4 on the next launch. Each report's final accepted sequence matched the on-disk value. Each new process reported the previous persisted value before its request. The independent publisher trial remains pending.

🟢 [Done] The localhost HTTPS rehearsal now uses seven CLI launches. Sequence 4 answers both trial questions. Sequence 5 omits the tank record from the complete snapshot and the tank question yields verified abstention. Sequence 6 republishes the same reduced record set and the vent question yields a verified answer. The two questions were checked separately because the CLI reports aggregate response counts. This checks the two snapshots across process restarts. Live review still checks the operator's withdrawal record and the actual retained answer.

🟢 [Done] A three-cycle localhost HTTPS run applied sequence 7, applied withdrawal sequence 8 and received an empty conditional 304 for sequence 8 in one CLI process. Both questions yielded verified answers on 7. On 8 the tank question yielded verified abstention and the vent question still yielded a verified answer. The 304 retained both response kinds and the accepted sequence. The persisted sequence matched the report at 8. The report includes response kinds by question position, with question and answer text omitted.

🟢 [Done] The localhost HTTPS rehearsal used the production runtime with a controlled clock to receive an empty 304 one second before withdrawal sequence 8's original expiry. The tank answer remained withheld and the vent answer remained available. One second after the original expiry, both questions yielded verified abstention. The in-memory and persisted accepted sequence remained 8. The controlled-clock check uses the runtime directly.

🟢 [Done] After withdrawal sequence 8 expired, the same runtime rejected a simulated sequence 7 response with a current validity window as `publisher_rollback`. Both questions still yielded verified abstention and the accepted sequence stayed at 8 in memory and on disk. A fresh sequence 9 retained the withdrawal and restored only the vent answer. The accepted sequence advanced to 9 in memory and on disk on that publication.

🟢 [Done] Three fresh CLI processes reused the on-disk state from withdrawal sequence 8, with the runtime clock set one second past its original expiry. The first request received an empty conditional 304 for sequence 8. Both questions yielded verified abstention and the report and disk retained sequence 8. The next process rejected a time-valid sequence 7 and verified the same abstentions and persisted value. The final process started at 8, applied sequence 9 carrying the withdrawal, and restored only the vent answer. The report and disk both advanced to 9.

🟢 [Done] The localhost HTTPS CLI reports the published withdrawal deadline unchanged on the next empty 304 within the same process. After expiry, a fresh CLI process receiving an empty 304 reports a null snapshot deadline and verified abstention for both questions. The report and on-disk sequence remain 8 across both 304 responses. During the first 304, the unaffected vent still yields a verified answer.

Run one explicit cycle:

```text
workspace/.venv310/bin/python scripts/eval/authoritative_external_pilot.py --config workspace/config/authoritative_external_pilot.json --cycles 1
```

Record operator-side revision and withdrawal evidence alongside the client decisions, requests, response-body bytes and complete-refresh latency. An unchanged 304 preserves the original expiry. Report authority review, revision completeness and availability separately from client timing and byte counts. The runner measures response bodies, so full-path costs require separate evidence for publisher-side work and transport overhead. Use the existing bounded pilot conditions and preserve the disabled promotion and automatic-default flags. [Pilot runner](../scripts/eval/authoritative_external_pilot.py)

### Pilot observation summary

🟢 [Done] The runner adds `observation_summary` to its JSON report. It separates accepted refreshes (`published` or `publisher_not_modified`) from the fraction of question checks yielding `verified_answer`. An accepted cold 304 can still yield abstention; an intended withdrawal can also reduce the answer rate. Neither rate establishes answer correctness or independent service availability.

The invocation summary also groups response counts by zero-based question position. This lets the operator track abstention for a withdrawn record independently of answers for retained records across the invocation. Match positions to the configuration used for that run.

Request counts, response-body bytes and refresh duration include measured-cycle counts and observed totals. Complete totals are null when any cycle lacks the measurement. Maximum duration covers measured cycles only. Durations cover refresh processing, excluding subsequent question checks. These summaries describe the bounded invocation, not an agreed operational window. Authority review, revision completeness, independent availability, agreed limits and complete-path cost remain explicitly unassessed. Promotion and automatic-default flags remain disabled.

日本語：更新受理率と回答率を分離し、通信量・処理時間の未計測を明示する集計を実装。独立運用の可用性や合格条件は未評価のまま保持する。

简体中文：已实现分别汇总刷新接受率与回答率，并明确标注通信量及耗时的缺失测量。独立运行的可用性及验收条件仍未评估。

🟢 [Done] The three-cycle localhost HTTPS withdrawal rehearsal reconciles the invocation summary with the cycle records. The tank yields one verified answer and two abstentions. The vent yields three verified answers. All three refreshes are accepted and four of six question checks yield verified answers. Three requests were recorded. The two published snapshots have positive response-body byte counts and the 304 has zero. The test reconciles measured-cycle counts, complete and observed totals, and maximum refresh duration with the individual records.

🟢 [Done] The live runner supports a bounded inter-cycle wait. For a coordinated three-cycle withdrawal rehearsal, use `--cycles 3 --interval-seconds 10` if the operator needs ten seconds between cycles. The interval defaults to zero and accepts whole seconds through 60. The report records the configured interval. Refresh duration metrics exclude these waits and the questions that follow each refresh.

🟢 [Done] In a localhost HTTPS rehearsal, the wait callback advanced both test clocks by ten seconds per interval. During the first wait the publisher replaced sequence 7 with sequence 9 withdrawing the tank record. During the second wait it enabled conditional 304 for sequence 9. The refresh reports and persisted sequence agreed at 7, 9 and 9. The tank produced one answer and two abstentions. The vent produced three answers. The controlled refresh durations were 0.25, 0.5 and 0.125 seconds and the report totaled 0.875 seconds while the simulated waits totaled 20 seconds.

The local runner tests cover its conditional refresh, observation summaries and configuration checks. Independent operation remains pending because the operator, endpoint and approved trust material were unavailable.

日本語：REFIT は別の遅延結果課題の候補です。取得前にCC BY 4.0 と圧縮サイズを確認済みで、世帯別の同期・観測範囲は未確認です。HTTPS 試験は独立運用者・接続先・審査済み信頼情報の確定待ちです。

简体中文：REFIT 是另一项延迟结果任务的候选来源。已确认 CC BY 4.0 和压缩大小，仍需核实各家庭的时间同步和观测覆盖。HTTPS 试点仍需独立运营方、端点和经审核的信任材料。

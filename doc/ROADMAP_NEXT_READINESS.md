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

## Independent HTTPS publisher pilot

Use the existing authoritative evidence V2 runner. Before scheduling the pilot, record the independent operator and scope owner, approved HTTPS endpoint and publisher ID, TLS trust and certificate rollover plan, and whether the sequence store is single-host or backed by a reviewed transactional authority. Confirm with the operator how complete revisions and withdrawals are produced and agree on the expected cadence and outage behavior. Keep the approved configuration and trust material in the managed workspace.

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

Run one explicit cycle:

```text
workspace/.venv310/bin/python scripts/eval/authoritative_external_pilot.py --config workspace/config/authoritative_external_pilot.json --cycles 1
```

Record operator-side revision and withdrawal evidence alongside the client decisions, requests, response-body bytes and complete-refresh latency. An unchanged 304 preserves the original expiry. Report authority review, revision completeness and availability separately from client timing and byte counts. The runner measures response bodies, so full-path costs require separate evidence for publisher-side work and transport overhead. Use the existing bounded pilot conditions and preserve the disabled promotion and automatic-default flags. [Pilot runner](../scripts/eval/authoritative_external_pilot.py)

The local runner tests cover its conditional refresh and configuration checks. Independent operation remains pending because the operator, endpoint and approved trust material were unavailable.

日本語：REFIT は別の遅延結果課題の候補です。取得前にCC BY 4.0 と圧縮サイズを確認済みで、世帯別の同期・観測範囲は未確認です。HTTPS 試験は独立運用者・接続先・審査済み信頼情報の確定待ちです。

简体中文：REFIT 是另一项延迟结果任务的候选来源。已确认 CC BY 4.0 和压缩大小，仍需核实各家庭的时间同步和观测覆盖。HTTPS 试点仍需独立运营方、端点和经审核的信任材料。

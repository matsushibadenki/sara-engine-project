# Independent task and publisher pilot readiness

## Independent delayed-outcome task

The REFIT Smart Home dataset (DOI 10.17028/rd.lboro.2070091.v1) is a candidate for a new task. The catalogue identifies a 2013–2015 study of 20 UK households with motion and door sensors and measurements before and after installation. The project span does not establish continuous coverage for any household. [University catalogue](https://www.research.ed.ac.uk/en/datasets/refit-smart-home-dataset/)

A hypothesis to review is whether the order of recent motion events predicts a later opening on a separately recorded door channel when the current sensor state is the same. First verify that the released files contain synchronized motion and door channels with stable household and sensor identities, explicit coverage and missingness, and usable timestamps. The catalogue does not establish these details, per-household duration, event density, download and expanded sizes, or the applicable reuse terms. The DOI landing page was inaccessible during this review. No data files were obtained.

Before acquisition, fix the household roster, acquisition and state budgets, chronological development split and outcome horizon. Require coverage through the entire horizon for both classes. Keep the established current-state definition and support thresholds, and verify in development that identical current states recur across distinct histories with adequate support and both outcomes. Compare against the strongest current-state control and an event-driven scalar with the same delayed outcomes and actual local writes. Stop before scoring if any gate fails. The proposed door outcome is a separate hypothesis. The original Kitchen ON outcome and its 70,212-arrival gate retain their recorded status. [Decision gate](R2_MECHANISM_DECISION_GATE.md) · [Prior task evidence](ROADMAP_HISTORY.md#existing-research-queue)

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

日本語：REFIT は別の遅延結果課題の候補です。取得前に世帯別の同期・観測範囲と利用条件の確認が必要です。HTTPS 試験は独立運用者・接続先・審査済み信頼情報の確定待ちです。

简体中文：REFIT 是另一项延迟结果任务的候选来源。获取前需确认各家庭的时间同步、观测覆盖和使用条款。HTTPS 试点仍需独立运营方、端点和经审核的信任材料。

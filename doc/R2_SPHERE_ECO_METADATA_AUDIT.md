# Finite SPHERE / ECO metadata audit — 2026-10-18

🟢 [Done] Both **documented streams** are excluded for the unchanged raw **Kitchen ON** task. SPHERE exposes overlapping PIR history packets; ECO documents entrance motion, electrical measurements and household occupancy. Neither is the required native notification contract. This closes the registered review without acquiring payloads or transforming another outcome into Kitchen ON. Missing household coverage, PIR counts, package sizes and dataset reuse terms stay unresolved.

This is a documented-format exclusion, not an empirical learner result, an assertion about every possible future export, or evidence against SNN research. The earlier [public metadata feasibility review](R2_PUBLIC_SOURCE_METADATA_FEASIBILITY.md) and every learner negative remain unchanged.

## Registration and bounded acquisition of metadata

Selection was registered at `2026-10-18T09:05:50Z`, SHA-256 `ea0d71701a0359bfbef8256a26a5039a9e4fa9f8452c6b2375c098961cf1d6df`. Code/tests were pinned at `09:09:31Z`, registration SHA-256 `8721feb602953b359b1d7743d879b077aaf1b14d18a9cf7a2b97ec26683e3773`. Both precede the first new external query. Previously reported metadata was already known; this is a registered follow-up, not blind discovery.

The order is **SPHERE 100 Homes → ECO**. The scripted SPHERE Challenge and other SPHERE cohorts are not substitutes. ECO audit keys **01–06** represent the six published household numbers, not verified archive filenames. SPHERE permits publisher enumeration of at most 100 household metadata entries, with no invented identifiers. No such enumeration was established, so its empty entry list means **manifest unavailable**, not zero homes or complete coverage.

| Budget | Registered ceiling | Used |
| --- | ---: | ---: |
| Source families | 2 | 2 |
| Household metadata entries | 106 | 6 ECO; no SPHERE identities invented |
| Search requests | 8 | 8 |
| Primary metadata page attempts | 20 | 13 |
| Retained evidence UTF-8 bytes | 1,048,576 | 9,923 |
| Maximum retained bytes per page | 65,536 | Within ceiling |
| Dataset payload bytes / preview rows | 0 / 0 | 0 / 0 |
| Archive range requests / learning writes | 0 / 0 | 0 / 0 |

Each search/navigation is reserved in the managed request journal **before** the web call. Cached finds revisit already opened documentation. Every reservation has a receipt, including six failed page attempts. Failure does not imply absent data or permission. Browser transfer bytes are unobservable here; the byte ceiling covers retained transcription, not HTTP/TLS traffic. The review stops below the page ceiling after the fixed searches and relevant primary metadata checks; unused capacity does not authorize more sources or a renewed search.

The workflow refuses bulk/archive delivery URLs, non-HTTPS targets, extra ECO identities, source-order changes, failed-page proof, evidence-owner swaps, invented zero counts and payload/preview/range/learning/seal changes. Once executed, the journal, evidence and inputs are hash-sealed. Short metadata paraphrases are retained with primary URL/section references; hashes pin those transcriptions, not full publisher pages or source datasets.

## Fixed contract and findings

The eligible raw messages remain `ON/OFF/OPEN/CLOSE/CLOSED`. The first observed eligible notification **300–600 seconds** after a cutoff is positive only if its raw sensor name, lowercased, begins with `kitchen` and its raw message is `ON`. Missing outcomes are not negatives. Eight-event history, <=300-second adjacent gaps, >=60-second cutoff spacing, exact current sensor/message/weekday/hour/15-minute bin, `log(80)`, support/class/day/recurrence/contrast gates and actual-update parity remain unchanged. No room renaming, bit conversion, power threshold, packet expansion/backdating or occupancy-target substitution is performed.

| Required metadata | SPHERE 100 Homes | ECO 01–06 |
| --- | --- | --- |
| Household valid recorded dates | No scoped identity/date manifest established | Raw PIR valid dates unresolved for all six |
| Native PIR density | Packet layout/cadence documented; per-house native notification counts unresolved | Electrical 1 Hz does not supply PIR cadence/counts |
| Raw Kitchen ON compatibility | **Incompatible** documented packet format | **Incompatible** documented entrance/occupancy/electrical streams |
| Causal availability | Packet history available with the packet; original per-event availability not certified | Required Kitchen ON availability not established |
| Compressed / expanded household bytes | Unresolved | Unresolved |
| Dataset reuse/access terms | Policy-qualified availability statement; no verified release license | Primary dataset-specific license not verified; no access request sent |

The [SPHERE technical guide, section 4.3.2](https://arxiv.org/html/1805.11907v2) describes binary histories sent periodically with overlap. Its timestamp belongs to the history endpoint; replaying all elements as newly available independent notifications would change the interface and chronology. The [study page](https://www.bristol.ac.uk/research/groups/digital-health/research/sphere/the-100-homes-study/) supports year-long installations, not per-house valid dates. The [project dataset listing](https://research-information.bris.ac.uk/en/projects/sphere-epsrc-irc/datasets/) did not establish the required 100 Homes manifest. No global claim of nonexistence follows.

The [Bristol publication record](https://research-information.bris.ac.uk/en/publications/a-guide-to-the-sphere-100-homes-study-dataset/) lists a **3.86 MB manuscript**; that is not a dataset package size. Scripted Challenge catalog sizes/licenses do not transfer to the 100 Homes candidate. The guide's policy-qualified research availability statement does not certify a dataset reuse license or accessible bounded package.

The [ECO author paper, section 5](https://vs.inf.ethz.ch/publ/papers/beckel-2014-nilm.pdf) describes a PIR at the entrance, selected manual occupancy periods and electrical measurements. A kitchen appliance's power is not a native Kitchen ON notification. Household 06's lack of occupancy information is retained without replacement. Primary catalog/API/project fetch failures prevent verification of package bytes and dataset-specific terms. Third-party CC-BY descriptions are not adopted as publisher permission.

The [2015 author paper, ECO Data Set section](https://vs.inf.ethz.ch/publ/papers/kleiminger_ubicomp2015.pdf) supplies the following **cleaned manual occupancy-label support**, not raw PIR recording periods:

| Fixed household key | Summer label days | Winter label days | Scope |
| --- | ---: | ---: | --- |
| 01 / r1 | 39 | 46 | Published cleaned label support |
| 02 / r2 | 83 | 45 | Published cleaned label support |
| 03 / r3 | 57 | 21 | Published cleaned label support |
| 04 / r4 | 38 | 48 | Published cleaned label support |
| 05 / r5 | 43 | 31 | Published cleaned label support |
| 06 | Unavailable | Unavailable | No occupancy information in the 2014 description |

The described label seasons are July–September 2012 and November 2012–January 2013. These counts do not establish raw PIR dates, binary arrival counts, Kitchen ON class support or chronological train/development coverage. All six households retain unresolved raw-PIR date/count/availability/byte/terms checks: **36 unresolved checks**, plus six incompatible Kitchen ON checks. No underlying labels or sensor rows are read. A failed attempt to read a large author dissertation provides neither extra household evidence nor a dataset size.

The fixed **70,212** arrival target and original split remain: at least 180 valid source dates, `train=min(180,2*N//3)`, `development=min(60,N//6)`. Future ceilings remain 192 MiB body acquisition, 2 GiB expanded passes, 512 MiB per member/pass, 5M training rows per identity, 15M aggregate training and 36M metadata timestamp parses, 263,295 outcomes, 789,885 changed writes per arm and 100K routes. No future acquisition is authorized by this audit.

## Does this path use SNN strengths?

The user's question is retained in the protocol and machine-readable result. Native sparse asynchronous input, bounded membrane/branch timing and local delayed updates are useful **hypotheses** for reducing active work on the CPU. Their value depends on whether prediction needs that state beyond the exact current input and whether the total observer/maintenance work stays low. Periodic packet delivery, overlap handling, timestamp repair and dense electrical streams can consume that saving before a neuron is updated.

Metadata establishes no advantage unavailable to ANN or non-spiking implementations. Event-driven ANN and scalar emulators can share sparse scheduling, memory and local rules. The earlier [shared observer milestone](SHARED_OBSERVER_LEARNING.md) and scalar replay controls make representation/observer effects explicit; previous failed usefulness gates are preserved. A future claim needs the same native input and causal availability, matched **actual changed local writes**, strong current/history controls, gap/outcome controls, exact scalar replay and complete CPU/event/observer/expiry/maintenance accounting. Neither low spike counts nor avoided dense backpropagation alone establishes a measured accuracy/energy benefit.

The resulting decision is to stop acquiring metadata for these incompatible documented streams under this task. Continuing only to force another source into Kitchen ON would not isolate an SNN mechanism. Before another source search, review the stored evidence for an identifiable temporal/local-learning hypothesis and a measurable advantage over the sufficient controls. This changes no task, threshold, score, negative conclusion or sealed period and does not begin R3 runtime optimization.

## Verification and continuation

Fresh-process result replay is exact. **241 related tests** pass, including literal target rejection, packet/entrance-motion non-substitution, unknown-not-zero handling, fixed identities/budgets, missing/failed/cross-source proof, immutable execution-input hashes and all **345** earlier JSON/gzip artifacts and registered code. A compatible synthetic schema still authorizes no payload and passes no quality gate. These are metadata-validator tests, not a learned prediction benchmark.

Evidence lives under `data/processed/benchmark_fixtures/sphere_eco_metadata_*_v1.json` plus the reserved `requests_v1.jsonl` journal. Replay performs no web or dataset access and preserves its receipt:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/eval/sphere_eco_metadata.py replay
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q -o cache_dir=workspace/cache/pytest tests/test_sphere_eco_metadata.py
```

🟢 [Done] The [read-only R2 research-decision audit](R2_RESEARCH_DECISION_AUDIT.md) subsequently registered and checked 19 known aggregate receipts, with zero new scores/learning. Local temporal components remain identifiable, but predictive/resource superiority over sufficient controls remains unestablished. Missing cost stages stay unknown; 333 related tests and exact replay preserve all earlier results and seals.

🟢 [Done] The [input-only work-accounting contract](R2_EVENT_WORK_ACCOUNTING.md) subsequently passes all 48 finite synthetic cases, including shared observer charges, delayed/late/missing receipts, idle expiry and exact event-driven scalar equivalence. Zero learning/scoring; 356 related tests and fresh-process replay preserve negatives and seals.

🟢 [Done] The subsequent [instrumentation-separation contract](R2_EVENT_WORK_SEPARATION.md) now passes the same 48 cases: 1,232 state comparisons including teardown, prior receipt hashes, receipt-only execution, 381 related tests and exact replay. All 363 earlier artifacts/code pins, negatives and seals are preserved. Returned buffers/control are counted; intra-call peaks remain unknown.

🟢 [Done] The subsequent [isolated allocation diagnostic](R2_EVENT_ALLOCATION_DIAGNOSTIC.md) is implemented, but its 144-worker finite run failed the fixed 8 MiB result-storage gate. No raw values or replay are retained; preserve this failure without relaxing the limit or rerunning. 411 tests pass with one expected pre-execution skip; 368 earlier artifacts/code pins and all seals remain intact.

🟢 [Done] The subsequent [report-representation preflight](R2_ALLOCATION_REPORT_PREFLIGHT.md) proves a conditional 7,302,888-byte ceiling within fixed 8 MiB, with lossless ownership/unknown-site encoding and pre-launch saved-result validation. 41 tests and exact static replay pass, preserving 372 prior artifacts, failure and seals. No allocation measurement or diagnostic worker is run.

🟢 [Done] A separately registered [allocation diagnostic v2](R2_EVENT_ALLOCATION_V2.md) now retains primary/replica reports of 608,412 bytes each, with 3,840 matching windows and 51 tests. Preserve 378 earlier artifacts/code pins, failed v1 and seals. No neuron memory saving or CPU/energy advantage is established.

🟠 [Next] Preregister a finite retained-evidence decision gate combining original predictive/mechanism results with completed accounting/separation/allocation receipts. Identify a decision-changing SNN mechanism and independently qualifiable task prerequisite, or hold further resource work. No new measurement, scoring, search/acquisition, automatic v3, threshold change or R3 optimization.

⭕️ [Pending] New payload acquisition and independent SNN-usefulness/energy claims require a compatible independently justified source/task contract, scoped access/size/coverage metadata and the original preregistered causal/quality/resource gates. SPHERE/ECO documented-stream exclusions are closed; no automatic metadata retry or CASAS expansion.

日本語: SPHERE・ECOに限定し、候補・判定条件・コード・総予算を外部確認前に登録しました。8検索・13ページで、公表ストリームは元のKitchen ON定義に不適合と判断し終了しました。ECO01～05の在室ラベル日数は確認できましたが、PIR記録期間や到着密度には流用せず、サイズ・利用条件の不明点も保持しました。本体取得・学習・候補追加・条件緩和・封印解除は0、241テストと別プロセス再現が成功し、既存345件の証跡を維持しました。SNNの疎な時間状態・局所更新は利点候補ですが、ANNにも共有できる性質と区別し、同じ入力・実更新回数・総処理量での検証を要求します。

简体中文：仅针对SPHERE、ECO，在外部查询前注册候选、判定条件、代码与总预算。使用8次搜索、13次页面请求后，判定已公开描述的数据流不符合原Kitchen ON定义并终止。确认了ECO01～05的在家标签天数，但不将其替代PIR记录期间或到达密度，保留体积及使用条件的未知项。本体采集、学习、候选扩充、放宽条件和解除封存均为0；241项测试与新进程重放通过，保留345项既有证据。稀疏时间状态和局部更新是SNN优势假设；需区分ANN也能共享的性质，并在相同输入、真实更新次数和完整处理成本下验证。

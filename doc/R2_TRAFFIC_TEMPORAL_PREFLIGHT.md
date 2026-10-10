# Fresh traffic temporal-necessity preflight

🟢 [Done] Task/code preregistration, source/support audit and local-conjunction
readout implementation. Analytical and unit-learning checks pass; the real-data
current-contrast support gate fails before learner scoring.
🟠 [Next] Select a fresh source with repeated, independently observed current
contexts supporting the registered temporal-necessity question. Do not relax
this consumed task's matching conditions or decode its sealed data.
⭕️ [Pending] Sealed traffic scoring, spike-specific usefulness and physical
energy measurements remain on hold.

## Source selection and frozen task

The new source is [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume),
an hourly traffic stream from Minnesota station 301, with contemporaneous
weather and holiday fields. Attribution: Hogue, J. (2019),
DOI [10.24432/C5X60B](https://doi.org/10.24432/C5X60B), CC BY 4.0.
This source is independent of the consumed appliance, electricity and Beijing
tasks. No learner or source numerical value was inspected before preregistration.
The archive and inner compressed/uncompressed CSV byte identities were pinned
without interpreting numeric values.

The metadata-only source screen also considered UCI Air Quality, but its own
description includes research-only/commercial-use exclusions alongside UCI's
CC BY notice. No files from that candidate were acquired; this uncomplicated
CC BY traffic source is used instead. This is a source eligibility choice,
not selection based on a competing learner score.

The registered target is **whether measured traffic at exactly cutoff + one
hour is at least 4,000 vehicles/hour**. This is a research class definition,
not an official congestion or safety limit. Inputs are the four hourly volumes
from cutoff − three hours through cutoff, plus current weather/holiday and
calendar. Train observations are before 2016; development is 2016; **2017 onward
remains sealed**. Histories and individually observed future labels must fit
entirely within one split. Missing hours, conflicts or invalid readings abstain;
no missing follow-up is inferred as a negative outcome.

The protocol hash is
`f16018e6c66c700c70ce7e4dbafb0aed724b38bf063f50032e4ff54673781f94`.
The compressed source hash is
`0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07`;
the uncompressed CSV hash is
`749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e`.
Registration retains the code hashes, an analytical branch witness and hashes
of earlier negative protocols/receipts and all seven appliance checkpoints.
Execution/replay rejects changes to those identities. Old sources and sealed
numeric values are not decoded or rescored.

## Strong current controls and local discrimination

Two learned current-state controls receive current traffic, exact hour,
weekday, month and the common clock/current channels. `current_exact` also
conditions on the exact current volume; `current_weather` adds contemporaneous
weather and holiday bins. An untrained current-threshold persistence reference
is also scored. The strongest of those three is selected by the registered
equal-day Brier rule; no weak current-only baseline is used as the sole hurdle.

The **temporal_pair** arm retains the same weather/current/calendar signature
as `current_weather`, plus clipped volume changes over one and three hours.
Related controls retain the sorted historical bins, reverse only the past
samples while holding current state fixed, or train with within-day shuffled
outcomes. Volume bins are 50 vehicles; trend bins are 250, capped at ±2. Weather
bins are fixed before source inspection. Bin saturation and omitted intermediate
trend information are explicit limitations, not complete ordinal retention.

Four scalar dictionaries represent a global context, clock context, current
volume/clock context and a specialized local conjunction. They compose
posteriors using `(positive + 8 × parent_probability) / (count + 8)`, starting
at 0.5. Context partitions are not mathematically nested: the common current
context deliberately coarsens hour and omits quarter; it is a fixed interpolation
hierarchy. Unlike the consumed appliance readout, this implementation does not
average many independent historical marginal probabilities. One outcome changes
one integer counter in each of four unique route namespaces, so all six learned
arms have exactly four actual changed writes per feedback. Persistence has zero
learning writes and is explicitly separate from that matched learned comparison.

The analytical witness compares histories `(3250,3500,2750,3000)` and
`(2750,3500,3250,3000)`, with identical weather/calendar/current value and equal
historical histograms. Both strong current controls and the histogram control
have identical channels, while the temporal conjunction addresses differ.
Opposite branch-local parameters give probabilities 0.75 and 0.25. Unit tests
also verify that local delayed counter increments can learn the distinction.
This is a representation/readout check, not a new synthetic research learner
experiment or a source-based proof of temporal necessity. Larger same-sign
increments can collide under the fixed saturation rule; tests retain that
counterexample instead of concealing the loss.

## Retained preflight result: insufficient contrasts

| Split | Admitted examples | Positive | Negative | Admitted days |
| --- | ---: | ---: | ---: | ---: |
| Training | 13,840 | 5,846 | 7,994 | 735 |
| Development | 6,106 | 2,552 | 3,554 | 323 |

The fixed class/day support gate passes. However, the training set has 13,627
distinct rich current contexts and **zero** groups meeting the registered
opposite-future-outcome/distinct-temporal-context criterion, versus **20**
required. **Stop before real-data learner scoring.** The six learner arms remain
unexecuted on this source; no training counter writes, learned scores,
development prediction traces or final traffic model checkpoints were produced.
Do not interpret the zero contrast count as proof that current state suffices
or that time has no effect. The rich present contexts are almost all distinct;
this source/matching contract does not supply the registered conditional test.

The input/readout audit separately passes: same-current/equal-histogram input
histories can address distinct local conjunctions, and bounded local updates
can learn that distinction in unit tests without differentiating a network.
That representability result does not authorize a real-data performance claim.
All related **55 tests pass**, including the explicit saturation collision,
sealed-cell spies, duplicate-hour collapse/conflict handling, purged splits,
early/forged receipt rejection, equal changed writes in unit tests, frozen
checkpoint replay in unit tests and retained-negative evidence hashes.

Source parsing reads 29,650 admitted numerical rows and leaves **18,554 sealed
rows timestamp-only**. It collapses 4,321 duplicate-hour rows, excludes 88
invalid/conflicting hours (90 conflicting duplicate rows and 10 invalid rows),
and retains 25,241 valid unique hours before split/history/follow-up admission.
Those quality checks are distinct from label/class support. No 2017-onward
traffic or weather value is decoded. Old appliance/electricity/temporal negative
receipts, protocols and all seven appliance model hashes remain unchanged.

Independent-process `--verify-replay` reproduces source access counts, case
identities and contrast counts exactly, while checking preserved hashes. Since
this preflight stopped before scoring, it performs **source-audit replay only**;
there are no learned traffic prediction traces to replay. The runner's generic
replay completion message must not be mistaken for an executed learner phase.

Retain this failed selection without coarsening current contexts, lowering the
20-group threshold, changing the forecast target or scoring the reserved years.
For the next independent source, assess whether its observable present state
can recur with distinguishable temporal histories before another learned
temporal-usefulness comparison. No fresh independent task demonstrating time's
incremental predictive necessity has been established by this preflight.

## One-shot gates and causal accounting

Before scoring, require 500 train/250 development examples per class, 100/60
admitted days, and 20 train-only current-weather groups with opposite observed
outcomes and distinct temporal context addresses. These source contrasts are
descriptive at the registered quantization, not causal matched trials or proof
that raw present observations cannot suffice. Failure stops before model scores.

For development, temporal_pair must improve equal-day Brier by at least 0.003
over the strongest current control, balanced accuracy by at least 0.005, with a
positive paired-day one-sided 95% bootstrap lower bound (2,000 fixed replicates).
Order reversal must worsen Brier by at least 0.001; shuffled outcomes must worsen
it by at least 0.01. Matched changed writes and exact frozen checkpoint replay
must pass. Histogram performance is retained to distinguish access to historical
values from order-specific effects. No multiple learner seeds or statistical
independent-trial claim is made; adjacent days can also be correlated.

Owned receipts contain projected keys and predicted probability, never a label.
The evaluator validates receipt identity and exact due time before accessing the
outcome cell. One-hour feedback is released before same-time prediction, with
at most one pending receipt. Development performs zero counter updates. The
shuffled arm's artificial training-only outcome schedule is a negative control,
not a deployable source. Source row-time and physical publication time must not
be conflated; clock/interval conventions are not independently certified here.

Only source time is read before the sealed-period admission check. Duplicate
weather rows within an hour are collapsed into one traffic observation, with
sorted weather-category unions; conflicting numeric/holiday rows exclude that
hour. Models are CPU-only scalar dictionaries, bounded at 65,536 routes and
64 MiB per readout; capacities are checked before allocation, with measured
end-state size below conservative key/count/scratch allowances. Four history
visits and count lookups per example, changed writes, checkpoint instrumentation
and replay work are retained. Concurrent original/replay readout bounds and
materialized evaluator memory are separately charged. This is not a measured
peak-RSS, physical energy or deployed latency claim.

## Reproduction

With `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`, run:

1. `python3 scripts/eval/traffic_temporal_preflight.py --acquire`
2. `python3 scripts/eval/traffic_temporal_preflight.py --register`
3. `python3 scripts/eval/traffic_temporal_preflight.py --execute`
4. `python3 scripts/eval/traffic_temporal_preflight.py --verify-replay`

Acquisition verifies byte hashes and writes only the compressed CSV under
`data/raw/r2_traffic/`. Registration and execution refuse overwrites; execution
is marked consumed even if it fails partway. Committed registration/result
evidence is not replaced during replay. Research count checkpoints belong under
`models/traffic_temporal/`, full generated reports under `workspace/evaluation/`,
and compact frozen receipts under `data/processed/benchmark_fixtures/`.

日本語: 新しい交通量課題を事前登録し、局所読み出しの順序識別は確認できました。一方、実データの現在状態対照群は必要20群に対して0群で、学習・採点前に停止しました。時間情報の必要性は未確認です。旧負の結果は不変で、2017年以降18,554行の数値は封印しています。

简体中文: 已预注册新交通任务，局部读出可区分顺序；但真实数据符合当前状态对照条件的组数为0，未达到20组门槛，因此在训练及评分前停止。尚未证明时间信息的必要性。旧负结果保持不变，2017年之后18,554行的数值继续封存。

# Preregistered prospective appliance comparison

🟢 [Done] Protocol registration, matched delayed learner execution, bounded-state
checks, retained final count checkpoints and independent-process replay.
🟠 [Next] A fresh temporal-necessity preflight against a strong current-state
reference before another real-data joint candidate; preserve this consumed task.
⭕️ [Pending] Sealed test scoring and physical energy measurement are on hold;
the development promotion gate failed.

The task remains the [frozen prospective appliance task](R2_APPLIANCES_PROSPECTIVE_TASK.md).
No April-onward numeric value or test outcome may be decoded or scored.
The new learning protocol SHA-256 is
`fbb5422d235d31764d00f37fae5f1ba0f05d2c41374854ea6ca092dc06058188`.
Before executing any real-data learner, `--register` retains a timestamp and
hashes of the protocol, extractor, runner, learner, neuron and accounting code.
Execution rejects changes to these identities and marks even partial runs as
consumed. There is no hyperparameter sweep, repeat-until-pass or automatic test
unsealing. Development is a screening split, not confirmatory evidence.

## Fixed comparison

Seven arms use a scalar, sparse dictionary of independent local Bernoulli
class counts, with one pseudocount per class. Every admitted delayed training
outcome increments exactly one integer class counter on each of 40 unique
active routes: **40 changed scalar writes per outcome in every arm**. These are
actual changes, not equal ceilings or padded no-op calls. Prediction averages
route posteriors using input-only strengths; it does not differentiate a network
or perform matrix calculations. Counts are zero-initialized deterministically;
the shuffle/bootstrap seeds do not represent multiple independent learner runs.

The common energy representation bins historical Wh values in fixed 20 Wh bins,
capped at bin 63. Each arm receives four contextual routes: bias, current bin,
hour and weekday. Quantization is common information loss, not a spike effect.

| Arm | Other 36 routes | Prediction strengths |
| --- | --- | --- |
| snapshot | Current bin repeated in 36 position namespaces | 1 |
| histogram | Sorted history bins, preserving the full quantized histogram | 1 |
| ordinal | All 36 bins with their chronological positions | 1 |
| transition_gap | Reconstruct all bins from change positions | 1 |
| analog | Same complete ordinal history | 1 + capped continuous membrane |
| joint | Same complete ordinal history | 2 on LIF spike, otherwise 1 |
| joint_shuffled | Same as joint; evaluator permutes training labels within day | Same as joint |

Snapshot repetition matches real writes but adds no historical information or
independent observations. Its duplicates have identical posteriors. The primary
comparison is against the best of **all five** non-spiking controls, including
complete ordinal and continuous-membrane references; snapshot is not the sole
baseline. Gap reconstruction and ordinal predictions must match exactly.

The LIF encoder uses the existing dendritic `Neuron`, one branch, a fixed
0.95 leak, threshold 1, branch threshold 0.5/gain 2 and two refractory steps.
Current is capped at 1 after dividing observed Wh by 200. It resets for each
36-sample history. Spikes modulate fixed readout strengths; internal synapses
are not trained. Analog uses the same input, leak and branch transfer without
reset/refractory behavior. An independently written scalar recurrence verifies
every LIF projection; this is replay verification, not an additional learning
arm or proof that spiking computation is uniquely useful.

## Delay, sealing and accounting

Cutoff-bound owned receipts retain routes/probability, never a label. A virtual
chronological clock releases training results exactly 30 minutes later, before
any prediction at that same timestamp. Time and receipt identity are checked
before the evaluator's outcome cell is accessed. Early, late, copied,
cross-owned and duplicated outcomes are rejected. At most three receipts are
pending; each phase drains its own outcomes and no history/outcome crosses a
split boundary. Development counts remain frozen, with zero updates.
This verifies a virtual row-time schedule. Actual sensor publication latency,
interval timestamp conventions and timezone are not independently certified;
the experiment does not establish a real-time deployment availability contract.

The shuffled arm is an explicit negative control: its training-only evaluator
precomputes a within-day label permutation preserving daily support, then
releases those artificial labels at the destination's due time. That artificial
schedule is not a deployable causal outcome source and is never used by other
arms. No development label is used for any learning.

Each arm charges 36 common history visits and 40 prediction-route visits per
example. LIF/analog steps, spikes, independent scalar-reference work, feedback
writes, checkpoint replay and CPU time are reported separately. The 5,120-route
vocabulary and conservative count/receipt/scratch bound are checked against
4 MiB per arm; end-of-phase object sizes must fit that bound. Source/evaluator
tape memory is separately measured and disclosed, rather than omitted from
the batch experiment. These are operation/state measurements, not physical
energy, latency guarantees or evidence of an energy advantage. Final research
count checkpoints live only under `models/appliances/`; generated full evidence
lives under `workspace/evaluation/`, with compact receipts under `data/processed/`.

## Fixed decision

The primary loss is the equal-day mean Brier across the 31 development days,
which avoids treating overlapping ten-minute examples as independent trials.
Select the best non-spiking reference by that metric, breaking ties in listed
order. A paired day bootstrap uses 2,000 fixed-seed replicates and the registered
one-sided lower 95% gain bound. Consecutive days can also be correlated; this
development screen is not a formal independent-trial significance claim.

All gates must pass: joint balanced accuracy at least 0.60, daily Brier at most
0.20, at least 0.01 Brier improvement over the strongest reference with positive
bootstrap lower bound, and at least 0.01 improvement over shuffled outcomes.
Actual changed updates, bounded credit/state, ordinal/gap identity and exact
saved-checkpoint replay must also pass. Failure preserves the result and stops
this candidate without changing its thresholds or scoring sealed data.

## Commands

Use `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` with:

1. `python3 scripts/eval/appliances_delayed_comparison.py --register`
2. `python3 scripts/eval/appliances_delayed_comparison.py --execute`
3. `python3 scripts/eval/appliances_delayed_comparison.py --verify-replay`

The third command performs only frozen development checkpoint replay. The first
two refuse to overwrite consumed registration/execution evidence. Public source
acquisition and its attribution are documented in the task contract.

## One-shot result: negative increment

| Arm | Equal-day Brier | Pooled Brier | Balanced accuracy |
| --- | ---: | ---: | ---: |
| snapshot | 0.159422 | 0.159225 | 0.588741 |
| histogram | 0.176023 | 0.175697 | 0.500000 |
| ordinal | 0.174090 | 0.173763 | 0.500000 |
| transition_gap | 0.174090 | 0.173763 | 0.500000 |
| analog | 0.174478 | 0.174148 | 0.500000 |
| joint | 0.174303 | 0.173975 | 0.500000 |
| joint_shuffled | 0.177993 | 0.177657 | 0.500000 |

All seven arms train on 7,060 admitted examples and perform **282,400 actual
counter increments each**, with all 282,400 counters changed. Each then makes
4,426 frozen development predictions with **zero writes**, maximum three owned
pending receipts and no unresolved receipt at phase end. Final checkpoint count
sums independently agree with those writes. Ordinal and reconstructed-gap
probabilities match exactly. Both LIF arms match the independent scalar
recurrence on every training/development history. All seven model checkpoints
replay development prediction digests and day-level metrics exactly, including
a separate Python-process verification. Related tests: **35 passed**.

The best reference is snapshot. Joint's Brier gain against it is **−0.014881**,
with paired-day lower 95% bound **−0.018367**, versus the registered +0.01
requirement. Joint improvement over shuffled outcomes is only **0.003690**,
also below +0.01. Joint balanced accuracy is 0.50 versus required 0.60. Quality,
SNN increment and shuffled-outcome gates fail; actual updates, changed writes,
ordinal/gap equivalence, bounded credit/state and checkpoint replay pass.
The candidate is closed without retuning, production promotion or test unsealing.

The analog/ordinal/joint arms all have balanced accuracy 0.50 at the fixed
0.5 threshold. Dilution in this additive posterior readout is a possible
explanation, not an established cause. The result rejects this
fixed candidate's usefulness, not all local learning or all SNN designs. Full
history availability alone does not ensure that the local readout exploits it.
Before another independently observed outcome comparison, preregister a fresh
temporal-necessity check against the strongest current-state reference and
test that the chosen bounded readout can discriminate the required histories.
Do not use this development result to tune bins, strengths, thresholds or labels.

Training count tables have 724–1,771 routes. Conservative per-arm state bounds
are 470,016–1,006,080 bytes, below 4 MiB; measured final readout objects are
approximately 132–271 kB (decimal). The separately charged batch
source/evaluator tape uses 7,381,606 bytes. Per training/development pass every
arm charges 254,160/159,336 history visits and 282,400/177,040 readout visits.
Joint emits 33,528/22,634 spikes, with 36 LIF steps per example; scalar recurrence
checks and saved-checkpoint replay add separately reported work. No history
values exceeded the registered bin cap. CPU timings include instrumentation
and scalar verification, so they do not isolate deployed inference latency.

The full local report SHA-256 is
`af1e858c4812f2070b1f8c03c8664c0a9155d5beeacd28ec0930d569b29f38e2`.
The committed compact receipt retains all training/development metrics, daily
losses, decision gates, code identities and checkpoint hashes. The seven small
research count checkpoints are committed under `models/appliances/` for frozen
replay after acquiring the public source. They are not production models.
All **8,173 sealed rows remain timestamp-only**, with zero sealed numeric
accesses. Raw archives/CSV and the full workspace report remain local.

日本語: 全7方式で実更新282,400回、開発更新0回を確認し、保存状態から完全再現しました。SNNの日平均Brierは0.174303、最良対照は0.159422で改善条件は不合格です。負の結果を保存し、4月以降の数値は封印したままです。

简体中文: 七种方法各实际更新282,400次，开发阶段零更新，保存状态可精确重放。SNN的日均Brier为0.174303，最佳对照为0.159422，未通过提升门槛。负结果保留，四月之后的数值继续封存。

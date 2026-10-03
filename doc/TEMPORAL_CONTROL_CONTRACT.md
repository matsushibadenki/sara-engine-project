# Temporal control contract v1

## English

[Done] Registered an input-only synthetic contract (`d4082689…b40e2`) and implemented two controls without reopening electricity values or changing the consumed v2 implementation. The audit pins the retained reference source hash. Its report is `workspace/evaluation/temporal_control_contract_v1.json`.

| Control | What is matched | What this establishes |
| --- | --- | --- |
| Equivalent scalar gate | Every admitted comparison route, arithmetic, reset and cooldown | A neuron object and spike vocabulary are unnecessary to reproduce these outputs; the underlying temporal computation is still present. |
| Count-yoked recency selector | Number of distinct readout routes and unit strength; calendar/bias unchanged | Route identity can differ while admission quantity and readout normalization are matched. It is not an independently deployable resource baseline. |

The second control receives only the reference output cardinality K and the same input history. It selects K distinct observed routes by most recent occurrence. It receives no reference route membership, outcome or learned rank. Matching concerns deduplicated readout route count, not individual spike count, timing or equal information content. K itself carries reference-dependent information. Reference computation must therefore be charged alongside selector work; selector-only CPU or state cannot support efficiency claims.

On 103 fixed synthetic traces (seed 31003 plus constant, cycling and recency-conflict fixtures), equivalent output parity and cardinality parity pass. Reference/recency route identity differs in 101 cases; one cycling case admits zero comparison routes and is retained rather than excluded. Calendar/bias input remains available in zero-count cases. Maximum reference/gate work is 2,280 ticks; recency scans 95 events and ranks at most 24 routes. Oversized input stops after probing 96 elements, before simulation; invalid routes/counts fail closed. Tests cover episode reset and deterministic selection.

This is a contract audit, not a learning benchmark or predictive result. The equivalent scalar gate preserves decay, threshold, zero reset and cooldown; renaming these does not prove temporal integration or refractory behavior dispensable. The recency control is reference-yoked, not fully independent. Neither establishes a spike-specific advantage. No real data, outcomes or candidate scores were read; 2014 remains sealed and the electricity v2 candidate remains closed.

[Next] Define a separately registered causal experiment only after checking that its target is independent of the reference implementation, its temporal controls receive matched input/state budgets, and any count-yoking cost is disclosed. Require independently deployable non-spiking temporal controls alongside the yoked diagnostic. No scoring or new real-data access is authorized by this contract. Fresh confirmation and production promotion remain [Later].

## 日本語

[Done] 実データを再閲覧せず、入力だけのsynthetic契約を固定しました。scalarの蓄積値・閾値・cooldownでLIFの出力を再現する対照と、同数Kのrouteを最近の出現順で選ぶ対照を実装しました。103系列で完全一致と出力数一致を確認し、101系列では選択routeが異なりました。出力0の系列も保持します。

一致するのはreadoutに渡す重複除去後のroute数であり、spike総数・時刻・情報量ではありません。recency対照はLIFからKを受け取る診断用対照なので、LIFを動かすコストも含める必要があります。等価gateは時間積分・閾値・reset・cooldownを残しており、それらが不要という証拠ではありません。性能評価は未実施、2014年は封印維持です。

[Next] targetが参照実装に依存せず、入力・state上限を揃えた独立非spiking時間状態対照も含む因果実験を別途事前登録します。本契約はscore計算や新しい実データアクセスを許可しません。[Later] 新鮮な確認データと本番採用。

## 简体中文

[Done] 未重新访问真实数据，冻结了仅输入的synthetic契约。实现了用scalar累积值、阈值和cooldown复现LIF输出的对照，以及按最近出现顺序选择相同数量K个route的对照。103条序列通过完全一致与数量一致检查，101条的route身份不同；保留零输出序列。

匹配的是传给readout的去重route数量，不是spike总数、时间或信息量。recency对照从LIF取得K，仅用于诊断，必须计入参考LIF计算成本。等价gate保留时间积分、阈值、reset与cooldown，并不证明这些机制可删除。尚未评估学习性能，2014年继续封存。

[Next] 独立预注册因果实验：target不能依赖参考实现，且必须包含输入与state上限匹配的独立非spiking时间状态对照。本契约不授权评分或新的真实数据访问。[Later] 新鲜确认数据与生产采用。

# Independent temporal design feasibility v1

## English

[Done] The input/target contract is frozen at SHA256 `59b13be6f44a539d9d9661cd4457a58daf518e08ca0b2673881ab815fd7ec270`. Independent non-spiking controls now include a bounded leaky analog trace, first-visit ordinal memory, and an order-blind histogram. They do not call LIF or receive its output count, route identities or targets. Input vocabulary, episode length and resource ceilings are shared; actual storage, arithmetic, output strengths and information content need not be equal.

The generator's exogenous binary mode selects `(0,1)` or `(1,0)` as the first two events. The outcome equals that mode, not any neuron's firing. Each pair shares 91 distractors plus terminal `(0,1)`, so route histograms, last occurrences, suffix, calendar-free routing and sequence length are identical. Only first-cue order differs. These are controlled synthetic targets, not evidence about natural intelligence.

The pre-learning audit uses 102 pairs / 204 sequences, including constant/cycling distractors and 100 fixed random suffixes (seed 6200410). Opposite-target readout-signature collisions are:

| Primitive | Colliding pairs |
| --- | --- |
| Independent leaky analog | 0 / 102 |
| Independent first-visit memory | 0 / 102 |
| Histogram | 102 / 102 |
| Retained LIF admitted-route interface | 102 / 102 |

**Decision:** do not register or run a learner superiority experiment with this LIF output interface on this exact target. Singleton early cues and their widely separated terminal repeats never cross its threshold; both modes therefore yield identical deduplicated admitted routes. A frozen deterministic readout receiving these identical inputs cannot recover which cue came first. This is an interface identifiability failure, not a trained accuracy measurement, and not a claim that every SNN loses temporal information.

The independent controls retain distinguishable signatures, which is necessary but does not establish successful local learning, better CPU cost or an energy advantage. Analog traces still contain decay-based temporal state; first-visit memory uses integer ordinal state without membrane, firing threshold or refractory mechanics. The audit supports examining what temporal information must remain readout-visible before adding more neuron mechanisms.

Report: `workspace/evaluation/independent_temporal_design_v1.json`. Contract integrity passed, learning readiness failed (the audit command exits nonzero deliberately). No learner was trained or scored, no real values were read, and 2014 remains sealed. Tests validate independence from the reference, paired nuisance matching, state reset, pre-work input bounds and immutable contract identity.

[Next] Specify a backend-independent readout-visible temporal-information contract and its event/state cost. Verify information preservation under paired histories before separately preregistering a learning experiment. Do not patch or retry the consumed electricity candidate or tune this rejected interface on target scores. Fresh real confirmation and promotion remain [Later].

## 日本語

[Done] LIFを呼ばないanalog時間trace・first-visit順序memory・histogram対照を実装し、入力／target契約を凍結しました。外生modeで最初の2イベントだけを反転し、回数・最後の出現時刻・共通suffixを一致させています。targetはLIFの計算結果から作りません。

102組204系列の検証で、analog／first-visitの異target入力衝突は0組、histogram／既存LIFのreadout入力衝突は全102組でした。既存LIFの重複除去された出力は最初の順序情報を消すため、この境界で学習比較を進めることは不適切です。実学習の精度測定でも、SNN全般の否定でもありません。情報が残る対照についても性能優位は未証明です。

[Next] backendに依存しないreadout可視の時間情報契約とevent／stateコストを定義し、情報保存を検証してから別の学習実験を事前登録します。実データの再閲覧・候補score計算・2014年閲覧は行っていません。[Later] 新鮮な実データ確認と採用。

## 简体中文

[Done] 实现不调用LIF的analog时间trace、first-visit顺序memory及histogram对照，并冻结输入／target契约。外生mode只反转最初两个事件，保持频次、最后出现位置及公共suffix一致；target不由LIF输出生成。

102对／204条序列中，analog／first-visit的异target输入碰撞为0，histogram／现有LIF的readout输入碰撞为全部102对。现有LIF去重后的输出丢失了最初顺序，因此不应在这一接口继续学习比较。这不是训练准确率，也不否定全部SNN；保留信息的对照仍未证明性能优势。

[Next] 定义与backend无关的readout可见时间信息契约及event／state成本，验证信息保存后另行预注册学习实验。未重新读取真实数据、未评分候选、未访问2014年。[Later] 新鲜真实数据确认与采用。

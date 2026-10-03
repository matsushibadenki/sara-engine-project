# Electricity matched learner comparison v1

## V1 execution integrity failure — retained

[Done] The streaming adapter, sufficient-statistic client/month metrics, three training-only baselines and exclusive execution runner were implemented. Synthetic tests confirmed that changing future values changes the label but not history routes. The real one-shot attempt then failed closed with `ValueError: Probability outside unit interval`, before any final report or adoption decision.

The cause is an omitted scoring contract, not evidence against LIF: `BoundedLocalOutcomeReadout` produces signed scores in [-1,1]. Even with binary targets, a zero-target update shared across differently weighted active routes can make a route negative; a subsequent route subset can yield a negative score. A minimal synthetic regression reproduces this without reopening real values. V1 did not register a conversion to probability, so silently clipping or remapping and rerunning would change the experiment.

The empty exclusive report reservation is preserved. Failure receipt: `workspace/evaluation/r2_electricity_development_v1_failure.json`. The execution did not record its phase at failure; development access must not be assumed absent. There are no final Brier, comparative or architecture-quality conclusions. Numeric parsing of 2014 remains zero. Resource acceptance was not established by this interrupted run.

[Next] Register and test an explicit score/probability/target contract under a new corrective identity, disclose this consumed attempt and any development contamination, and require a fresh confirmation boundary before promotion. Do not rerun v1 or reinterpret its missing metrics as a quality failure.

日本語: [Done] adapter・集計・baseline・一回実行runnerを接続しましたが、実行は確率範囲検証で停止しました。局所readoutは符号付きscoreを返し、二値targetでもroute集合が変わると負のscoreになり得ます。これはscoring契約の欠落であり、LIFの性能評価結果ではありません。試行は消費済みとして保持し、黙って補正・再実行しません。停止時のphaseは記録されていないため、developmentが未閲覧とは主張しません。2014年は封印維持です。[Next] 新しい訂正protocolで確率変換を明示し、採用には新鮮な確認境界を要求します。

简体中文：[Done] 已连接adapter、指标、baseline与单次runner，但真实运行因概率范围检查而停止。局部readout返回有符号score，即使采用二值target，变化的route集合也可能产生负score。这是scoring契约缺失，不是LIF质量结论。保留已消耗的试次，不静默修正或重跑。停止阶段未记录，因此不能声称未访问development。2014年仍封存。[Next] 在新的纠正协议中明确概率转换，并要求新的确认边界后才能采用。

## Implementation checkpoint

[Done] `electricity_learner.py` implements the 96-observation history encoder and scalar, compact, pure LIF and event-reset LIF units. Input consumption is capped at 97 observations to reject oversized streams before simulation. Route bounds are validated first. The stateful arms perform at most 2,280 unit ticks per episode; current is delivered only to the observed route while inactive units leak. All units reset at episode start. Scalar/compact share exact pulse semantics intentionally, providing a required equivalence reference rather than a predicted quality difference.

Receipt-bound local learning enforces outcome identity and exactly four intervals of delay. Development resolves receipts without updates. Six focused tests pass, including protocol identity, bounded input, unit reset, scalar/compact parity, delayed outcome rejection, frozen development and elementary Brier validation. This is not an end-to-end leakage audit: the streaming eligibility-to-encoder adapter and aggregated metric runner remain [Next]. No real candidate scores were computed. The LIF comparison remains a pure-unit research adapter, not a production neuron replacement.

日本語: [Done] 履歴encoderと4系統のlearnerを実装しました。入力・routeを計算前に検証し、遅延とoutcome IDを拘束、development更新を禁止しています。focusedテスト6件成功。実データの候補評価は未実行です。[Next] 時刻付きstreaming adapter、集計指標、評価runnerを接続し、end-to-end境界を検証します。

简体中文：[Done] 已实现历史encoder与四种learner。计算前检查输入与route，绑定延迟和outcome ID，禁止开发阶段更新。六项focused测试通过，尚未计算真实候选分数。[Next] 连接带时间边界的streaming adapter、汇总指标和评估runner，验证end-to-end边界。

## English

[Done] The prospective learner protocol is frozen at SHA256 `0093d72c34a72baf49f21220e6cadbef2af0993eb3b0cce366e2e495b7fef89f`, binding the feasibility protocol and retained support report. No candidate has been scored. Numeric access remains limited to 2011–2013; 2014 stays sealed even if development succeeds.

All four arms receive the same 95 chronological history comparisons and calendar/bias routes, the same delayed outcomes and local readout rule, and the same resource caps. Scalar and compact event are required to have identical prediction/update traces. The pure stateful LIF arm is compared with both these controls and an event-reset LIF intervention. The existing synthetic `EventUnitLearner` is not reused unchanged: it adds temporal pair features only for its stateful arm and includes branch nonlinearities, confounding input information with unit computation.

The new comparison tests a bundled membrane/spike/refractory mechanism, not each mechanism separately. Pure LIF is an isolated research unit, not evidence that the production dendritic neuron improves this task. Shared calendar effects are controlled with a calendar baseline and monthly robustness gate; client counts are never interpreted as independent replications.

The candidate is fixed as stateful LIF. Adoption requires a macro-client Brier improvement of at least 0.01 over the strongest scalar/baseline reference, improvement in at least nine months with no monthly degradation exceeding 0.02, a 0.01 advantage over event-reset LIF, exact scalar/compact parity and all resource caps. Frozen development readouts cannot learn from development labels. CPU measurements are not energy measurements.

[Next] Implement the cutoff-only streaming episode adapter and matched units. Verify delayed receipts, history/outcome separation, scalar parity, metrics and pre-work caps with synthetic fixtures before the one-shot train/development evaluation. Any adoption-gate failure ends this exact candidate without retuning. Deep credit, structural plasticity and held-out evaluation remain [Later].

## 日本語

[Done] learner比較protocolを凍結し、元のtaskと支持数reportのhashに結び付けました。候補評価は未実行です。scalar、compact event、stateful LIF、各eventでresetするLIFに、同じ時系列route・学習信号・資源上限を与えます。既存比較器のstateful限定の時系列特徴を持ち込まず、入力差との混同を避けます。

採用条件はclient均等重みのBrierで最強対照より0.01以上改善、12か月中9か月以上で改善、月次悪化0.02以内、reset対照より0.01以上改善、scalar／compactの完全一致、資源上限遵守です。これは膜電位・spike・refractoryの組合せの比較であり、各要素の個別効果ではありません。clientと暦の相関は残るため独立標本の主張をしません。

[Next] cutoffまでの履歴だけを渡すadapterとlearnerを実装し、synthetic境界テスト後に一度だけ評価します。失敗後の条件調整は禁止です。2014年は成功しても封印を維持します。

## 简体中文

[Done] 冻结learner比较协议，绑定原task和支持量report的hash；尚未评估候选。scalar、compact event、stateful LIF及每次event重置的LIF共享时序route、学习信号和资源上限。不沿用仅向stateful提供时序特征的旧比较器，以避免输入差异混淆。

采用条件包括：客户等权Brier比最强对照改善至少0.01，至少9个月改善且任何月份恶化不超过0.02，比reset对照改善至少0.01，scalar／compact完全一致，以及遵守资源上限。本实验检验膜电位、spike、refractory的组合，而非分别归因。客户与日历相关性仍存在，不主张独立样本。

[Next] 实现只读取cutoff之前历史的adapter与匹配learner，通过synthetic边界测试后执行一次评估。失败后不调整条件。即使成功，2014年仍封存。

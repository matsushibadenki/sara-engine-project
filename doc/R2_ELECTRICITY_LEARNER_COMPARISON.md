# Electricity matched learner comparison v1

## Completed v2 diagnostic — candidate closed

[Done] The one-shot corrected diagnostic completed under Python 3.10.18. Report SHA256: `ab5d5f19a2465d51a826881562763d501e4da1eb7029670c409fe275b3eaccaa`. Training included 164,965 episodes; development included 114,662 episodes across 349 clients, exactly matching the frozen support counts. Final report and started/completed receipts are under `workspace/evaluation/r2_electricity_development_v2*`; a retained result receipt is under `data/processed/benchmark_fixtures/r2_electricity_v2_result_receipt.json`.

| Arm | Development macro-client Brier | Balanced accuracy | Arm CPU seconds |
| --- | --- | --- | --- |
| Scalar / compact event | 0.248273 | 0.521347 | 7.099 / 6.871 |
| Stateful LIF | 0.247993 | 0.520303 | 45.913 |
| Event-reset LIF | 0.268166 | 0.504726 | 51.222 |

The strongest registered Brier reference was scalar. Improvement was only `0.000280325`, below the fixed `0.01` requirement. Primary gate failed; calendar (nine improved months), reset-control, parity and resource gates passed. Scalar/compact prediction and update traces are exactly equal. Peak reported learner state remained below 7,444 bytes, with at most 2,280 LIF ticks per episode. Held-out numeric parsing is zero. These single-run CPU measurements are not energy measurements or a replicated performance estimate.

**Decision:** close this exact v2 candidate without tuning encoding, parameters or thresholds on these scores. No production adoption or 2014 access. This is a negative exploratory diagnostic, not an untouched confirmatory result. Balanced accuracy also does not improve over scalar.

The reset-control interpretation is limited: with current 0.6 and threshold 1, resetting on every event makes comparison units unable to spike, leaving only calendar/bias routes. Winning against it does not establish a spike-specific benefit or separate membrane/refractory effects. [Next] Consolidate the retained evidence and design output-rate-matched non-spiking temporal controls with synthetic contract tests; any new scored hypothesis needs a separate registration and fresh confirmation boundary. Do not retry this exact task/candidate for promotion.

日本語: [Done] v2診断が完了しました。学習164,965件、開発114,662件・349clientで支持数と一致。LIFのBrierは0.247993、scalarは0.248273で、改善0.000280は固定基準0.01未満でした。主要gateは不合格、scalar／compactの予測・更新は完全一致。候補を終了し、再調整・本番採用・2014年閲覧は行いません。reset対照は比較routeを発火できなくするため、その勝利はspike固有の効果を意味しません。[Next] 出力頻度を揃えた非spiking時間状態対照をsynthetic境界で設計し、将来の確認は別の事前登録と新鮮な境界で行います。

简体中文：[Done] v2诊断完成。训练164,965次、开发114,662次／349客户，与支持量一致。LIF Brier为0.247993，scalar为0.248273；改善0.000280低于固定0.01门槛。主要gate失败，scalar／compact的预测与更新完全一致。关闭该候选，不调参、不采用、不访问2014年。reset对照无法让比较route发放，因此胜过它不证明spike特有优势。[Next] 在synthetic边界设计输出频率匹配的非spiking时间状态对照；后续确认必须另行预注册并使用新的确认边界。

## Disclosed corrective v2 contract

[Done] Corrective protocol v2 is frozen at SHA256 `11779a93583c66e510ddb3c639827aaff0d1985306d761d4ed9e52746d746533`. It explicitly changes the learning target to `2 * binary_target - 1` and converts the bounded signed score to probability using `(score + 1) / 2`. The initial probability is 0.5; no clipping or fitted calibration is used. All four arms receive identical targets and mappings. Task eligibility, source, encoding, local readout configuration, budgets, baselines and diagnostic thresholds remain as registered in v1.

V2 is an exploratory corrective diagnostic, not an untouched confirmation: the failure phase of v1 was not recorded, so 2013 is conservatively treated as previously accessed. The implementation always reports `adopted=false` and `confirmatory=false` for v2, even if all diagnostic gates pass. The held-out 2014 access boundary remains closed. V1's failure receipt and empty reservation remain untouched.

The new runner exclusively reserves a distinct report and syncs a started receipt before scoring. Exceptions produce a separate failed receipt without arbitrary exception-message payloads; the reservation prevents silent retries. On completion it records the report hash. Python version and implementation hashes are retained. CPU process time covers arm prediction/resolution, not source parsing, encoding or the whole evaluation process; state accounting covers learner state plus conservative transient LIF arrays, not archive buffers or metric bookkeeping. These measurements cannot establish energy savings or a replicated CPU advantage.

Synthetic tests cover signed endpoints, changing route subsets, scalar/compact parity, frozen development, unknown-contract rejection and durable consumed-attempt behavior. Related electricity tests: 29 passed. Real v2 execution results, if completed, are recorded separately below.

日本語: [Done] 訂正版v2を凍結しました。学習targetを増加+1・減少−1、確率を`(score+1)/2`と明示し、初期確率は0.5です。2013年は既閲覧として保守的に扱い、v2は診断専用で採用を許可しません。2014年の封印は維持します。実行開始・失敗・完了receiptとソースhashを保存し、黙った再試行を拒否します。関連テスト29件成功。CPU時間はarmの予測・更新範囲であり、消費電力ではありません。

简体中文：[Done] 已冻结纠正版v2。明确学习target为增+1、减−1，概率为`(score+1)/2`，初始概率0.5。保守地将2013年视为已访问，v2仅用于诊断，不允许采用；2014年继续封存。保存开始、失败、完成receipt及源码hash，拒绝静默重试。相关29项测试通过。CPU时间仅覆盖各arm的预测与更新，不代表能耗。

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

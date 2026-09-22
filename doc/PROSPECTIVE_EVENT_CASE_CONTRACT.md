# Case-level prospective event outcome contract

The reusable [`prospective_event_contract.py`](../src/sara_engine/evaluation/prospective_event_contract.py) is a research guard for future timestamped-event tasks. It does **not** score a model or retroactively validate earlier experiments.

An admissible case has timezone-aware, chronologically ordered feature-event times ending exactly at its decision cutoff. An outcome at or before cutoff is not prospective. A future outcome within the frozen horizon is positive only if it falls within verified follow-up for that case. A negative requires case-level observation through the entire horizon; a global dataset end or an unverified last event is rejected as follow-up evidence. Unobservable cases abstain. The return value contains only cutoff, horizon, label and feature-event count, not future event data or raw attributes.

`CaseFollowup.evidence` must be `explicit_terminal` or `continuous_case_observation`. This is an assertion by the caller, not proof supplied by the function. A dataset adapter must establish that the claimed endpoint genuinely belongs to the case and that its evidence type is justified. A terminal event before the horizon is conservatively censored unless the required horizon is covered. Outcome timestamps beyond the claimed follow-up are inconsistent and rejected. Source-specific task protocols still need to freeze decision event, horizon, terminal meaning, splits, class minimums, feature allowlist, and candidate controls **before** development scoring.

This contract addresses a concrete finding in the [BPI 2018 source gate](R2_BPI2018_SOURCE_GATE.md): the exploratory 180-day count gate used the log-wide final timestamp as a negative follow-up proxy. Its negative counts are not individually verified. The fixed count-gate result remains unchanged and failed independently because 2015 and 2017 had only one positive each.

## 日本語要約

新しいイベントタスクでは、特徴時刻が判断時点で止まり、結果がその後に起こり、陰性にはケースごとに確認した追跡期間があることを必須にします。データセット全体の最終時刻や根拠のない最終イベントは追跡証拠として拒否します。呼び出し側が申告するケース単位の追跡根拠そのものは別途監査が必要です。このガードは過去のスコアを遡及的に有効化しません。

## 简体中文摘要

新的事件任务必须保证特征时间止于决策时刻、结果发生在其后，且负例具有经核实的逐案例观察期。数据集整体结束时间或未经证实的最后事件不能充当随访证据。调用方声称的逐案例证据仍需独立审计；这一保护措施不会追溯性地使旧分数有效。

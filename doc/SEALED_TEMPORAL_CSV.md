# Sealed temporal CSV access

Prospective event experiments must admit a row in this order: raw case ID → frozen case membership and allowed split → event time → case-specific feature cutoff → descriptive feature cells. The reusable `SealedTemporalCsvReader` implements this order and returns only a hashed case ID, split, admitted event time, and explicitly allowlisted features. Outcome columns, raw IDs, and post-cutoff descriptive values are not returned.

The reader requires a frozen case manifest with timezone-aware cutoffs. Callers must supply an explicit forbidden-column list and a time parser; the reader rejects overlap between the feature allowlist and case/time/forbidden columns. Spy-row tests verify that excluded cases expose only the ID cell and future events expose only ID and time cells. This is a row-access guard, not a proof that a source has credible negative-label follow-up or sufficient prospective class support; those require the separate [case-level contract](PROSPECTIVE_EVENT_CASE_CONTRACT.md) and feasibility gate. Existing UCI scores were not rerun with this guard.

## 日本語

将来予測のイベント行は「ケースID → 凍結済みケースと分割 → イベント時刻 → ケース固有の判断時点 → 記述特徴」の順に確認します。`SealedTemporalCsvReader` は除外ケースや判断時点後の行について、記述列にアクセスしません。結果には生のケースIDや目的変数列を含めません。このガードだけでは負例の追跡可能性や必要件数は保証されないため、別のケース単位契約と実行可能性判定が必要です。既存のUCI結果は再実行していません。

## 简体中文

前瞻性事件行按“案例ID → 冻结的案例与分组 → 事件时间 → 该案例的决策截止时间 → 描述性特征”的顺序审核。`SealedTemporalCsvReader` 不会读取排除案例或截止时间之后事件的描述性字段，也不会返回原始案例ID或结果字段。该访问防护本身不能证明负例有可靠随访或样本量充足；仍需独立的案例级契约与可行性门槛。既有UCI结果没有重新运行。

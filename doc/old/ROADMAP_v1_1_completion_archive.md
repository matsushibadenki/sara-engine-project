# SARA Engine Roadmap

SARA Engine (Spiking Architecture for Reasoning and Adaptation) は、CPU-first / SNN-based / no-backprop-runtime の制約を守りながら、低消費電力・継続学習・疎イベント推論に競争力を作るための研究実装です。最上位KPIは単純な精度ではなく、**ANN系AIに対する performance-per-energy（成功・精度・適応性能あたりのイベント/エネルギーコスト）** です。

関連する運用文書は [SARA-Engine_Documentation_Hub.md](../SARA-Engine_Documentation_Hub.md) に整理しています。実装前のルールは [policy.md](../policy.md)、コマンド一覧は [TOOLS.md](../TOOLS.md)、リリース手順は [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) を参照してください。

## **開発スケジュール・ロードマップ (Master Schedule)**

今後の工程は、まず **リリース優先** で実用上の信頼性を固め、その後に **学習精度・推論精度の強化** へ進む方針とします。  
ANN系AIに正面から追随するのではなく、まず「CPU中心・低消費電力・バックプロパゲーション不要」というSNNの強みを保ったまま、出荷可能な品質を確立することを優先します。性能改善は、常に spike/event/update/search/replay/latency などの代謝コスト proxy とセットで評価し、人間の脳に近い疎・局所・省エネルギーな運用へ寄せます。

### **Architecture Integration Policy**

複数の研究アーキテクチャを導入しても、SARA Engine の設計は次の一本の背骨に統合する。  
**共通 sparse event protocol 上で、小さな専門サブモデル群が bounded state を持ち、局所更新・反実仮想 trace・解釈可能な監査ログを残しながら協調する脳型 runtime** を基本形とする。

* **共通インターフェース:** すべての新機構は、dense tensor を前提にせず、`SparseSpikeEvent`、event id、route trace、prediction error、correction event、memory steering event、stable digest などの小さなイベント表現へ落とし込む。  
* **小さく足す:** 新しい論文・アルゴリズムは、まず小さな primitive、trace builder、observed-only evaluator、または bounded state module として実装する。runtime の中核依存へ昇格する前に、テスト・summary・release/operational report へ伝播させる。  
* **採用基準:** JEPA / LeJEPA / linear RNN / SSM / RWKV系 / Mamba系 / predictive coding / sparse verifier / neuromorphic backend などから採用するのは、疎イベント処理、局所学習、継続記憶、解釈可能性、energy proxy、分散サブモデル統合を強める部分に限定する。  
* **非採用基準:** backpropagation runtime、GPU常駐の dense training、巨大 Transformer を中核にする設計、無制限 context、無制限 replay、巨大な隠れ状態、監査不能なブラックボックス推論は release-critical path に入れない。  
* **スケーリング方針:** 能力向上は単一巨大モデル化ではなく、専門サブモデル数、route品質、局所 memory、manifold retrieval、sleep/replay consolidation、event budget、hardware-aware event IR を増やすことで達成する。規模を増やしても、各モジュールは bounded state と明示的な入出力 trace を保つ。  
* **昇格ルール:** 新機構は observed-only から開始し、quality / success / energy proxy / state budget / traceability / regression trend を満たした場合のみ candidate へ昇格する。release gate の必須条件にするのは、複数履歴・反実仮想・運用レポートで安定性が確認された後に限る。  
* **実装の簡潔さ:** 研究探索であっても、最初の実装は小さく、読みやすく、テストしやすくする。重い抽象化や大規模依存を避け、既存の `project_paths`、Phase 3/4/5 evaluator、release soak、operational readiness の流れに沿って追加する。  
* **解釈可能性:** 重要な出力には、どの専門サブモデルが関与したか、どの memory / hypothesis / route が支えたか、反実仮想 lane とどう分離されたか、どの局所 credit が返ったかを trace として残す。  
* **policy順守:** 生成物は `data/`、`workspace/`、`models/` の管理ディレクトリへ限定し、root直下や任意ディレクトリへ ad hoc output を作らない。新しい出力経路は `src/sara_engine/utils/project_paths.py` の利用を優先する。  

この方針により、研究要素は「寄せ集め」ではなく、疎・局所・解釈可能・低電力・スケールしやすい SARA runtime の部品として統合する。

### **Primary Efficiency Objective**

* **最重要目標:** ANN系AIより高い performance-per-energy を限定タスクから順に実証し、最終的に脳型の疎イベント処理に近いエネルギー効率へ近づける。  
* **採用する評価軸:** `performance_energy_ratio_proxy`、`ann_cost_advantage_proxy`、`dense_embedding_ann_cost_advantage_proxy`、`sparse_event_cost_score`、`brain_efficiency_alignment_proxy`、実測可能になった段階で `joule_per_success`。  
* **実装ルール:** 新しい推論・記憶・ワールドモデル・学習機構は、精度だけでなく `success per event cost` または `quality per energy proxy` を改善または維持できる場合に優先する。  
* **避ける方向:** 精度を上げるために dense ANN 風の常時全探索、無制限コンテキスト拡張、GPU前提、backprop runtime、巨大な再学習を必要とする実装を release-critical path に入れない。  
* **脳型効率へ近づける実装優先度:** sparse routing、temporal compression、idle replay、sleep consolidation、metabolic budget、structural pruning、Nested Learning memory scheduling、event-level ANN baseline comparison。  
* **段階ゲート:** `ann_efficiency_roadmap_gate.py` で、(1) sparse proxy instrumentation、(2) limited real-data advantage、(3) scale-ladder advantage、(4) strict operational regression guard、(5) neuromorphic transfer readiness、(6) real joule measurement readiness を順に確認し、ANN比の accuracy-per-energy 目標に直結しない実装を昇格させない。  
* **反証ゲート:** 実データで勝つだけでなく、absent-query / partial-evidence / contrastive near-miss control で「証拠がない時、一般語だけが一致する時、似た文書で希少な決定語だけが違う時」の挙動を確認する。dense scan が何かを過選択する場面でも、SARA側は no-hit / weak-evidence を trace 付きで返し、near-miss では希少な決定語を common overlap より先に処理できることを ANN効率ロードマップの Stage 2/3 で必須化する。  

### **Current Status Snapshot (2026-05-16)**

現時点のSARA Engineは、基盤研究フェーズだけに留まらず、限定用途での practical CPU-first SNN assistant としてかなり使える段階まで進んでいます。とくに Rust コア、保存形式、CLI、release gate、lightweight accuracy suite、memory health diagnostics は高い完成度に達しています。

#### **実装がかなり進んでいる領域**

* **Release / Operations:**  
  * `release_soak.py`、`release_gate.py`、release checklist により、配布前の acceptance gate が定着。  
  * `quick` / `release` / `extended` profile を持つ CPU-only soak 運用が可能。  
  * release summary に retrieval hygiene と conversational readiness を集約可能。  
  * `stage_b_readiness` を release gate に接続し、world-model prototype の最低条件を shipping gate で検証可能。  
  * `stage_e_readiness` を release gate に接続し、common spike space / temporal compression / dendritic context gate / lightweight Spiking H-JEPA trace の最低条件を shipping gate で検証可能。  
* **Artifact / Memory Management:**  
  * `stable_v1` context encoding、indexed memory artifact、TurboQuant 対応 save/load が実装済み。  
  * `inspect-memory`、`upgrade-memory`、`build-replay-data` により、旧 artifact の点検・移行・再構築が可能。  
  * memory health report で `conversational_readiness`、`session_memory_snapshot`、`diagnostic_memory_hits`、`predictor_state_snapshot`、`future_state_runtime_state` を確認可能。  
* **Lightweight Practical Dialogue:**  
  * `SaraInference` に fast intent path、practical fallback、session memory、next-step suggestion を実装。  
  * `name`、`location`、`origin`、`preference`、`goal`、`task` などを会話内 short-term memory として保持可能。  
  * 英語・日本語の軽量 practical 応答と、`goal + task` に基づく next-step assistance を実装済み。  
  * `predictor_state` と `future_state_runtime_state` により、future-state prediction、短期遷移追跡、shift-aware next-step response を持てる状態になった。  
* **Evaluation / Quality Tracking:**  
  * `SaraAgent` / `SaraInference` / `SpikingLLM` の lightweight benchmark 群が揃い、Phase 3 accuracy suite で一括評価可能。  
  * `retrieval_hygiene`、trend/delta、release summary 連携まで実装済み。  
  * `SaraAgent` の retrieval blending に explicit intent priority を追加し、照応質問（例: 「それを書くメリット」）でも広い文脈より質問意図を優先して安定想起できるようにした。Phase 3 suite では `response_keyword_recall=0.9167`、`retrieval_grounding>=0.43`、completion gate PASS を確認済み。  
  * `adaptive_readiness`、`predictive_readiness`、`efficiency_readiness`、`direction_shift_following` を継続観測可能にした。  
  * `energy_efficiency_benchmark` を performance-per-energy 中心へ拡張し、`ann_cost_advantage_proxy` と `brain_efficiency_alignment_proxy` を Phase 3 focus で継続観測可能にした。  
  * Phase 5 entry として、Spiking H-JEPA の lightweight latent-transition benchmark / gate を追加し、prediction error、correction event、anti-collapse diversity、counterfactual separation を検証可能にした。  
  * world-model spatial benchmark に top-down projection、room adjacency、door connectivity、route planning、affordance selection、route execution、invalid-action rollback を追加し、2D観測から内部空間状態・行動結果を扱う最低条件を Stage B readiness へ接続済み。  
  * human-readable summary と release soak summary の双方で、`Dialogue Shift Detail`、`Predictive Detail`、predictive command/shift trend を追跡可能にした。  
* **Training Material Management:**  
  * corpus DB に `category`、`quality_score`、`source_version`、`is_active` を持たせ、素材管理を一元化。  
  * `db-import`、`db-status`、`db-list`、`db-export`、`db-activate`、`db-deactivate`、`db-reset` を整備。  
  * `category` / `source` / `quality_score` / `active state` ベースの preview、dry-run、採用切り替えが可能。  

#### **現状で限定的に実用化できる領域**

* CPU-only 環境での lightweight dialogue assistance  
* 継続的な memory inspection と artifact health review  
* 管理された training material curation と export  
* release candidate の lightweight operational validation  
* `goal/task` と future-state prediction に基づく lightweight next-step guidance  
* 方針転換追従と predictive command stability を含む release-time observability  

#### **まだ中長期課題として残っている領域**

* 大規模 continual learning の本格実証  
* 数千万〜億単位を見据えた scale-out / distributed event routing  
* Spiking H-JEPA の本格実装  
* 本格的な multimodal integration  
* 汎用会話品質の大幅向上  
* 学習素材レビュー、近似重複排除、承認フロー、サンプリング最適化などの高度な dataset governance  

#### **進捗感の要約**

* **Foundation / Rust Core / Serialization / CLI / Release Gate:** 高  
* **Practical Dialogue / Session Memory / Next-Step Assistance / Predictive Runtime:** 中高  
* **Training Material Management / Data Operations:** 中高  
* **Large-Scale Intelligence / Multimodal / Advanced Predictive Coding:** 中  
* **Observability for Adaptive / Predictive Runtime:** 中高  

### **Phase 1: Foundation & Rust Core Acceleration (完了/最適化中)**

* **目標:** 生物学的学習則（STDP等）の確立と、Rustによる超高速なイベント駆動シミュレータの統合。  
* **成果:** `sara_rust_core` によるバックエンド統合、SNN Transformer、Spatiotemporal STDPモジュールの完成。CPU上での高速推論の達成。  
* **位置づけ:** 以後のリリース安定化と精度改善の土台フェーズ。

### **Phase 2: Release Readiness & Practical Stability (実装完了 / 運用継続)**

* **目標:** 現行のSNN基盤を「試作」から「安全に配布できるプレリリース品質」へ引き上げる。  
* **完了済みの項目:**  
  * direct memory の保存・復元経路の統一と unsafe `eval()` の除去。  
  * `SaraAgent` の runtime diagnostics、セッション永続化、CLI 診断表示の追加。  
  * FORCE artifact の入力検証、`UnifiedSNNModel` の二重更新バグ修正。  
  * 軽量 soak test、CLI dispatch test、release metadata test の追加。  
  * `scripts/old/` の legacy 扱いの明文化。  
  * `sara-chat` / `sara-train` の end-to-end カバレッジ拡張と、training runtime 差し替え可能な CLI 分割。  
  * `release_soak.py` / `release_gate.py` による配布前チェックリスト運用の固定。  
  * `scripts/eval/operational_readiness.py` を追加し、Phase 3/4 completion と release gate を単一レポートで統合判定できる運用導線を実装。  
  * `operational_readiness.py --strict-production` を追加し、extended 相当の soak 条件（`shipping_ready`, duration/turn/iteration 下限）を最終昇格条件として機械判定できるようにした。  
  * `operational_readiness` の report / summary に Stage B promotion snapshot（candidate readiness、連続通過数、required streak、recommended、next step hint/actions）を追加し、world-model prototype の昇格判断を最上位レポートで即時監査できるようにした。  
  * `operational_readiness.recovery_actions` を優先度付き（`priority`/`expected_effect`/`affected_checks`）へ拡張し、Stage B promotion 推奨時の follow-up（hint/actions）を repair log 記録コマンドとして自動提案できるようにした。  
  * `operational_readiness` に `iterative_repair_plan`（`completed`/`stalled`/`stop_reason`/`next_step_hint`/`next_actions`）を追加し、失敗時の次コマンドを最上位レポートから機械可読で辿れるようにした。  
  * `operational_readiness` に `--repair-log-path` と execution log 連携を追加し、成功済みコマンドを iterative plan から自動除外して `iteration`/`executed_steps`/`successful_steps` を進捗として反映できるようにした。  
  * `operational_readiness` に `repair_plan`（selected/covered/uncovered/fallback/coverage_ratio）と `--repair-plan-path` を追加し、運用レポートとは別に repair artifact を保存・再利用できるようにした。  
  * `operational_readiness` に `error_details` / `error_details_summary` / `failure_focus` を追加し、失敗カテゴリの集中度と優先修復アクションを最上位レポートで機械可読に追跡できるようにした。  
  * `collect_operational_readiness_artifacts` を追加して recovery/repair/focus/iterative を再利用可能な単一インターフェースへ統合し、CLI失敗時にも `failure_focus` と `iterative next actions` を即時表示できる運用導線にした。  
  * `operational_readiness` に `--record-repair-command` / `--record-repair-status` / `--record-repair-checks` / `--record-repair-source` を追加し、repair log への記録・pending完了反映を単体で実行できるようにした。  
  * `operational_readiness` に retry queue（`repair_retry_queue` / `repair_retry_cooldown_blocked`）を追加し、`--retry-max-attempts` / `--retry-cooldown-seconds` で再試行候補とcooldown待ち候補を機械可読に抽出できるようにした。  
  * `operational_readiness` に `--auto-dispatch-retry` を追加し、retry queue から pending へ再試行コマンドを自動投入して再評価できる半自動 repair loop を実装した。  
  * `operational_readiness` に `--append-iterative-next-actions` と `--pending-ttl-seconds` を追加し、iterative next actions の pending 追記と stale pending の timeout 化を同一ループ内で管理できるようにした。  
  * `operational_readiness` に `--repair-complete-command` / `--repair-complete-status` / `--repair-complete-checks` / `--repair-complete-source` を追加し、pending修復の完了反映をCLIから明示実行できるようにした。  
  * report/summary に `repair_pending_count` と `repair_timeout_count` を追加し、修復キュー滞留の監査を最上位レポートで追跡できるようにした。  
  * `operational_readiness` の `--auto-dispatch-retry` を拡張し、`--auto-dispatch-min-priority` / `--auto-dispatch-diversify-checks` / `--auto-dispatch-max-per-check` による優先度しきい値・チェック分散選択付きの半自動 retry dispatch を実装した。  
  * `operational_readiness` の retry/cooldown 出力を priority score/tier 付きに統一し、summary に auto-dispatch の実行コマンド・スキップ理由別コマンド（pending/limit/low-priority/check-quota）を表示できるようにした。  
  * `operational_readiness` に `operational_checklist`（managed output path 検証、report/summary review readiness）を追加し、report/summary/repair artifact の監査可能性を最上位レポートで機械判定できるようにした。  
  * `operational_readiness` の refresh失敗早期終了分岐でも `operational_checklist` と repair artifact を保存するようにし、途中失敗時でも監査証跡が欠けないようにした。  
  * `operational_readiness.main` の再評価反映処理を `_apply_operational_evaluation_to_output` へ共通化し、auto-dispatch/iterative追記後の重複 `output.update` を解消して保守性と更新漏れ耐性を高めた。  
  * `operational_readiness` summary に `error_detail_count` / `error_detail_total` / type・category別カウントを追加し、失敗傾向を単一サマリーで追跡できるようにした。  
  * `operational_readiness` の repair artifact 生成を `_build_operational_repair_artifact` に統合し、`operational_checklist` 反映後の最終スナップショットを report/summary/repair artifact で一貫保存できるようにした。  
  * `operational_readiness` に runbook 出力（`--runbook-path` / `build_operational_runbook`）を追加し、`failure_focus`・`iterative next actions`・`retry queue` を実運用向け Markdown に統合して保存できるようにした。  
  * `operational_checklist` の managed path 検証対象を runbook まで拡張し、report/summary/repair artifact/runbook の4点を同一基準で監査できるようにした。  
  * `operational_readiness` に runbook action manifest（`--runbook-actions-path` / `build_operational_runbook_actions`）を追加し、`iterative next actions`・`retry queue`・`fallback_actions` を重複排除して優先順 JSON として保存できるようにした。  
  * runbook 本文に `Execution Manifest` セクションを追加し、運用者が Markdown から即時に実行対象コマンド（source/priority付き）を確認できるようにした。  
  * `operational_checklist` の review readiness を runbook action manifest（`runbook_actions` list と managed path）まで拡張し、監査対象を report/summary/repair artifact/runbook/actions の5点へ統一した。  
  * `operational_readiness` に `--append-runbook-actions` / `--append-runbook-actions-max` / `--append-runbook-actions-min-priority` を追加し、runbook action manifest を pending repair log へ優先度フィルタ付きで安全投入できるようにした。  
  * `append_operational_runbook_actions_to_repair_log` を追加し、既存 pending との重複回避、優先度しきい値、投入件数上限を満たした半自動運用ループ（plan→queue）を実装した。  
  * `v1_release_gate` report に `readiness_score`、`category_summary`、`failure_focus` を追加し、Stage B/D/E/Phase5/operational/version などの出荷カテゴリ別に PASS 率と主要失敗軸を機械可読に監査できるようにした。  
  * `v1_release_gate` report に `recovery_actions` を追加し、v1 最終判定で失敗したカテゴリに応じて Phase3 / Phase4 / Stage B-D-E / Phase5 / operational / version の再検証・修復コマンドを優先度付きで提示できるようにした。  
  * `v1_release_gate` の managed artifact に `v1_release_gate_actions.json` を追加し、失敗カテゴリの復旧アクションを重複除去・優先度順で機械可読に取得できるようにした。  
  * `operational_readiness` の `runbook_actions` 生成に `--v1-actions-path` を追加し、`v1_release_gate_actions.json` の復旧アクションを operational runbook action manifest へ重複除去付きで統合できるようにした。  
  * `v1_release_gate_actions.json` の各 action に `generated_at` を付与し、`operational_readiness --v1-actions-max-age-seconds` で鮮度フィルタを適用して stale/missing-timestamp action を除外できるようにした。  
  * `operational_readiness` の runbook action manifest 生成に v1 action hygiene check を追加し、stale/missing-timestamp が検出された場合は `python scripts/eval/v1_release_gate.py` を high-priority action として自動提示できるようにした。  
  * `operational_readiness` に `--ann-efficiency-roadmap-report-path` を追加し、ANN efficiency roadmap の `next_evidence_actions` を `ann_efficiency_next_evidence` source として runbook action manifest へ統合できるようにした。  
  * `operational_readiness` に `--runbook-max-per-source` を追加し、`iterative_next_action` / `retry_queue` / `fallback_action` / `v1_recovery_action` の source 偏りを抑えて実行計画を分散できるようにした。  
  * `runbook_action_summary`（total/source_counts/priority_counts）を report/summary へ追加し、runbook action manifest の分布を最上位サマリーで監査できるようにした。  
  * runbook action build stats（considered/appended/skipped_duplicate/skipped_source_cap/skipped_max_actions）を追加し、manifest の候補落ち要因を運用サマリーから追跡できるようにした。  
  * runbook Markdown の `Execution Manifest` に candidate/skip counters を追加し、現場運用で JSON を開かずに候補落ち理由を確認できるようにした。  
  * source cap の skip 理由を `skipped_source_cap_by_source` で保持し、どの action source が上限に達したかを summary/runbook で追跡できるようにした。  
  * `operational_readiness` に `--runbook-max-actions` を追加し、runbook action manifest の総件数上限を CLI から制御できるようにした。  
  * `build_operational_runbook` は report に保存済みの `runbook_actions` を優先利用するようにし、CLIで指定した runbook 上限制御との不整合を防ぐようにした。  
  * runbook `Execution Manifest` に configured `max actions` / `max per source` を表示し、どの設定で候補が間引かれたかを運用記録に残せるようにした。  
  * `skipped_max_actions_by_source` を追加し、総件数上限で落ちた候補の source 偏りを summary/runbook で追跡できるようにした。  
  * `skipped_duplicate_by_source` を追加し、重複除外が特定 source に偏っていないかを summary/runbook で監査できるようにした。  
  * `skipped_empty_command_by_source` を追加し、空コマンド候補の流入元を summary/runbook で追跡できるようにした。  
  * runbook action build rates（drop/duplicate/empty/source-cap/max-actions）を追加し、候補落ち圧を実行間で比較しやすくした。  
  * `operational_checklist` に `runbook_drop_rate_ok` を追加し、drop rate が高すぎる runbook manifest を監査上の警告として検知できるようにした。  
  * `--runbook-drop-rate-threshold` を追加し、manifest drop rate 警告の感度を運用ポリシーに合わせて調整できるようにした。  
  * `runbook_drop_rate_ok=false` の場合に `runbook_drop_rate_recovery` action を runbook manifest へ自動追加し、制約緩和付き再評価コマンドへ直結できるようにした。  
  * README / release notes / packaging metadata の継続整備。  
* **残タスク / 運用継続:**  
  * wall-clock を長めに取った soak run の運用基準策定と `extended` profile の実運用定着。  
  * release profile / extended profile の定期実行フローを CI または手順書として固める。  
  * tag release workflow（`.github/workflows/release.yml`）に publish 前 contract test を固定し、`requirements-ci-release.txt` で依存を集中管理して「テスト未検証 publish」を防ぐ。  
* **ROADMAP closure audit（2026-05-30）:**  
  * DONE: release-critical path は Phase 3/4/5、Stage B/C/D/E/F、release soak、operational readiness、v1 release gate の主要 contract / summary / recovery action が接続済みで、未接続項目は release-blocking ではなく operational cadence として扱う。  
  * DONE: observed-only 研究候補は minimum gate へ即時昇格せず、acceptance candidate、promotion review、research journal、bounded evidence loop、trend / regression watchlist のいずれかへ分類済みとする。  
  * DONE: long-term research backlog（大規模 continual learning、scale-out、multimodal 本格化、汎用会話品質、大規模 neuromorphic 実機 adapter）は、現時点の「未完了バグ」ではなく、次世代研究フェーズへ送る non-blocking backlog として明示する。  
  * DONE: roadmap completion audit は `scripts/eval/roadmap_completion_audit.py` で機械確認し、closure marker / unchecked marker / backlog category をテストで固定する。  
  * DONE: research product completion gate は `scripts/eval/research_product_completion_gate.py` で、policy / ROADMAP closure / Phase 3-5 / strict operational readiness / ANN efficiency roadmap / energy measurement session plan / memory repair operation / managed output boundary / neuromorphic HAL smoke を単一の完成度ゲートとして監査する。  
* **完了条件:**  
  * リリースチェックリストの主要項目を継続的に満たせる。  
  * CPU-only 環境で回帰テストと soak run が安定完走する。  
  * モデル保存・復元・CLI 導線・診断導線に致命的不整合がない。

### **Phase 3: Accuracy Uplift for Learning & Inference (completion gate achieved)**

* **目標:** 学習精度・推論精度を段階的に引き上げ、用途を絞った領域で ANN 系に見劣りしない水準を目指す。  
* **基本方針:**  
  * 誤差逆伝播なし、密な行列演算なし、GPU必須なしの制約は維持する。  
  * 「総合汎用性能」で ANN を追うのではなく、「CPU上のワット当たり性能」「少量データ適応」「常時稼働」で勝てる設計を優先する。  
* **主要タスク:**  
  * `SaraAgent` / `SaraInference` / `SpikingLLM` の評価指標を整備し、品質改善を数値で追えるようにする。  
  * direct memory と LTM / hippocampus の役割分担を整理し、誤想起・ノイズ想起を減らす。  
  * topic tracking, routing, retrieval, readout を精密化し、会話品質と応答一貫性を改善する。  
  * FORCE / reservoir / JEPA 系の比較ベンチを作り、タスク別に最も精度の高い更新則を選別する。  
  * One-shot / few-shot / continual learning の評価で、少数データ条件下の強みを定量化する。  
* **重点対象タスク:**  
  * テキスト分類、トークン分類、時系列予測、異常検知、軽量エージェント対話。  
  * 画像・音声は「完全汎用生成」より、分類・連想・予測補助から優先する。  
* **完了条件:**  
  * 少なくとも限定タスク群で、ANN 系と比較可能な精度/安定性/電力効率のレポートが揃う。  
  * 推論品質が「省エネルギーの代償として大きく劣る」状態から脱却する。
* **進行中の実装:**  
  * `AgentDialogueEvaluator` と `scripts/eval/agent_dialogue_benchmark.py` により、`response_keyword_recall`、`fallback_control`、`retrieval_grounding` を lightweight benchmark として継続観測可能にした。  
  * `InferenceSequenceEvaluator` / `SpikingLLMSequenceEvaluator` と `scripts/eval/inference_accuracy_benchmark.py` / `scripts/eval/spiking_llm_accuracy_benchmark.py` により、`SaraInference` と `SpikingLLM` の one-shot / fuzzy retrieval / continual retention / short streaming を CPU-only で継続観測可能にした。  
  * `scripts/eval/phase3_accuracy_suite.py` により、`SaraAgent` / `SaraInference` / `SpikingLLM` をまとめた lightweight accuracy gate を運用できる状態にした。  
  * `SaraAgent` 側に `off_topic_suppression`、`multi_turn_consistency`、`retrieval_stability` を追加し、会話品質と想起安定性の内訳を継続観測可能にした。  
  * `SaraInference` / `SpikingLLM` 側に noise robustness 系指標を追加し、少量記憶・継続学習・ノイズ混入時の品質を同一フレームで観測可能にした。  
  * `retrieval_hygiene` focus と trend/delta 表示を phase3 suite / release soak summary に追加し、品質変化を前回比で追える状態にした。  
  * `task_switch_adaptation_benchmark.py`、`future_state_consistency_benchmark.py`、`energy_efficiency_benchmark.py` を追加し、Stage A の `adaptive_readiness`、`predictive_readiness`、`efficiency_readiness` を計測可能にした。  
  * `direction_shift_following` を `AgentDialogueEvaluator` に追加し、話題転換・方針転換後の追従性を summary / release summary の両方で detail と trend 付きで追える状態にした。  
* **現在の到達点:**  
  * lightweight benchmark と aggregated suite が通る状態まで実装済みで、Phase 3 の baseline instrumentation と品質 gate は一段完了。  
  * `phase3_completion`（overall/trend/focus/stage B-C-D minimum を束ねた completion gate）を追加し、Phase 3 完了条件を機械可読に判定できるようにした。  
  * `scripts/eval/phase3_accuracy_suite.py` 実行結果（`workspace/evaluation/phase3_accuracy_suite.json`）で `phase3_completion.passed = true` / `completion_score = 1.0` を確認し、`scripts/eval/phase3_completion_gate.py` でも CLI gate 通過を確認した。  
  * `phase3_completion_gate.py` を強化し、`passed` だけでなく `completion_score>=1.0` と `checks` map 内の未達も検出するようにした。Phase5 着手前に Phase3 の表面通過だけでなく内訳の完全達成を確認できる。  
  * Phase3 trend tracking に `gate_regression_count` を追加し、raw parameter-efficiency diagnostic（例: `average_quality_per_mb`）の観測は維持しつつ、release-blocking regression 判定は正規化済み gate metric に限定できるようにした。  
  * Phase 3 の観測軸は「正答率」だけでなく「誤想起抑制」「話題維持」「想起安定性」まで広がり、release summary 上でも追跡可能になった。  
  * 現在はさらに、direction shift・predictive command integrity・future-state shift tracking まで trend/delta 付きで追跡可能になっている。  
  * より広い ANN 比較、タスク拡張、長期 continual learning の大規模検証は Phase 4 以降の拡張テーマとして継続する。  

### **Phase 4: Scale-out & Continuous Learning (completion gate achieved)**

* **目標:** 数千万〜億単位のニューロン規模へのスケールアップと、破局的忘却のないオンライン学習の実証。  
* **タスク:**  
  * 動的構造変更（Structural Plasticity）の安定化とスケーラビリティ向上。  
  * LTM (Long-Term Memory) と海馬モジュール間の知識転送・記憶固定化のアルゴリズム強化。  
  * マルチコア・分散環境でのイベントルーティングの最適化と、Rust側の非同期処理の強化。  
  * 限定タスクで確立した高精度化手法を、大規模構成へ移植する。  
* **着手済みの内容:**  
  * `SaraAgent` に `memory_role` / `stability_score` を導入し、semantic / episodic の軽量な役割分担と retrieval stabilization を実装。  
  * `SparseMemoryStore.search()` と `CorticoHippocampalSystem.in_context_inference()` に query metadata ベースの context-aware pruning を追加し、hippocampus / LTM 前段でのノイズ想起抑制を開始。  
  * `SaraInference` の direct memory fuzzy match に suffix continuity・drift penalty を加え、direct memory / hippocampus / LTM の三層で retrieval hygiene を揃え始めた。  
  * Agent / Inference の diagnostics を共通フォーマットへ統一し、安定化内訳を同じ view で保存・表示・評価できる状態にした。  
* **完了条件:**  
  * 精度改善とスケール改善が両立し、継続学習時の品質劣化が許容範囲に収まる。
* **完了判定の実装:**  
  * `scripts/eval/phase4_scale_continual_benchmark.py` を追加し、`structural_plasticity_stability` / `hippocampal_transfer_integrity` / `scale_out_retention_integrity` / `continual_drift_recovery_integrity` を CPU-only で一括検証できるようにした。  
  * `scripts/eval/phase4_completion_gate.py` を追加し、`phase3_completion` 通過済みであることを前提に、Phase 4 必須メトリクスの minimum gate（すべて 1.0）を機械判定できるようにした。  
  * `workspace/evaluation/phase4_scale_continual_benchmark.json` を標準成果物とし、Phase 4 の完了可否を CLI で再現可能にした。  
  * `phase4_scale_continual_benchmark.py` に `quality_metrics`（structural synapse ratio、hippocampal score retention、scale-out retention/latency、continual drift/recovery）を追加し、Phase4 の精度・ロジック品質を数値で監査できるようにした。  
  * `phase4_completion_gate.py` を強化し、`metrics==1.0` だけでなく `threshold_results` と `quality_metrics` の下限/上限も検証するようにした。Phase4 の薄い成功レポートや遅延・保持率の劣化を completion gate で検出できる。  
  * `scripts/eval/phase4_operational_cycle.py` を追加し、`release` / `extended` の運用サイクル（`operational_readiness --refresh-artifacts` + strict-production 経路）を単一 CLI で実行・記録できるようにした。  
  * `phase4_operational_cycle.py` から `operational_readiness` の runbook 制御パラメータ（`--runbook-max-actions` / `--runbook-max-per-source` / `--runbook-drop-rate-threshold` / `--v1-actions-max-age-seconds`）を透過的に指定できるようにし、Phase4 の定期運用ジョブでも manifest 偏り・drop rate 警告感度・v1 action 鮮度を同一ポリシーで管理できるようにした。  
  * `workspace/release/phase4_operational_cycle_report.json` / `phase4_operational_cycle_summary.txt` を追加し、Phase 4 の定期運用結果を managed path へ継続保存できるようにした。  
  * `.github/workflows/phase4-operational-cycle.yml` を追加し、日次 schedule は dry-run で契約監視、手動 dispatch は full cycle 実行を選択可能にして、release/extended の定期実行フローを CI へ固定化した。  
  * Phase4 運用CIの依存管理を `requirements-ci-phase4.txt` へ分離し、`numpy` / `msgpack` / `transformers` / `matplotlib` を workflow 直書きせず再利用できる形にして、import chain 由来の段階的 CI 失敗を再発しにくくした。  
  * `scripts/eval/release_gate.py` と `scripts/eval/operational_readiness.py` の `sara_engine` 直接importを段階的に除去し、stage contract / project paths をファイル直接ロードへ切り替えて、`__init__` 経由の重依存連鎖（audio/visualizer/transformers等）でCIが落ちるリスクを低減した。  
  * `src/sara_engine/__init__.py` を eager import 方式から lazy export 方式へ刷新し、`import sara_engine` 時に全サブモジュールを即時ロードしない設計へ移行した。これにより gate/benchmark 系のスクリプトが package-level import 副作用で失敗するリスクをさらに下げた。  

### **Phase 5.5: Real-Data Curriculum Scaling (new)**

* **目標:** roadmap実装のゲート整備後に、実データ学習を `small -> medium -> large` の段階で安全に拡張し、精度と energy proxy を同時に監査する。  
* **進行中の実装:**  
  * `scripts/train/run_real_data_curriculum.py` を追加し、`db-export -> train-self-org -> train_snn_lm -> gates` を stage別プロファイル（small/medium/large）で一括実行できるようにした。  
  * `small` は quality重視の pilot 学習 + Phase3/5 検証、`medium` は Phase4 completion を追加、`large` は strict operational readiness まで含める構成にした。  
  * `scripts/sara_cli.py train-curriculum` を追加し、既存統合CLIから stage/dry-run/skip-gates を指定して同導線を実行できるようにした。  
  * `preflight-only` と curriculum preflight report を追加し、コーパスDBの存在、active素材数、品質しきい値適合件数、stage推奨件数を学習前に検査できるようにした。  
  * managed artifact として `workspace/reports/real_data_curriculum_<stage>.json` を保存し、各段階の実行コマンドと成功/失敗を機械可読に追跡できるようにした。  
  * `train-self-org` の curriculum 経路を `data/processed/corpus.txt` に揃え、旧 `data/corpus.txt` の大規模残骸を誤って読む問題を解消した。`small` 実行は 890 material / 1 epoch / Phase3 completion / Phase5 entry+completion gate まで通過済み。  
  * Phase3 trend は curriculum stage別 history に分離し、pilot 実行が既存評価履歴の微小揺れで失敗しないようにした。CLI も内部 runner の終了コードを伝播するため、CI/運用で失敗を取りこぼさない。  
  * `real_data_external_validity.py` を追加し、同じ実コーパス由来タスクで SARA の sparse event retrieval と ANN風 dense-scan proxy を比較できるようにした。`real_data_qa_accuracy`、`real_data_summary_keyword_coverage`、`continual_memory_hit_rate`、`performance_energy_ratio_proxy`、`ann_cost_advantage_proxy` を同一reportへ保存する。  
  * `real_data_external_validity_history.json` と `trend.no_regressions` を追加し、QA/要約/継続記憶の絶対劣化と ANN比energy advantage の相対劣化を継続監視できるようにした。  
  * external validity report/history に corpus/task fingerprint を保存し、コーパスやタスク条件が変わった場合は比較を安全にスキップして偽の回帰検知を避けるようにした。  
  * external validity report に `thresholds` / `check_details` を追加し、各チェックの値・閾値・比較状態を JSON 単体で監査できるようにした。  
  * curriculum 経路では `real_data_external_validity_<stage>_history.json` に履歴を分離し、small/medium/large の規模差による偽回帰を避けながら stage別の external-validity trend を追跡できるようにした。  
  * `train-curriculum` の Phase5 gate 後に external validity benchmark を stage別規模（small=256/24、medium=1024/64、large=4096/128）で接続し、実データ学習の次段階へ進む前に ANN比の energy advantage が崩れていないかを確認する。  
  * `release_gate` / `operational_readiness --refresh-artifacts` / `v1_release_gate` に `real_data_external_validity.py` を接続し、通常release・strict operational・v1最終判定の全層で実データQA・要約・継続記憶・ANN比energy advantageの回帰を出荷ブロッカーとして検出できるようにした。  
  * `real_data_external_validity_ladder.py` を追加し、small/medium/large の外部妥当性を一括実行して、各profileのQA・要約・継続記憶・ANN比energy advantageを集約できるようにした。  
  * ladder report は `min_real_data_qa_accuracy`、`min_ann_cost_advantage_proxy`、`min_performance_energy_ratio_proxy`、`all_profiles_passed`、`large_profile_present`、`no_trend_regressions_all_profiles` を持ち、規模拡張時の最悪値を gate/summary で直接確認できる。  
  * `scripts/sara_cli.py eval-external-validity-ladder` を追加し、統合CLIから任意profile（`name:max_docs:max_cases`）または標準small/medium/largeラダーを実行できるようにした。  
  * `ann_efficiency_roadmap_gate.py` と `scripts/sara_cli.py eval-ann-efficiency-roadmap` を追加し、energy benchmark・external validity・scale ladder・strict operational・neuromorphic readiness を単一の staged roadmap gate として評価できるようにした。  
  * `operational_readiness --refresh-artifacts` に ladder 実行を追加し、strict運用時に単発ベンチだけでなく scale ladder の最小 ANN比・最小 performance-energy ratio が崩れていないかを確認する。  
  * SARA側 retrieval を `metabolic_sparse_rarity_early_stop_verified_fallback` へ更新し、希少語優先・確信時早期停止・難ケースのみ全候補再確認の3段構成にした。実コーパス ladder では精度を維持したまま、最小 `ann_cost_advantage_proxy` / `performance_energy_ratio_proxy` が `147.98x` から `446.95x` へ改善した。  
  * external validity report に `sara_metabolic_cost_reduction_proxy`、`sara_metabolic_early_stop_rate`、`sara_metabolic_avg_processed_query_tokens`、`retriever_strategy` を追加し、脳型の「必要な時だけ追加計算する」効率改善を監査可能にした。  
  * external validity report に absent-query negative control を追加し、`negative_control_abstention_integrity`、`negative_control_ann_overselection_observed`、`negative_control_cost_advantage_proxy` を保存する。SARA側は no-hit query で `-1` を返し、ANN風dense proxyの過選択と比較して低コスト abstention を監査できる。  
  * partial-evidence negative control を追加し、一般語だけが一致して決定的な希少語が存在しない場合は `best_match_ratio < min_match_ratio` として abstain する。`partial_evidence_abstention_integrity` と `partial_evidence_cost_advantage_proxy` を external validity / ladder / ANN efficiency roadmap gate へ接続した。  
  * contrastive near-miss control を追加し、共通語が多い類似文書間で希少な決定語を先に処理し、正しい文書を選びつつ dense scan より低い event cost に抑えることを `contrastive_control_accuracy`、`contrastive_control_rare_decider_first_rate`、`contrastive_control_cost_advantage_proxy` として監査する。  
  * dense embedding ANN-style baseline を追加し、評価専用の hashed-vector cosine baseline と SARA sparse routing を比較する。dense vector は production runtime path には入れず、`dense_embedding_ann_proxy_qa_accuracy` と `dense_embedding_ann_cost_advantage_proxy` を external validity / ladder / ANN efficiency roadmap gate へ接続した。  
  * `energy_measurement_readiness.py` を追加し、`data/raw/energy_measurements.jsonl` に記録した SARA/ANN の `joules` と `success_count` から `joule_per_success` と実測 ANN比を計算できるようにした。現時点で実測行がない場合は `protocol_ready_pending_measurements` として、proxy-only 主張と real-joule evidence を明確に分離する。  
  * `scripts/sara_cli.py record-energy-measurement` を追加し、`run_id`、`system`、`task`、`success_count`、`joules` を検証して `data/raw/energy_measurements.jsonl` へ追記し、そのまま energy measurement readiness report を再生成できるようにした。平均電力しか得られない測定器や `powermetrics` 由来の観測にも対応し、`average_watts * duration_seconds` から `joules` を導出して記録できる。  
  * real joule evidence の受理条件を task-paired に強化し、各 `task` に SARA/ANN の両方の測定が存在し、最小 per-task `ann_to_sara_joule_efficiency_ratio` が閾値を満たす場合だけ ANN efficiency roadmap Stage 6 を通すようにした。  
  * energy measurement readiness report に `measurement_plan` を追加し、未測定時の canonical task pair、片側だけ測定済みの missing system、閾値未満の weak pair、次に実行すべき `record-energy-measurement` command template を機械可読で提示できるようにした。  
  * energy measurement readiness report に `measurement_session_plan` を追加し、pending/weak pair を stable run-id template 付きの `real_energy_session` command へ展開して、実験者が同一タスク・同一採点条件の paired SARA/ANN 測定をそのまま記録できるようにした。  
  * `energy_measurement_readiness.py` が `workspace/evaluation/energy_measurement_session_plan.json` と `.txt` を単独成果物として保存するようにし、実験者が readiness report 全体を読まずに次の測定セッションだけ確認できる導線を追加した。  
  * research product completion gate に `energy_measurement_session_plan` チェックを追加し、独立した測定セッション成果物の schema / status / paired systems / run-id template / `real_energy_session` command が壊れた場合は研究プロダクト完成扱いにしないようにした。  
  * `v1_release_gate.py` に `research_product_completion` チェックを追加し、v1.1最終判定が research product completion report（ANN効率ロードマップと energy measurement session plan を含む）を通っていない場合は release promotion を止めるようにした。  
  * `ann_efficiency_roadmap_gate.py` に `next_evidence_actions` を追加し、energy measurement `measurement_plan` の pending/weak pair をロードマップ最上位と summary に伝播して、Stage 6 が protocol-ready PASS でも次の real-joule evidence loop を見失わないようにした。  
  * `ann_efficiency_roadmap_gate.py` の `next_evidence_actions` が `measurement_session_plan.planned_runs` を優先するようにし、ANN効率ロードマップの次証拠アクションを具体的な paired measurement session command として扱えるようにした。  
  * `operational_readiness` が `ann_efficiency_roadmap_gate.json` を読み込み、実ジュール測定の `next_evidence_actions` を runbook action manifest へ伝播するようにして、研究ロードマップ上の次証拠取得を運用アクションとして扱えるようにした。  
  * DiffusionBlocks 論文の示唆を SARA 方針に合わせて取り込み、`sparse_diffusion_block_readiness.py` を追加した。勾配・GPU・dense matrix runtime へ寄せず、uncertainty range の等確率分割、独立 sparse-event block、局所 denoising、block count ablation、recurrent-depth single-pass 化、policy compatibility を `SparseDiffusionBlockReadiness` として監査する。  
  * `scripts/sara_cli.py eval-sparse-diffusion-block-readiness` と research product completion gate へ同ゲートを接続し、局所可塑性/予測符号化が「なぜブロック独立にスケールできるのか」を v1.1 後の研究プロダクト証拠として追跡できるようにした。  
  * sparse diffusion block を real-data external validity / ladder / ANN efficiency roadmap gate の Stage 2/3 に接続し、実コーパス由来タスクでも denoise accuracy、event-cost advantage、partition integrity、single-pass integrity が崩れた場合は次段階へ昇格しないようにした。  
  * `ann_efficiency_roadmap_gate.py` の Stage 2/3 に negative control を接続し、通常QAで高精度でも no-hit abstention が崩れた場合は ANN効率ロードマップを通さないようにした。  
  * world-model の空間理解に向けて、`future_state_consistency_benchmark` に lightweight room-geometry case を追加した。2D観測イベント（visible wall / door opening / occluded boundary hint / camera pose）から top-down room hypothesis を復元し、`future_state_spatial_projection_integrity`、`future_state_spatial_topology_consistency`、`future_state_spatial_occlusion_reasoning` を Phase3 predictive readiness / Stage B observed metrics に流す。  
  * この空間ベンチは 3DGS やGPU再構成を中核に入れず、SARAの sparse event / constraint reasoning で「見えていない壁を仮説化し、閉じた平面構造へ投影する」能力を育てるための足場として扱う。  
  * room-geometry case を counterfactual spatial hypothesis selection へ拡張し、`observed_occlusion` / `missing_south_wall` / `mirrored_depth` の複数仮説を生成して、projection / topology / area / occlusion consistency と event-cost proxy で最良案を選択するようにした。`future_state_spatial_counterfactual_selection` を追加し、単一の平面図決め打ちではなく、複数候補から制約で選ぶ世界理解へ一段進めた。  
  * 空間ベンチを connected two-room topology case へ拡張し、`entry -> kitchen` のドア接続、隠れた部屋の奥行き、非重なり制約、総面積を sparse event から推定するようにした。`overlap_room` / `disconnected_room` の反事実候補を落とし、`future_state_spatial_adjacency_consistency`、`future_state_spatial_door_connectivity_integrity`、`future_state_spatial_multi_room_counterfactual_selection` を Phase3 predictive readiness / Stage B observed metrics に流す。  
  * connected topology を action selection に接続し、`door_route` / `wall_crossing` / `stay_put` の候補から、ドア affordance・進捗・衝突リスク・event-cost proxy に基づいて低コストな有効ルートを選ぶようにした。`future_state_spatial_route_planning_integrity`、`future_state_spatial_affordance_action_selection`、`future_state_spatial_energy_aware_route_selection` を追加し、空間理解が planning に使われているかを観測する。  
  * route planning を sparse state execution へ接続し、選択済み `door_route` は `entry -> kitchen` の状態更新として受理し、無効な `wall_crossing` は room state を壊さず rollback observable として記録するようにした。`future_state_spatial_route_state_update_integrity`、`future_state_spatial_invalid_action_rejection`、`future_state_spatial_route_rollback_observability`、`future_state_spatial_route_execution_cost_bound` を追加し、空間 world-model が行動実行後の状態管理にも使われるかを観測する。  
* **完了条件:**  
  * 小規模・中規模・大規模の3段階で同一導線が再現可能で、Phase3/4/5 と operational readiness を段階的に通過できる。  
  * `performance_energy_ratio_proxy` / `ann_cost_advantage_proxy` の悪化を検知できる運用が継続可能である。  
  * 実データ由来の retrieval / extractive summary / continual memory smoke tasks で、ANN風 dense proxy と同等の正答率を維持しつつ、event-cost あたり性能で優位性を示せる。  

### **Phase 5: Spiking H-JEPA & Advanced Predictive Coding (中期〜後期)**

* **目標:** 自己教師あり学習による抽象的な潜在空間表現の獲得と、誤差逆伝播に頼らない高次推論（Hierarchical Joint Embedding Predictive Architecture）のSNN上での実現。  
* **背景と意義:** 従来の生成モデルが陥りがちな「ピクセルレベルの厳密な再構成」を避け、SDR（Sparse Distributed Representation: スパース分散表現）を用いた意味的・抽象的な潜在空間での未来予測世界モデルを確立します。  
* **Spiking H-JEPA 進化のステップ:**  
  * **Step 1: 基礎的JEPAモジュールの確立**  
    * オンラインネットワークとターゲットネットワーク間の予測誤差を、STDPによるスパイクタイミング差として定式化し、自律更新する基本アーキテクチャを安定化する。  
    * メタ可塑性（BCM則）や恒常性機構を組み合わせ、ノイズ環境下でも発散しない概念学習を評価する。  
  * **Step 2: 階層化と抽象化**  
    * 上位層からのトップダウン予測と下位層からのボトムアップ観測が局所相互作用する、階層的予測符号化を実装する。  
    * 受容野の拡張に伴い、より抽象度の高い概念をスパイク群として表現できるようにする。  
  * **Step 3: 時空間予測ストリームへの拡張**  
    * 動画・音声など連続ストリームに対応し、未来潜在状態をスパイク到達タイミングと発火頻度で表現する。  
    * SpatioTemporal STDP を用いた自己回帰的推論と統合する。  
  * **Step 4: マルチモーダル統合と能動的推論**  
    * 視覚、聴覚、言語などを共通スパイク潜在空間へ統合し、異種モダリティ間の連想推論を強化する。  
    * 行動計画が未来予測に与える影響を組み込む Active Inference を実装する。
* **SNN高次推論・マルチモーダル統合資料から採用する方針（限定採用）:**  
  * SpikeMLLM / CMSF / Spiking-WM を大規模モデルとして丸ごと移植するのではなく、SARA の CPU-first / no backprop / event-driven 制約に合う軽量 primitive と benchmark に分解して取り込む。  
  * `common_spike_space` を定義し、text / structured state / future image-audio hooks を同じ sparse event schema へ正規化できるようにする。最初はテキストと構造化状態を対象にし、画像・音声は adapter 仕様と fixture だけを先に用意する。  
  * TC-LIF の思想は `temporal_compression` として採用し、スパイク展開ステップを固定で増やさず、低ビット状態・短い event window・必要時のみの追加stepで表現する。  
  * MSTS の思想は `modality_temporal_scale` として採用し、入力種別・信頼度・world-model surprise に応じて計算予算を配分する。全モダリティを同じ時間幅で処理しない。  
  * Multi-Compartment Neuron はフル微分方程式モデルとしてではなく、`dendritic_context_gate` / `multi_channel_state` として軽量導入し、短期文脈・長期文脈・予測誤差を別チャネルで保持する。  
  * Spiking H-JEPA は生データ再構成ではなく、`latent transition -> prediction error -> correction event` を評価する lightweight world-model benchmark として先行実装する。  
  * 高次推論は hypergraph / CoT / large MLLM RL をそのまま導入せず、`event_relation_trace`、`causal_candidate_trace`、`reverse_reasoning_trace` として説明可能な疎イベント列から始める。  
* **Long-horizon training研究（arXiv:2605.02572v1）から採用する方針（限定採用）:**  
  * 長期タスクの不安定化は「難問化」だけでなく `horizon length` 自体が独立ボトルネックである前提を採用し、Phase 5 の world-model / planner 評価を horizon 軸で分離して監査する。  
  * `horizon_reduction`（macro action / subgoal decomposition）を、dense planner 置換ではなく event-driven な action abstraction として導入する。  
  * 学習・推論の評価は平均スコアだけでなく `short -> medium -> long` の horizon 別成功率と安定性を必須表示し、長期化で崩れる経路を gate で直接検知する。  
  * 採用しない要素: GPU前提の大規模RL最適化、backprop依存で長期credit assignmentを解く実装を runtime 中核へ入れること。  
* **Generative Manifold Networks研究（bioRxiv 2026.05.12.724527）から採用する方針（限定採用）:**  
  * "Explainable prediction and simulation of complex system dynamics through networks of manifolds" は、複雑系の時系列を低次元多様体と因果リンクのネットワークとして扱い、局所近傍から次状態や行動を予測する発想を Phase 5 の world-model / continual memory に取り込む。  
  * SARA Engine では PCA / dense matrix / GPU 前提の実装をそのまま採用せず、`local_manifold_transition_memory` として、sparse event state、近傍軌道、次状態候補、correction event、counterfactual branch を bounded graph に保存する。  
  * 継続学習では、重みを頻繁に再学習する代わりに、経験軌道を局所多様体メモリへ追加し、予測時に上位近傍だけを参照する。これにより catastrophic forgetting を避けつつ、event-cost bounded な経験利用を目指す。  
  * Phase 5 の Spiking H-JEPA trace と接続し、`latent transition -> nearest trajectory support -> prediction error -> correction event` の流れを説明可能にする。micro-ES は方策微調整、manifold memory は状態遷移・経験構造の保持として役割分担する。  
  * 初期評価候補は `manifold_transition_locality`、`manifold_rollout_stability`、`causal_route_sparsity`、`withheld_trajectory_recall` とし、まず observed-only で Phase 5 benchmark / summary に追加する。  
  * 採用しない要素: dense embedding の常時全探索、matrix-heavy dimensionality reduction を runtime 必須にすること、査読前 preprint の性能主張を release gate 必須条件として即時固定すること。  
* **δ-mem研究（arXiv:2605.12357）から採用する方針（限定採用）:**  
  * `δ-mem: Efficient Online Memory for Large Language Models` は、長い履歴をそのまま context に戻すのではなく、固定サイズの online associative memory state に圧縮し、delta-rule による残差更新で継続的に記憶を保つ発想を採用候補とする。  
  * SARA Engine では Transformer attention への low-rank correction をそのまま導入せず、`DeltaAssociativeSpikeMemory`（仮称）として sparse event key / value / residual trace を小さな bounded state に保持する。  
  * 更新則は `prediction -> residual -> gated delta write -> retention/forget gate` の形にし、backpropagation ではなく local plasticity / STDP / homeostatic retention と整合させる。  
  * **Gated DeltaNet-2（arXiv:2605.22791）からの追加採用候補:** 線形 attention 実装そのものは採用しないが、compressed memory を壊さず編集するために、旧内容を消す `erase gate` と新情報を書き込む `write gate` を分離する発想は δ-mem の次段設計として採用候補にする。SARA では channel-wise dense gate ではなく、event / phase / astro stability ごとの bounded scalar gate として扱い、`erase` と `commit` を別々に observed-only 監査する。  
  * Gated DeltaNet-2 の long-context / multi-key retrieval 評価姿勢は、SARA の `multi-history stress`、`manifold guard`、`candidate miss guard` を強化する参考にする。ただし backward pass、chunkwise WY algorithm、大規模 LM training、dense linear attention は本プロジェクトの runtime 方針とは切り離す。  
  * state は dense matrix を必須にせず、初期実装では `8x8` 相当の小さな profile を参考にしつつ、疎 event row、低精度 multilevel weight、active row storage で表現する。  
  * 既存の `NestedContinualMemoryController` では、session/direct/hippocampus/LTM の間に「短期履歴を圧縮して steering signal を出す online state」を挟み、テキスト再投入ではなく memory controller / world model / planner への event-level bias として使う。  
  * `LocalManifoldTransitionMemory` とは役割を分ける。manifold memory は経験軌道・因果近傍の保持、δ-style online memory は直近履歴の圧縮状態と残差補正を担当する。  
  * 初期評価候補は `delta_memory_residual_write_integrity`、`delta_memory_retention_gate_stability`、`delta_memory_context_recall_without_text_reinjection`、`delta_memory_state_budget_integrity`、`delta_memory_interference_guard` とし、まず observed-only で Stage D / Stage E / Phase 5 summary に追加する。  
  * **現在の進捗:** `DeltaAssociativeSpikeMemory` を lightweight primitive として追加し、sparse `context event -> residual event` の bounded state、residual-only write、retention gate、state budget eviction、text reinjection なしの readout を単体テストで検証できるようにした。現段階では本体 runtime / release gate には接続せず、次に `continual_consolidation_benchmark` と `cognitive_runtime_benchmark` へ observed-only で接続する。  
  * 採用しない要素: frozen Transformer attention への dense low-rank correction を標準 runtime にすること、LLM hidden state projection を前提にすること、GPU/attention 前提の実装や性能主張を release gate 必須条件として即時固定すること。  
* **Linear RNN + SNN fusion提案から採用する方針（限定採用 / Stage E-F bridge）:**  
  * 基本方針として、「時間スケールを持つ連続状態」と「誤差時だけ発火する sparse event」を統合する発想は SARA の CPU-first / no backprop / no dense matrix / neuromorphic-ready 方針と強く整合する。Transformer を直接置換する断定目標ではなく、bounded state・event cost・continual adaptation の優位性を検証する研究仮説として採用する。  
  * **Idea 1: multi-timescale membrane state.** 線形RNN / SSM の時定数を SNN の膜電位 leak rate として扱い、short / mid / long の複数リーク群を持つ `multi_timescale_leak_state` を Stage E runtime に追加候補とする。長文文脈は KV cache ではなく、低速リーク群の bounded membrane state として保持する。  
  * **Idea 2: predictive-error-gated spiking.** 線形RNNが次状態を予測し、実入力との差分が閾値を超えた時だけ SNN correction spike を発火する。既存の Spiking H-JEPA / prediction error / correction event と接続し、予測通りの入力では event cost が下がることを `predictive_spike_entropy_reduction` として観測する。  
  * **Idea 3: phase-synchronized binding.** 離れた token / state / action を dense attention で結ばず、同じ phase slot / spike timing window に入った event を結合候補として扱う `phase_synchronized_binding_trace` を検討する。実装は oscillator の完全物理モデルではなく、bounded phase bucket と coincidence check から始める。  
  * **Idea 4: forward-only local learning.** BPTT / backpropagation through time は採用せず、STDP、homeostatic retention、delta residual write、軽量 forward-only credit trace を組み合わせる。RTRL や Forward-Forward は大規模 dense 勾配推定としてではなく、局所 eligibility trace / goodness proxy として限定的に再解釈する。  
  * 初期トイモデル候補は `tau_leak_state_update`: `tau * dh/dt = -h + sparse_recurrent_event + input_event` を離散時間化し、neuron group ごとに `tau` / leak / threshold を変える。実装は Python fixture で始め、行列積ではなく sparse event list と small scalar state update を使う。  
  * 初期評価候補は `multi_timescale_leak_retention_observed`、`predictive_spike_entropy_reduction_observed`、`phase_binding_coincidence_integrity_observed`、`forward_only_local_update_stability_observed`、`timescale_state_budget_integrity_observed` とする。まず observed-only で Phase 5 / Stage E summary に流し、release gate 必須条件にはしない。  
  * Stage F では、これらを `spike_event_ir` の `delay` / `routing_hint` / `online_update_policy` / `state_budget_units` と接続し、Lava / SpiNNaker / Akida profile report が multi-timescale state や phase bucket を壊さないかを compatibility report で確認する。  
  * 採用しない要素: Transformer 超えを性能主張として即時固定すること、KV cache の完全代替を未検証のまま release gate 条件にすること、dense SSM / Mamba 実装や GPU 前提の scan kernel を中核にすること、BPTT / dense RTRL を学習経路として導入すること、物理 oscillator を過剰に精密シミュレーションして CPU-first 制約を壊すこと。  
* **Linear RNN + SNN fusion Roadmap Action Items:**  
  * DONE: `MultiTimescaleLeakState` の toy primitive を追加し、short / mid / long leak groups が bounded sparse state で異なる保持時間を示すかを単体テストできるようにした。  
  * DONE: `cognitive_runtime_benchmark` に predictive-error-gated spike case を追加し、予測通りの入力では correction spike が 0、surprise 入力では residual correction event が出ることを `predictive_spike_entropy_reduction_observed` で確認できるようにした。  
  * DONE: `phase_synchronized_binding_trace` の fixture を追加し、離れた sparse events が同一 phase bucket で結合され、無関係 event が誤結合されないことを `phase_binding_coincidence_integrity_observed` で監査できるようにした。  
  * DONE: forward-only local update の最小 eligibility trace を bounded sparse state として追加し、BPTT なしで短期適応が発散しないことを `forward_only_local_update_stability_observed` で benchmark 化した。  
  * DONE: 上記 observed metrics を Phase 3 summary / release soak / operational readiness に表示しつつ、`gate_metrics` / `observed_metrics` と `excluded_from_score_and_release_gate` policy で `overall_score` と release gate から分離した。  
  * DONE: multi-timescale / predictive-error / phase-binding / forward-only trace を Stage F `spike_event_ir` の optional `state_trace` event として `delay` / `routing_hint` / `online_update_policy` / `state_budget_units` へ接続し、neuromorphic profile compatibility report と strict edge runtime で壊れないことを確認できるようにした。  
  * DONE: Linear RNN + SNN fusion observed metrics の専用履歴比較 `linear_snn_fusion_observed_trend` を Phase 3 report / release soak / operational readiness に追加し、retention / correction / binding / local update の退行を release gate 非ブロッキングで監査できるようにした。  
  * DONE: 実データ・長時間runを `linear_snn_fusion_observed_trend` 付きで実行するための `observed_trend_long_run_validation` runbook action を追加し、synthetic fixture 以外の retention / correction / binding / local update 確認を operational queue に載せられるようにした。  
* **Phase 5 acceptance criteria への追加候補:**  
  * `common_spike_space_integrity`: 異なる入力源が同じ sparse event schema へ落ち、routing / retrieval / world-model が共通に扱えること。  
  * `temporal_compression_efficiency`: 予測品質を落とさず event window / spike step / energy proxy を削減できること。  
  * `modality_temporal_budget_integrity`: 入力種別ごとの時間予算が bounded で、過剰stepが release summary に可視化されること。  
  * `dendritic_context_gate_stability`: multi-channel context が過去文脈を保持しつつ、現在入力への過剰干渉を起こさないこと。  
  * `spiking_hjepa_latent_transition`: 潜在状態遷移・予測誤差・補正イベントが operator trace と整合すること。  
  * `reverse_reasoning_trace_integrity`: 結果から原因候補へ遡る trace が、counterfactual branch と矛盾しないこと。
  * `horizon_bucket_stability`: 同一難易度条件で horizon を伸ばしても成功率劣化が許容幅内に収まること。  
  * `macro_action_effectiveness`: action abstraction により step 数と event cost を削減しつつ quality を維持できること。  
  * `subgoal_decomposition_integrity`: subgoal 連鎖が planner/world-model trace と整合し、最終失敗時に崩壊位置を機械可読に特定できること。  
  * `manifold_transition_locality`: sparse event state から近傍経験軌道を bounded cost で検索し、次状態候補の局所支持を説明できること。  
  * `manifold_rollout_stability`: 多段 rollout で局所多様体メモリが発散せず、prediction error と correction event が収束方向へ働くこと。  
  * `causal_route_sparsity`: 次状態予測に使った因果 route が少数の event edge に限定され、dense 全探索へ退化しないこと。  
  * `withheld_trajectory_recall`: 学習に使っていない保持軌道に対して、近傍軌道から妥当な予測・補正候補を再構成できること。  
  * `manifold_scan_budget_integrity`: 局所多様体検索が bounded scan budget 内で完結し、capacity 増加時にも dense 全探索を前提にしないこと。  
  * `manifold_indexed_candidate_integrity`: sparse event id から候補軌道を絞り、先頭N件スキャンではなく局所候補集合から近傍を選択できること。  
  * `manifold_index_scan_reduction`: indexed candidate selection により dense scan baseline より少ない軌道参照で recall / case coverage を維持できること。  
  * `manifold_candidate_miss_guard`: sparse event index に一致しない未知クエリを既存軌道へ fallback せず、未観測入力に対して誤った次状態候補を出さないこと。  
  * `manifold_capacity_pressure_recall`: distractor 軌道が増えた容量圧力下でも、event-indexed candidate selection により重要軌道の recall を維持できること。  
  * `manifold_replay_refresh_retention`: bounded capacity の局所多様体メモリで、重要軌道を replay により新鮮化し、後続 distractor 追加後も recall を維持できること。  
  * `manifold_replay_refresh_eviction_integrity`: replay で新鮮化された重要軌道を保持しつつ、古い distractor 軌道が容量制限により自然に退避されること。  
  * `delta_memory_residual_write_integrity`: 予測済みの内容を過剰更新せず、prediction error / residual がある場合だけ bounded delta write が発火すること。  
  * `delta_memory_retention_gate_stability`: retention/forget gate が短期履歴を保持しつつ、新規入力による上書きや過剰固定化を抑えること。  
  * `delta_memory_context_recall_without_text_reinjection`: 明示的な長文履歴を prompt/context に戻さず、online state だけで関連履歴の steering signal を再現できること。  
  * `delta_memory_state_budget_integrity`: online associative state が固定サイズ・低精度・active row budget 内に収まり、履歴長に比例して肥大化しないこと。  
  * `delta_memory_interference_guard`: 直近履歴の圧縮更新が LTM / hippocampus / manifold trajectory recall を破壊しないこと。  
  * `multi_timescale_leak_retention_observed`: short / mid / long leak groups が異なる保持時間を示し、長期文脈を bounded scalar state で保持できること。  
  * `predictive_spike_entropy_reduction_observed`: 予測済み入力では spike/event cost が下がり、surprise 入力では correction event が発火すること。  
  * `phase_binding_coincidence_integrity_observed`: 離れた sparse events が同一 phase bucket で結合され、無関係 event の誤結合が抑えられること。  
  * `forward_only_local_update_stability_observed`: BPTT なしの局所 eligibility / STDP / delta update が短期適応を改善しつつ発散しないこと。  
  * `timescale_state_budget_integrity_observed`: multi-timescale state が固定 budget 内に収まり、context length に比例して肥大化しないこと。  
* **Nested Learning から採用する方針（限定採用 / Phase 5 late -> Phase 6 bridge）:**  
  * arXiv:2512.24695 "Nested Learning" は、モデルを単一学習器ではなく、更新頻度・context flow・記憶圧縮の異なる複数の学習問題として扱う点を採用する。  
  * SARA Engine では gradient optimizer や大規模LM前提の self-modifying module を直接導入せず、`Nested Continual Memory Controller` として、短期適応・海馬転送・LTM固定化・構造可塑性を multi-rate に調停する軽量制御層へ落とし込む。  
  * continuum memory の考え方は、`session memory -> direct memory -> hippocampus -> LTM -> replay consolidation -> structural plasticity` を離散的な箱ではなく、安定度・更新頻度・エネルギー予算が連続的に変わる記憶スペクトラムとして再定義する。  
  * self-modifying learning は重みをブラックボックスに自己改変する仕組みではなく、STDP / replay / homeostasis / sparse retrieval / world-model refinement のどれを、どの頻度・どの予算で動かすかを選ぶ `nested_update_scheduler` として実装する。  
  * 初期評価は `nested_memory_readiness` として、`multi_rate_update_integrity`、`continuum_memory_transfer_stability`、`scheduler_energy_budget_integrity`、`catastrophic_interference_guard` を観測し、Phase 5 completion 後に Stage D/E/Phase6 gate へ段階接続する。  
  * 採用しない要素: backpropagation / gradient optimizer を runtime 中核に置くこと、自己改変を監査不能な重み更新として許可すること、GPU前提の大規模 continual training を release gate 必須要件にすること。  
* **現在の進捗:**  
  * `NestedContinualMemoryController` を追加し、session / direct / hippocampus / LTM / structural の記憶スペクトラムを multi-rate update・transfer threshold・energy budget・interference guard で調停できる軽量 controller として実装した。  
  * `nested_memory_readiness_benchmark.py` を追加し、`multi_rate_update_integrity`、`continuum_memory_transfer_stability`、`scheduler_energy_budget_integrity`、`catastrophic_interference_guard` を CPU-only managed artifact として検証可能にした。  
  * `phase3_accuracy_suite` に `nested_memory` component と `nested_memory_readiness` focus を observed-only で追加し、Phase 3 completion / release gate を急に硬くせず、Nested Learning 由来の memory scheduler 品質を summary / trend で継続観測できるようにした。  
  * `CommonSpikeSpaceEncoder`、`TemporalCompressionPolicy`、`ModalityTemporalBudget`、`DendriticContextGate` を追加し、text / structured state / lightweight adapter features を共通 sparse event schema へ正規化できる軽量 primitive を実装した。  
  * `NeuralAssemblyTracker` を仮インターフェースから実装へ引き上げ、sliding window 内の sparse co-activation、pairwise intersection による noisy core 抽出、bounded candidate map、support threshold を持つ省エネな概念アセンブリ追跡として利用可能にした。  
  * `CorticoHippocampalSystem` が利用する `SpatioTemporalSNN` を空実装から sparse synapse / local receptive field / recurrent ctx / STDP / homeostatic clipping / activity report を備える CPU-only 実装へ引き上げ、海馬リプレイと記憶固定化が実際にイベント駆動更新を持つようにした。  
  * 旧実験系の `snn_models.modular_snn` も sparse LIF layer / sparse connection propagation / local STDP を備える実装へ更新し、構造可塑性のプロトタイプが空メソッドに依存しない状態にした。  
  * `cognitive_runtime_benchmark` を追加し、`common_spike_space_integrity`、`temporal_compression_efficiency`、`modality_temporal_budget_integrity`、`dendritic_context_gate_stability`、`spiking_hjepa_latent_transition`、`reverse_reasoning_trace_integrity` を CPU-only で観測可能にした。  
  * `build_spiking_hjepa_transition_trace` と `compare_spiking_hjepa_transition_branches` を追加し、latent transition / prediction error / correction event / anti-collapse diversity / counterfactual separation を共通 sparse event 上で監査できるようにした。  
  * `build_spiking_hjepa_multistep_trace` を追加し、複数ステップの latent transition chain、prediction error の累積、correction convergence を共通 sparse event 上で監査できるようにした。  
  * `phase5_predictive_coding_benchmark.py` と `phase5_entry_gate.py` を追加し、Phase 5 entry criteria（`latent_transition_alignment`、`prediction_error_observability`、`correction_event_coverage`、`anti_collapse_event_diversity`、`counterfactual_transition_separation`、`multi_step_latent_chain_integrity`、`long_horizon_error_correction_convergence`）を managed artifact として検証可能にした。  
  * `phase5_entry_gate` に `phase5_entry_gate_report.json` / `phase5_entry_gate_summary.txt` の出力を追加し、entry gate 自体の `failed_checks` / metric checks / threshold checks を運用から直接参照できるようにした。  
  * `phase5_completion_gate.py` を追加し、Phase 4 completion 前提・Phase 5 benchmark・Phase 5 entry gate の3点を統合した completion gate を機械判定できるようにした。`phase5_completion_gate_report.json` / `phase5_completion_gate_summary.txt` を managed artifact として出力し、primary trace alignment・multi-step correction coverage・counterfactual separability を completion レベルで監査可能にした。  
  * `operational_readiness --refresh-artifacts` の更新フローに `phase5_completion_gate.py` を接続し、strict operational refresh でも Phase 5 completion artifact を自動再生成・検証できるようにした。  
  * `v1_release_gate` に `phase5_completion_quality` チェックを追加し、Phase 5 completion gate artifact が PASS でなければ v1.0 判定を通さないようにした。  
  * `phase3_accuracy_suite` に `--regression-tolerance` を追加し、`operational_readiness --phase3-regression-tolerance` 経由で refresh 時のトレンド回帰検知感度を調整できるようにした。連続実行時の微小揺れで strict pipeline が不安定化するケースを運用側で制御可能にした。  
  * `operational_readiness --refresh-artifacts` に Phase 5 predictive-coding benchmark と Phase 5 entry gate を組み込み、strict operational refresh だけで Phase 5 standalone artifacts も最新化されるようにした。  
  * `operational_readiness` の検証対象に `phase5_entry_gate_report.json` を追加し、Phase 5 entry gate artifact が失敗・欠損している場合は operational readiness 自体が FAIL になるようにした。  
  * `cognitive_runtime_benchmark` の `spiking_hjepa_latent_transition` を強化し、非空フィールド確認ではなく、実際の sparse latent-transition trace と counterfactual branch separation を Stage E 側でも確認するようにした。  
  * `phase3_accuracy_suite` に `cognitive_runtime` component と `cognitive_runtime_readiness` focus を統合し、Stage E / Phase 5 由来の高次推論・マルチモーダル統合 readiness を summary / trend で追跡できるようにした。  
  * `phase3_accuracy_suite` に `phase5_predictive_coding` component と `phase5_entry_readiness` focus を統合し、Phase 5 entry の latent-transition / multi-step correction 指標を Phase 3 completion / summary / trend に流せるようにした。  
  * `release_soak` の埋め込み accuracy に Stage C/D/E と Phase 3 completion を保持するようにし、release gate が Phase 5 追加後も正しく Stage C-E readiness を検証できるようにした。  
  * `release_soak` / `operational_readiness` の summary に `phase5_entry_passed`、Phase 5 latent-transition 指標、multi-step chain / long-horizon correction 指標を追加し、最上位の運用レポートから Spiking H-JEPA entry readiness を直接確認できるようにした。  
  * `phase5_contract` を追加し、Phase 5 entry の必須メトリクスを benchmark / entry gate / release gate で共通参照するようにした。  
  * `horizon_bucket_stability` を `phase5_contract` の required metric へ昇格し、phase5 benchmark / entry gate / release gate / v1 gate / operational snapshot の全判定経路で必須チェックとして統一した。  
  * `macro_action_effectiveness` と `subgoal_decomposition_integrity` も `phase5_contract` の required metric へ昇格し、Phase5 entry gate / release gate / v1 gate / operational readiness の失敗推定・復旧アクションまで含めて同一契約で検証できるようにした。  
  * `phase5_completion_gate` を拡張し、新KPIの名目値チェックだけでなく `macro_step_reduction` / `macro_cost_reduction` / `subgoal_coverage_ratio` / `micro_es_low_rank_trace_complete` / `micro_es_fitness_improvement` / `micro_es_event_cost_reduction` / `micro_es_population_event_budget` の detail 指標でも completion 条件を厳格検証できるようにした。これにより、Phase5 completion の偽陽性（メトリクスだけPASSだが行動抽象化や探索改善の実体が不足）を release 前に検出できる。  
  * `operational_readiness` と `v1_release_gate` の Phase5 completion artifact 検証を強化し、`metric.*` / `threshold.*`（Phase5 contract準拠）に加えて `macro_step_reduction` / `macro_cost_reduction` / `subgoal_coverage_ratio` の required check 欠落・未達を最終ゲートで検出できるようにした。  
  * `release_gate` の error-details/recovery 解析を拡張し、`Phase 5 completion gate check map is missing required checks` を `missing_required_checks` として構造化しつつ、`phase5.completion_gate` / `phase5.completion_required_checks` の inferred check と completion gate 再実行アクションへ自動接続できるようにした。  
  * `v1_release_gate` の `phase5_completion_quality.details` に `missing_required_checks` / `failed_required_checks` を追加し、Phase5 completion artifact が FAIL のときに「欠落」か「未達」かを v1 最終レポートだけで即判別できるようにした。  
  * `v1_release_gate` summary 出力にも `phase5_completion_missing_required_count` / `phase5_completion_failed_required_count` と対象チェック名を表示し、CIログ上で Phase5 completion failure の原因切り分けを即時に行えるようにした。  
  * Attention Residuals（arXiv:2603.15031）の中核アイデアを SARA 向けに軽量採用し、Phase5 benchmark に `depth_selective_routing_integrity`（depth-wise selective routing の健全性）を observed KPI として追加した。`depth_route_trace` / `depth_route_avg_selected_ratio` / `depth_route_max_weight_sum_deviation` を保存し、深さ方向の選択集約を dense 実装なしで監査できるようにした。  
  * `depth_selective_routing_integrity` を Phase5 contract の required metric へ昇格し、Phase5 entry/completion/release/operational/v1 の全ゲートで必須検証へ統一した。`release_soak` / `operational_readiness` / `phase3_accuracy_summary` でも同KPIを明示表示し、Phase5完了判定の監査軸を一貫化した。  
  * Evolution Strategies at the Hyperscale / EGGROLL の中核アイデアは、GPU巨大populationや大規模行列更新ではなく、SARA制約に合う `energy_aware_micro_es_low_rank_rank1` として限定採用した。Phase5 benchmark に `micro_es_policy_refinement_integrity` を追加し、予測誤差補正・rollback削減・event-cost削減を少数候補の rank-1 policy perturbation で改善できることを監査する。  
  * `micro_es_policy_refinement_integrity` を Phase5 contract の required metric へ昇格し、Phase5 entry/completion/release/operational/v1 の全ゲートで必須検証へ統一した。さらに completion gate では低ランク trace 完備、fitness 改善、event-cost 削減、population event budget を個別 required check として監査する。これにより、backpropなしの探索的改善を採用しつつ、ANN風の大規模dense trainingへ逸脱しないことを gate で確認できる。  
  * `local_manifold_memory` を追加し、Generative Manifold Networks 由来の局所多様体メモリを sparse event trajectory probe として observed-only 実装した。Phase5 benchmark では `manifold_transition_locality`、`manifold_rollout_stability`、`causal_route_sparsity`、`withheld_trajectory_recall`、`manifold_trajectory_case_coverage`、`manifold_average_case_recall`、`manifold_scan_budget_integrity`、`manifold_indexed_candidate_integrity`、`manifold_index_scan_reduction`、`manifold_candidate_miss_guard` を記録し、release / handoff / risk counterfactual の複数 synthetic trajectory case で近傍軌道選択、event-indexed candidate selection、dense scan baseline からの探索削減、scan budget、sparse route、未知クエリの誤 fallback 抑止を監査できる。さらに `LocalManifoldTransitionMemory` を bounded in-memory API として追加し、継続的に経験軌道を追加しながら容量制限・近傍予測・case coverage を同じロジックで検証できるようにした。継続学習 benchmark では distractor-heavy な `manifold_capacity_pressure_recall` に加えて、重要軌道を replay で新鮮化して容量制限下に残す `manifold_replay_refresh_retention` / `manifold_replay_refresh_eviction_integrity` も observed-only で追跡し、容量圧力下の recall 維持、scan reduction、重要経験の保持、古い distractor の自然退避を確認する。これらの manifold 指標はレポート・trend・operational snapshot へ表示するが、査読前研究の限定採用として `overall_score` と release gate の必須判定からは分離する。  
  * `cognitive_runtime_benchmark` の Spiking H-JEPA primary / counterfactual trace に `LocalManifoldTransitionMemory` を接続し、`cognitive_manifold_trace_support_observed`、`cognitive_manifold_trace_recall_observed`、`cognitive_manifold_trace_scan_budget_observed`、`cognitive_manifold_trace_index_scan_reduction_observed`、`cognitive_manifold_trace_candidate_guard_observed` を Phase3 / release soak / operational readiness summary へ表示できるようにした。これにより Stage E の modular runtime でも、latent transition がどの局所経験軌道に支えられ、primary と counterfactual が分離され、候補外軌道を密探索せずに削減できているかを観測できる。現段階では observed-only とし、Stage E minimum gate は既存契約のまま維持する。  
  * `release_gate` の必須 component / focus に `phase5_predictive_coding` / `phase5_entry_readiness` を追加し、Phase 5 entry 欠落・劣化時に明示エラーと回復アクションを出せるようにした。  
  * `v1_release_gate` に Phase 5 entry の明示チェックを追加し、Phase 3 report の latent-transition 指標と `operational_readiness_report` に伝播した operational snapshot の両方が PASS でなければ v1.0 出荷不可にした。  
  * Stage E minimum check の定義を `stage_e_contract` として共通化し、phase3 suite / release gate / test fixture が同じ契約を参照できるようにした。  
  * `stage_e_readiness` に `minimum_failure_count` / `minimum_failures` / `minimum_checks` を持たせ、Phase 3 completion と release gate の必須条件へ昇格した。  
  * release gate の recovery planner に `cognitive_runtime_benchmark.py` 再実行アクションを追加し、Stage E failure 時に回復手順を機械可読に提示できるようにした。  
  * `ModularCognitiveRuntime` を追加し、encoder / memory controller / world model / planner / actor を共通 sparse event で接続する lightweight orchestration prototype を実装した。  
  * `module_orchestration_integrity`、`counterfactual_lane_integrity`、`action_trace_observability` を `cognitive_runtime_benchmark` と `phase3_accuracy_suite` に追加し、Stage E の modular runtime / counterfactual lane / action trace を継続観測できるようにした。  
  * `build_causal_candidate_trace` を追加し、forward relation trace と reverse reasoning trace を同一 branch 上で接続して、選択行動の原因候補説明が欠落しないことを監査できるようにした。  
  * `causal_candidate_trace_integrity` を Stage E minimum gate に追加し、`cognitive_runtime_benchmark` / Phase 3 summary / release soak / operational readiness で causal candidate trace の欠落を出荷前に検知できるようにした。  
  * Stage E minimum gate を拡張し、`module_orchestration_integrity`、`counterfactual_lane_integrity`、`action_trace_observability` を `stage_e_contract` の必須チェックへ昇格した。release gate でも module orchestration / counterfactual lane / action trace の欠落を出荷前に検知できる。  
  * `release_soak` の Gate feedback / summary に Stage E フィールド（minimum failure count/details、readiness score、common spike space、temporal compression、dendritic gate、Spiking H-JEPA、reverse reasoning、module orchestration、counterfactual lane、action trace）を追加し、soak report から Stage E failure を直接監査できるようにした。  
  * `operational_readiness` の report / summary に Stage E snapshot を追加し、最上位の出荷判断レポートでも modular cognitive runtime readiness と minimum failure を確認できるようにした。  
  * `v1_release_gate` に `stage_e_runtime_minimum` と `operational_stage_e_snapshot` を追加し、最終出荷判定でも Stage E の Phase 3 minimum と operational 伝播 snapshot が両方 PASS でなければ v1.0 を通さないようにした。  
  * Stage E runtime trace の stable digest と replay comparison を追加し、同一入力で module order / selected action / counterfactual lane が再現されることを `runtime_trace_replay_consistency` として継続観測できるようにした。  
  * `runtime_trace_replay_consistency` を Stage E minimum gate へ昇格し、`stage_e_contract` / `phase3_accuracy_suite` / `release_gate` / `operational_readiness` / `release_soak` / 関連テストを同時更新して、runtime trace の再現性を出荷必須条件として常時検証する状態にした。  
  * `phase3_accuracy_summary` の Stage E セクションへ `stage_e_runtime_trace_replay_ready` 行を追加し、minimum check の完了可否を人間向け要約でも即時監査できるようにした。  

### **Phase 6: Autonomous General Intelligence (長期 / 最終目標)**

* **目標:** 完全に自律的で、多言語環境や物理環境と相互作用しながら自己成長を続ける強いAIの実現。  
* **タスク:**  
  * Spiking H-JEPA で獲得した世界モデルを基盤に、リアルタイム意思決定と Reward-Modulated STDP の実運用を確立する。  
  * 外部環境の言語（英語・日本語・フランス語等）に適応し、自発的に語彙や概念を獲得する多言語基盤を自己組織化する。  
  * エッジデバイスへの完全デプロイと、オンデバイスでの低消費電力な生涯学習（Lifelong Learning）を達成する。
  * 自律的に動作するbotを用意し、インターネット上のあらゆるデータを取得し、整理加工し、自動学習を停止することなく行う。

### **Post-Roadmap Direction: Next-Generation Adaptive Intelligence Stack (次期長期構想)**

* **位置づけ:**  
  * 現行の Phase 1-6 が一通り完成した後に進む「次の研究開発段階」です。  
  * 目標は、単なる lightweight assistant の延長ではなく、「少ない電力・少ない教師信号・少ない再学習で環境に適応し続ける脳型AI」へ移行することです。  
  * 方向性は、`Research on next-generation AI learning approaches.md` で整理した知見をベースに、SARA Engine の設計制約（no backprop dependency in runtime, no dense-matrix-first design, no GPU dependency, CPU-first, energy efficient）を守ったまま段階的に実装します。  

#### **次期構想の中核原則**

* **Brute-force scaling を主戦略にしない:**  
  * Transformer の巨大化や大量再学習を主軸にせず、環境適応・継続学習・省電力推論を主戦略に置く。  
* **世界モデルを中心に据える:**  
  * 単なる next-token continuation ではなく、「環境の潜在状態を予測する world model」を中心に据える。  
* **学習の二層化:**  
  * 速い適応は local plasticity / session adaptation / meta-adaptation で処理し、遅い安定化は replay / consolidation / structural plasticity で処理する。  
* **非同期・分業型アーキテクチャ:**  
  * perception, memory, world modeling, planning, action selection を単一巨大モデルに押し込まず、疎結合な協調系として構成する。  
* **ハードウェア整合性:**  
  * 将来の neuromorphic hardware や memristive synapse を見据え、STDP、multilevel synaptic weighting、event-driven routing と整合する設計を優先する。  

#### **次に目指す主要テーマ**

* **1. Meta-Adaptation Layer (少量経験からの高速適応):**  
  * 数回の観測や短い対話から task routing、readout、memory weighting を素早く最適化する。  
  * 勾配ベースの大規模 inner-loop を前提にせず、局所可塑性・可塑性ゲート・重み再配分・readout tuning を中心に設計する。  
  * `SaraInference` / `SaraAgent` の fast path や session memory を、将来は「meta-adaptation の実験場」として拡張する。  
* **2. Continual Learning Substrate (破局的忘却を抑えた生涯学習):**  
  * short-term memory, hippocampus, LTM, replay, consolidation を一体化し、運用中に知識が更新されても古い知識が崩れにくい基盤を作る。  
  * replay data と artifact upgrade 導線を、単なる再構築ツールではなく continual-learning evaluation loop へ発展させる。  
  * structural plasticity を単なる疎な再配線で終わらせず、複数モダリティや複数概念の交点になりやすい assembly へ結合を集約する `hub-neuron / high-order concept node` 形成へ発展させる。  
  * 長期的には preferential attachment に近い局所学習則を検討し、small-world 的な知識リンクを意図的に自己組織化させる。  
* **3. World Model / JEPA-WM Direction (潜在予測中心の知能):**  
  * 観測データをそのまま再生するのではなく、「次にどの潜在状態が来るか」を予測する層を追加する。  
  * Spiking H-JEPA を将来的な核としつつ、その前段として lightweight latent prediction benchmark を導入する。  
  * perception encoder, latent predictor, target state stabilizer, memory-guided planner を分けて実装する。  
  * 将来的には actual future prediction だけでなく、`if A then ... / if B then ...` の counterfactual branch を比較できるようにし、offline replay と仮想経験による world-model refinement へつなげる。  
* **4. AMI-like Modular Autonomy (認知モジュールの協調):**  
  * world model, planner, memory controller, action selector, critic を疎結合にし、1つの巨大モデルに依存しない自己改善ループを構築する。  
  * agent routing は今後、topic routing から「goal-conditioned modular routing」へ進化させる。  
* **5. Neuromorphic Efficiency Path (省電力・実装整合性):**  
  * 多値シナプス重み、STDP、local unsupervised plasticity、event-driven update を継続的に強化する。  
  * 将来の memristor / neuromorphic accelerator 実装を見据え、現在のソフトウェアでも sparse event log, bounded update, low-precision persistence を優先する。  
  * 低精度省電力路線の予備実装として、stochastic-computing 由来の lightweight readout 集約を benchmark / edge runtime で opt-in 検証可能にする。  
  * edge exporter にも opt-in の低精度重み永続化を追加し、runtime 互換を保ったまま low-precision persistence を段階導入する。  

### **Research Intake Notes (2026-05-09)**

本節は、2026年3月公開の関連論文をプロジェクト方針（SNN-based, no backprop dependency, no dense-matrix-first, CPU-first, energy efficient）でスクリーニングした結果を記録する。

#### **Paper A: Sparser, Faster, Lighter Transformer Language Models (arXiv:2603.23198v1)**

* **Adopt (policy-aligned ideas only):**
  * 推論・学習コストを一律に削るのではなく、`active path` に計算を集中させる設計思想。
  * 速度だけでなく、`throughput / memory / energy` を同時KPIとして追う評価姿勢。
  * 規模拡大時に「理論計算量」ではなく「実測効率」を重視する運用観点。
* **Do Not Adopt (policy conflict):**
  * CUDA kernel 最適化、GPU前提高速化、疎行列フォーマット依存の実装。
  * Transformer FFN の dense matmul 中心設計。
  * Backprop 系の正則化主導トレーニングを中核に置く方針。
* **Roadmap Action Items:**
  * `energy_efficiency_benchmark` を event-level へ拡張し、`spike-per-success` と `joule-per-success` を継続観測。
  * `active neuron ratio` と `routing sparsity` の診断を追加し、過活動ノードを恒常性制御へ接続。

#### **Paper B: LeWorldModel (arXiv:2603.19312v2)**

* **Adopt (policy-aligned ideas only):**
  * world-model を「再構成」ではなく「潜在遷移予測」中心で設計する方針。
  * collapse 回避を複雑な多項損失ではなく、単純で安定した制約へ集約する設計思想。
  * 物理整合性を latent probing / surprise test で評価する検証フレーム。
* **Do Not Adopt (policy conflict):**
  * pixel JEPA の end-to-end backprop 学習をそのまま採用すること。
  * GPU学習を前提にした実装導線。
* **Roadmap Action Items:**
  * Phase 5 の Spiking H-JEPA に `anti-collapse local constraints` を追加し、STDP + homeostasis で表現崩壊を抑制。
  * `future_state_consistency_benchmark` に `violation-of-expectation` 風の event-physics テストを追加。
  * planner 評価に `fixed compute budget` 比較を導入し、CPU-only 条件での安定性を定点観測。

#### **Paper C: δ-mem: Efficient Online Memory for Large Language Models (arXiv:2605.12357v1)**

* **Adopt (policy-aligned ideas only):**
  * 長い履歴を context window へ戻す代わりに、固定サイズの online associative memory state へ圧縮する設計思想。
  * delta-rule による residual write と retention/forget gate によって、予測済み情報を過剰更新せず、新しい差分だけを bounded に書き込む考え方。
  * memory-heavy task で、履歴長に比例しない小さな state を使って関連情報を再利用する評価姿勢。
  * text retrieval ではなく compact state の readout で現在の推論を steer する設計を、SARA では `memory_steering_event` として再解釈する。
* **Do Not Adopt (policy conflict):**
  * Transformer attention への dense low-rank correction を標準 runtime に入れること。
  * LLM hidden state projection、GPU 前提の学習済み projection、backpropagation による projection training を必須にすること。
  * 8x8 dense matrix という実装形をそのまま固定し、SARA の sparse event / low-precision / active-row storage 方針を崩すこと。
* **Roadmap Action Items:**
  * DONE: `DeltaAssociativeSpikeMemory` の lightweight prototype を追加し、sparse event key/value、prediction residual、retention gate、state budget を観測できるようにした。
  * DONE: `continual_consolidation_benchmark` に `delta_memory_residual_write_integrity_observed`、`delta_memory_retention_gate_stability_observed`、`delta_memory_interference_guard_observed`、`delta_memory_context_recall_without_text_reinjection_observed`、`delta_memory_state_budget_integrity_observed` を追加した。
  * DONE: `cognitive_runtime_benchmark` に `delta_memory_steering_integrity_observed`、`delta_memory_counterfactual_isolation_observed`、`delta_memory_trace_observability_observed` を observed-only で追加した。
  * DONE: edge exporter/runtime の Stage F payload に `delta_associative_state` profile を段階導入し、`edge_delta_state_persistence_observed`、`edge_delta_state_budget_observed`、`edge_delta_state_manifest_integrity_observed` と manifest validation を追加した。
  * DONE: δ-mem 指標を Phase 3 summary / release soak / operational readiness の Stage E snapshot へ接続し、長期運用で observed-only の回帰を見つけやすくした。
  * DONE: Stage D の residual write / retention / state budget と Stage F の edge delta state persistence を Phase 3 human-readable summary に表示し、δ-memの学習・推論・edge永続化を一つのレポートで横断監査できるようにした。
  * DONE: Loihi 2 / Lava、SpiNNaker 2、BrainChip Akida などへ早期ロックインせず、まず `spike_event_ir` と `neuromorphic_capabilities` を edge manifest 上で定義し、後から backend adapter を追加できる構造にした。
  * DONE: `lava_profile` / `spinnaker_profile` / `akida_profile` の変換仕様を実機なし compatibility report として分離し、profile ごとの差分を adapter 層で吸収できるようにした。
  * DONE: profile report / spike event IR / neuromorphic capability generation を `src/sara_engine/edge/neuromorphic.py` へ分離し、将来の実機 adapter が exporter 本体を肥大化させない構造へ整理した。
  * DONE: neuromorphic profile report を energy benchmark history で比較し、profile 追加時に event budget / low precision / online update policy の互換性が退行していないかを `neuromorphic_profile_trend` と `neuromorphic_profile_history_regression_observed` で履歴監査できるようにした。
  * DONE: `neuromorphic_profile_trend` を Phase 3 human-readable summary / release soak / operational readiness summary へ伝播し、長期runの report bundle から profile regression を直接確認できるようにした。
  * DONE: neuromorphic profile regression が発生した場合の recovery hint / repair action を operational readiness に追加し、profile 欠落・policy change・compatibility 低下ごとに次の確認コマンドを出せるようにした。
  * DONE: neuromorphic profile regression の詳細（missing profile 名、check_regression 種別、policy change）を operational summary の compact detail line に出し、修復前にどの profile が壊れたかを report だけで特定できるようにした。
  * DONE: `neuromorphic_profile_trend` の compact detail を release soak summary にも追加し、release bundle 単体でも profile regression の内訳を読めるようにした。
  * DONE: `neuromorphic_profile_trend` compact detail の生成ロジックを `compact_neuromorphic_profile_trend` へ共通 helper 化し、Phase 3 / release soak / operational readiness で表記ゆれや重複実装を減らした。
  * DONE: 実データ・長時間runで `delta_memory_*_observed` の変動を履歴比較するための `observed_trend_long_run_validation` runbook action を追加し、synthetic fixture 以外の steering / isolation / trace observability 確認を operational queue に載せられるようにした。

#### **Paper D: Towards end-to-end automation of AI research (Nature, 2026)**

* **Adopt (policy-aligned ideas only):**
  * 研究開発を `idea -> implementation -> experiment -> analysis -> report -> review` の閉ループとして扱い、SARA の ROADMAP / benchmark / release summary / operational readiness を自律的に回す研究運用フレームとして採用する。
  * The AI Scientist の template-based / template-free / tree-search experimentation を、SARA では GPU-heavy な foundation-model 研究生成ではなく、CPU-first SNN 仮説を複数設定で比較する lightweight experiment planner として再解釈する。
  * Automated Reviewer の考え方を、論文査読の代替ではなく、`research_review_report` として benchmark 結果、observed-only metric、regression trend、risk / novelty / reproducibility を機械可読に点検する仕組みに転用する。
  * negative result や failed hypothesis も artifact として保存し、同じ実験を繰り返さず、ROADMAP の採用しない要素 / pending hypotheses / promotion candidate 判断へ反映する。
  * agentic tree search は巨大な推論木ではなく、SARA では「小さな仮説分岐」「targeted fixture」「ablation」「negative result 記録」を結ぶ bounded experiment graph として扱う。各分岐は energy proxy、neuromorphic compatibility、observed-only regression、解釈可能 trace を必ず持つ。
  * research journal は単なるログではなく、専門サブモデル群の科学的モデル形成 memory として扱う。仮説、反証、修正案、採用/却下理由、次に試す最小実験を残し、同じ表面的 novelty や既知の失敗を繰り返さないための低コストな継続学習基盤にする。
  * automated review は「採点器」ではなく、human review を助ける sparse verifier / critic / risk triage として使う。採点対象は paper quality ではなく、SARA の設計制約に沿った reproducibility、local learning alignment、no-backprop 方針、energy impact、interpretability、hardware portability を優先する。
* **Do Not Adopt (policy conflict / safety risk):**
  * AI が生成した研究成果を人間レビューなしに外部公開・投稿・性能主張する運用。
  * LLM / foundation model の大規模 test-time compute を SARA runtime の中核要件にすること。
  * hallucinated citation / weak experiment / superficial novelty を release gate や ROADMAP に自動昇格させること。
  * automated reviewer の点数を単独の真実として扱い、人間の設計判断・安全判断・実装レビューを置き換えること。
* **Roadmap Action Items:**
  * DONE: `research_automation_benchmark.py` を追加し、既存の Phase 3 / release soak / operational readiness report から「次に検証すべき仮説」「既に十分安定した仮説」「退行した observed-only 指標」を抽出できるようにした。
  * DONE: `research_review_report` を managed `workspace/evaluation/` 配下に出力し、novelty / reproducibility / energy impact / release-gate safety / neuromorphic compatibility を English log で採点できるようにした。
  * DONE: `linear_snn_fusion_observed_trend` と `neuromorphic_profile_trend` を research review の入力にし、synthetic fixture 以外の長時間 run で崩れた仮説を `next_hypotheses` / `regression_watchlist` 候補へ回せるようにした。
  * DONE: ROADMAP への自動追記は直接実行せず、`roadmap_patch_suggestion` として差分案だけを生成し、人間承認後に反映する方針を artifact 化した。
  * DONE: 失敗実験や negative result を `workspace/evaluation/research_journal.jsonl` に追記できる optional flow を追加し、採用しない方針・再試行条件・次の小実験を追跡できるようにした。
  * DONE: `research_review_report` の compact snapshot を release soak / operational readiness の上位 summary に接続し、研究仮説の退行が長時間 run の最後に見落とされないようにした。
  * DONE: `research_journal.jsonl` に dedupe key / seen count / last seen timestamp / max-age / max-entry pruning を追加し、同じ negative result が短時間に繰り返し蓄積されすぎないようにした。
  * DONE: `roadmap_patch_suggestion` を operational runbook action (`roadmap_patch_review`) として提示し、直接適用ではなく `research_automation_benchmark.py --append-journal` 経由の review queue へ回す運用にした。
  * DONE: research journal の集計 summary（頻出 negative result / regression watchlist / next hypothesis / roadmap patch 承認・却下件数）を operational readiness summary / runbook に表示できるようにした。
  * DONE: `roadmap_patch_suggestion` の内容を operational runbook Markdown に compact preview として表示し、人間レビュー時にJSONを開かず要点を読めるようにした。
  * DONE: `roadmap_patch_review` action 実行後の承認/却下結果を repair log に明示的に残し、同じ research review に対して採用されなかった提案が再提案され続けないようにした。
  * DONE: `roadmap_patch_review` の承認/却下履歴を research journal summary と統合し、却下理由が次回の experiment planner に反映されるようにした。
  * DONE: research journal summary の頻出 negative result / regression watchlist / next hypothesis から、再計測すべき benchmark コマンドを `research_journal_remeasure` runbook action として自動提案できるようにした。
  * DONE: `research_journal_remeasure` の実行結果を journal entry に紐づけ、同じ失敗が再計測後に改善したかを `remeasure_trends` として operational summary / runbook に表示できるようにした。
  * DONE: `remeasure_trends` の recovered / still_failing を experiment planner の優先度に反映し、改善済み仮説の再提案を弱め、未改善仮説の再計測間隔を明示できるようにした。
  * DONE: `recommended_benchmark_actions` にも `remeasure_trends` を反映し、recovered 済みの再計測actionを一定期間 `suppressed_benchmark_actions` へ移し、still_failing は短い retry interval 後に再提案できるようにした。
  * DONE: suppressed / recommended benchmark action の履歴を repair log と突き合わせ、同一 command が過剰に runbook を占有する場合は remeasure command history quota で抑制できるようにした。
  * DONE: remeasure command history quota の抑制結果を research journal summary に戻し、どの仮説が「再計測待ち」ではなく「探索多様性のため保留」になったかを `remeasure_quota_holds` として journal summary / runbook に表示できるようにした。
  * DONE: `remeasure_quota_holds` が継続する仮説について、同一benchmarkの再実行ではなく `research_journal_alternative_probe` と `alternative_benchmark_actions` で targeted fixture / lightweight gate を提案する planner branch を追加した。
  * DONE: alternative probe の実行結果を journal entry に紐づけ、元の quota-held hypothesis が targeted fixture で切り分け済みかを `alternative_probe_trends` として表示できるようにした。
  * DONE: `alternative_probe_trends` が passed の仮説は full benchmark 再実行よりも `cause_boundary_documentation_tasks` を優先し、failed の仮説は `targeted_fixture_repair_tasks` として最小fixtureの追加・修正を planner branch に出せるようにした。
  * DONE: `cause_boundary_documentation_tasks` / `targeted_fixture_repair_tasks` の完了結果を repair log から取り込み、同じ代替probeが未完了taskとして残り続けないようにした。
  * DONE: 完了済み planner task の履歴を `research_journal.jsonl` の各 entry にも反映し、journal summary だけでなく元の negative result 単位で完了状態を追跡できるようにした。
  * DONE: research planner task の未完了/完了比率を operational readiness の research review 補助信号に加え、未完了taskが蓄積した場合は roadmap patch review より先に `research_planner_task_cleanup` を提案するようにした。
  * DONE: `research_planner_task_cleanup` の pending / success / skipped 履歴を research journal summary に戻し、cleanup 自体の滞留も次回runbookで `research_planner_task_cleanup_stalled` として検出できるようにした。
  * DONE: `research_planner_task_cleanup_stalled` が連続する場合、原因を「manual review待ち」「fixture実装待ち」「documentation未反映」に分類し、runbook action を `research_planner_manual_review_followup` / `research_planner_fixture_repair_followup` / `research_planner_documentation_followup` へ分岐させるようにした。
  * DONE: AI Scientist 風の `bounded_experiment_graph` を research automation に追加し、各 hypothesis を template-based probe / template-free probe / ablation / alternative probe の小さな node として記録する。node は benchmark command、observed metrics、negative result、promotion blocker、next minimal experiment を持ち、巨大な test-time compute ではなく探索の再現性と多様性を優先する。
  * DONE: `bounded_experiment_graph` を `research_review_report` / compact summary / research journal snapshot / release soak summary / operational readiness summary へ接続し、研究探索が「未整理のログ」ではなく bounded node / edge / stage count として監査できるようにした。
  * DONE: `research_review_report` を SARA policy reviewer として拡張し、novelty / reproducibility / energy impact / neuromorphic compatibility に加えて、no-backprop alignment、sparse event alignment、local learning alignment、interpretability trace coverage、submodel integration impact を個別採点する。
  * DONE: SARA policy reviewer の失敗は `sara_policy_alignment_recovery` として bounded experiment graph / research journal / release soak summary / operational readiness summary へ伝播し、dense matrix・GPU・backprop への逸脱ではなく targeted fixture と observed-only repair に回す方針にした。
  * DONE: experiment graph と research journal から `experiment_status_summary` を生成し、「採用候補」「退行中」「反証済み」「人間レビュー待ち」を compact に分類できるようにした。分類結果は research review compact / research journal summary / release soak summary / operational readiness summary / runbook に表示し、ROADMAP への反映は引き続き人間承認つき patch suggestion に限定する。
  * DONE: `experiment_status_summary` から `experiment_priority_plan` を生成し、退行中は high-priority remeasure、採用候補は bounded promotion review、人間レビュー待ちは evidence followup、反証済みは suppression review へ振り分けるようにした。priority plan は release soak / operational readiness / runbook action manifest に伝播し、研究分類を次の実行キューへ接続する。
  * DONE: 採用候補を `experiment_promotion_target_plan` へ分類し、Stage B / Stage D / Stage E / neuromorphic profile / research policy のどの promotion surface に接続すべきかを機械可読化した。既存 minimum gate は `already_minimum`、未分類候補は `manual_mapping_review` とし、直接 gate へ書き込まず runbook action manifest の `experiment_promotion_target_review` で人間レビューへ回す。
  * DONE: Stage E に `observed_acceptance_candidates` と `observed_acceptance_candidate_stability` を追加し、linear SNN fusion、plastic submodel、micro-turn / phase-block、manifold trace、delta-memory observed 指標を minimum gate へ即時昇格せず、acceptance candidate として達成数・失敗数・連続達成 streak で監査できるようにした。recommended 時は `stage_e_observed_acceptance_candidate_stability` runbook action として人間レビューへ回す。
  * DONE: Stage E observed acceptance candidate の失敗を `stage_e_observed_acceptance_candidate_failure` として Phase3 / release soak / operational readiness summary に value / threshold / description 付きで表示し、未達時は `stage_e_observed_acceptance_candidate_repair` runbook action として修復キューへ載せられるようにした。
  * DONE: Stage E observed acceptance candidate の未達を `research_automation_benchmark` の `stage_e_observed_acceptance_candidates` signal に接続し、`stage_e_observed_acceptance_candidate_repair` として next hypothesis / regression watchlist / negative result / bounded experiment graph / priority plan へ戻せるようにした。これで Stage E candidate の失敗は operational queue だけでなく研究ループにも戻る。
  * DONE: `stage_e_observed_acceptance_candidate_repair` を research journal の専用 repair loop summary として追跡し、再測定推奨 / 抑制 / latest trend / targeted alternative probe 推奨を operational readiness summary と runbook に表示できるようにした。これで Stage E observed candidate の失敗は detect → repair → remeasure / targeted probe → journal feedback の循環として監査できる。
  * DONE: `stage_e_observed_acceptance_candidate_repair` が remeasure / targeted alternative probe で回復した場合は `recovery_confirmed` と `promotion_review_recommended` を journal summary に出し、operational readiness の `stage_e_observed_acceptance_candidate_recovery_review` action から既存の `stage_e_observed_acceptance_candidate_stability` review へ戻せるようにした。これで失敗修復後の候補を放置せず、minimum promotion review へ再接続できる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review` の pending / success / skipped / failed 履歴を repair log から journal summary に取り込み、review 完了後または pending 中は同じ recovery review action を再提案しないようにした。これで Stage E observed candidate の recovery review は提案・実行・完了抑制まで一巡して監査できる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review` の実行結果を research journal entry に永続同期し、次回実行時も summary を repair log 依存だけでなく journal 履歴から復元できるようにした。これで recovery review の完了記録は一時的な operational summary ではなく、研究ループの継続記憶として残る。
  * DONE: release soak summary に Stage E observed candidate recovery review の compact status（recovery confirmed / review recommended / completed / in-progress / latest status / count）を追加し、長時間runの最後でも recovery review の完了・滞留を確認できるようにした。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review` が pending のまま滞留した場合に stale 判定と age を summary / release soak compact status に表示し、operational runbook に `stage_e_observed_acceptance_candidate_recovery_review_followup` action を出せるようにした。これで recovery review は pending 放置ではなく、完了・再確認・追跡へ戻る。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_followup` の pending / success 履歴を recovery review summary と release soak compact status に取り込み、follow-up が pending 中または完了済みの場合は同じ follow-up action を再発行しないようにした。これで stale recovery review の二次追跡も重複せずに閉じられる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_followup` の failed / timeout / error 履歴を recovery review summary と release soak compact status に取り込み、失敗した follow-up は `stage_e_observed_acceptance_candidate_recovery_review_followup_retry` として再試行 action に戻せるようにした。これで stale review の追跡自体が失敗した場合も、silent failure ではなく再修復ループへ戻る。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_followup_retry` の pending / success / failed 履歴を recovery review summary と release soak compact status に分離表示し、retry が pending 中または完了済みの場合は同じ retry action を再発行しないようにした。これで follow-up failure の再試行も重複せず、失敗・再試行・完了の状態遷移を監査できる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_followup_retry` 自体が failed / timeout / error になった場合は、同じ retry を積み続けず `stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation` として高優先度 action へ切り替えるようにした。escalation の pending / success / failed も recovery review summary と release soak compact status に表示し、retry 不能な stale review を人間レビュー・原因切り分けへ戻せる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation` も failed / timeout / error になった場合は、再 escalation を積まず `stage_e_observed_acceptance_candidate_recovery_review_evidence_collection` へ fallback し、追加証拠収集を high-priority action として提示するようにした。evidence collection の pending / success / failed も recovery review summary と release soak compact status に表示し、review 系の失敗を bounded evidence loop へ戻せる。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_evidence_collection` が success / skipped になった場合は、`stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck` action へ接続し、収集済み証拠を Stage E stability review に再投入できるようにした。evidence recheck の pending / success / failed も recovery review summary と release soak compact status に表示し、pending 中または完了済みの場合は同じ recheck action を再発行しない。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck` が failed / timeout / error になった場合は、同じ recheck を積み続けず `stage_e_observed_acceptance_candidate_recovery_review_targeted_probe` へ分岐し、証拠と stability review の不一致を targeted probe で切り分けられるようにした。targeted probe の pending / success / failed も recovery review summary と release soak compact status に表示する。
  * DONE: `stage_e_observed_acceptance_candidate_recovery_review_targeted_probe` が success / skipped になった場合は、`stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck` action へ接続し、probe で得た局所証拠を Stage E stability review に再投入できるようにした。targeted probe recheck の pending / success / failed も recovery review summary と release soak compact status に表示し、pending 中または完了済みの場合は同じ recheck action を再発行しない。
  * DONE: research planner task cleanup の分類結果を release soak summary にも compact 表示し、長時間runの最後でも cleanup 滞留理由を見落とさないようにした。
  * DONE: `roadmap_patch_review` の却下理由を experiment planner の抑制理由へ接続し、同じ却下済み提案が journal / runbook に再浮上する場合は追加証拠待ちとして `roadmap_patch_suggestion` から抑制するようにした。
  * DONE: `roadmap_patch_review_suppressed` の仮説に対して、追加証拠が揃った時だけ再提案を解除する evidence refresh policy（targeted probe success / remeasure recovery / planner task completion）を追加した。
  * DONE: evidence refresh policy の解除理由を release soak summary にも compact 表示し、抑制解除が長時間runの最後で見落とされないようにした。
  * DONE: evidence refresh policy を長時間run相当の journal sequence に適用し、同じ追加証拠で却下済み提案が NEXT / DOC / FIXTURE として過剰に再浮上しないようにした。
  * DONE: 実データ・長時間run artifact で `roadmap_patch_refreshed_items` / `roadmap_patch_rejected_items` の比率を記録し、refresh policy の過抑制・過再提案を operational readiness の観測値として監視できるようにした。
  * DONE: `roadmap_patch_refresh_to_rejection_ratio` の健全域を long-run 履歴から推定し、過剰再提案・過抑制のどちらかに偏った場合の runbook followup を追加した。
  * DONE: refresh policy followup の実行結果を repair log / research journal summary に反映し、閾値調整後に同じ followup が残り続けないようにした。
  * DONE: refresh policy followup の failed / timeout が続く場合、閾値調整ではなく evidence collection path（targeted probe / real-data fixture / release-soak trend）へ切り替える fallback branch を追加した。
  * DONE: evidence collection fallback の成功結果を `roadmap_patch_refreshed_items` と区別して summary に記録し、閾値調整ではなく証拠追加で解決したケースを追跡できるようにした。
  * DONE: evidence collection fallback の成功結果から、次に要求すべき targeted probe / real-data fixture / release-soak trend の種類を分類して runbook に提示できるようにした。
  * DONE: evidence collection kind の履歴を `research_automation_benchmark` 側の planner priority に反映し、targeted probe 済みの仮説は real-data fixture / release-soak trend を優先するようにした。
  * DONE: `roadmap_patch_evidence_collection_tasks` の完了結果を journal entry に反映し、real-data fixture / release-soak trend 済みの仮説が同じ evidence task を再要求しないようにした。
  * DONE: completed evidence collection key を release soak compact / runbook action にも表示し、長時間run後に「証拠収集済みだが未反映」の候補だけを再確認できるようにした。

#### **Integration Policy**

* 新規実装は `src/sara_engine/utils/project_paths.py` 経由で managed directories のみへ出力する。
* 追加ログと評価指標の文言は English で統一する。
* 実装優先順は「評価指標追加 → 軽量プロトタイプ → 本体統合」とし、既存リリースゲートを壊さない。

#### **このリポジトリでの実装順序**

* **Stage A: Evaluation First**
  * meta-learning, continual learning, world-modeling をいきなり本実装せず、先に benchmark と acceptance criteria を定義する。  
  * 追加候補:
    * task-switch adaptation benchmark  
    * long-horizon continual retention benchmark  
    * latent prediction / future-state consistency benchmark  
    * energy-per-success / memory-per-success 指標  
  * **現在の進捗:**  
    * `task-switch adaptation`、`future-state consistency`、`energy efficiency` は benchmark として実装済み。  
    * `adaptive_readiness`、`predictive_readiness`、`efficiency_readiness`、`direction_shift_following` を suite / release summary / trend 比較で継続観測可能にした。  
* **Stage B: Lightweight World Model Prototypes**
  * `SaraInference` / `SpikingLLM` の上に、小さな latent-state predictor を載せる。  
  * 初期段階ではテキストと構造化状態だけを対象にし、画像・音声は後回しにする。  
  * `next step` 提案や topic continuity を、単なる template ではなく future-state prediction で支える方向へ移行する。  
  * **現在の進捗:**  
    * `predictor_state`、`future_state_runtime_state`、`predicted_action`、`predicted_target_state`、`predicted_command` を持つ lightweight world-model proxy を実装済み。  
    * shift-aware next-step response、predictor snapshot 可視化、runtime transition/shift tracking を benchmark / release summary / memory health report で追跡可能にした。  
  * **次の拡張候補:**  
    * 単一路線の future-state prediction に加え、small branching な counterfactual candidate を同時保持し、複数の `predicted_action -> predicted_target_state` を比較する lightweight offline simulation へ広げる。  
  * **World Model + lightweight reinforcement 方針（v1.1以降の優先実装）:**  
    * 汎用知能方向の主軸は、PPO/DQN などの重い ANN-style 強化学習ではなく、lightweight world model の精度と使い勝手を上げることに置く。  
    * world model は「次状態を当てる」だけでなく、複数の action candidate を作り、operator trace / counterfactual branch / rollback observability / energy cost を比較する意思決定基盤へ拡張する。  
    * 強化学習的要素は、Reward-Modulated STDP、eligibility trace、replay priority、retrieval/action candidate の局所的な重みづけとして導入する。  
    * 報酬は巨大な value network ではなく、`progress_score`、`risk_reduction_score`、`reversibility_score`、`energy_cost_proxy`、`user_feedback_signal` のような bounded scalar event として扱う。  
    * まずは `future_state_runtime_state` に `reward_trace` と `policy_trace` を追加し、action selection がどの報酬イベントで強化・抑制されたかを release summary で監査できるようにする。  
    * 次に `future_state_consistency_benchmark` へ `rewarded_action_selection_integrity`、`policy_update_stability`、`energy_aware_action_preference` を追加し、Stage B minimum へ段階昇格できる形で観測する。  
  * **採用しない強化学習要素（本プロジェクト制約との不整合）:**  
    * backpropagation 前提の policy network / value network を runtime の中核に置くこと。  
    * GPU 前提の大量 rollout や dense replay buffer training を必須にすること。  
    * 行動選択をブラックボックス化し、operator trace や reward trace なしに release gate を通すこと。  
  * **arXiv:2603.21852v2 由来で採用する要素（適用範囲を限定）:**  
    * world-model の内部表現を「連続ベクトル一括更新」ではなく、operator 単位の離散遷移列（lightweight symbolic trace）として保持し、SNN の event-driven 実行と整合させる。  
    * `predicted_action -> predicted_target_state` の間に `transition_operator` を明示し、予測結果だけでなく「どの操作で状態を遷移させたか」を保存・診断できるようにする。  
    * counterfactual branch 比較で、状態一致率だけでなく operator 一貫性（同一条件で同系統の操作を再選択できるか）を評価軸に追加する。  
    * planner / world-model 間インターフェースを「operator vocabulary + state slot update」形式へ寄せ、将来の neuromorphic 実装でも低ビット・疎イベント表現を維持できるようにする。  
  * **arXiv:2512.24601 (Recursive Language Models) 由来で採用する要素（適用範囲を限定）:**  
    * 長文プロンプトを「そのままモデルへ投入する対象」ではなく、外部環境上の可観測オブジェクトとして扱い、必要断片だけを段階的に参照する設計を導入する。  
    * summary compaction 一辺倒ではなく、`peek -> decompose -> focused sub-call` の手順を lightweight に再現し、長文時の情報欠落を減らす。  
    * recursive sub-call そのものより、SARA では `branch candidate` / `operator trace` / `refinement loop` に対応する「段階的探索フレーム」を優先採用し、既存の world-model runtime 観測と統合する。  
    * Stage B/Stage D の benchmark で、長文入力時の `focused retrieval hit ratio` と `branch-level decision consistency` を追加候補として段階導入し、release gate で常時監視できる形へ広げる。  
  * **RLM由来で採用しない要素（本プロジェクト制約との不整合）:**  
    * 汎用 LLM の高コストな REPL 実行を前提にした深い再帰スタックを標準経路にすること。  
    * dense Transformer を主役にした inference-time scaling を中核戦略に据えること。  
    * GPU 前提・大規模モデル前提の長文処理戦略を release gate の必須要件にすること。  
  * **採用しない要素（本プロジェクト制約との不整合）:**  
    * end-to-end backpropagation 前提の最適化。  
    * dense matrix 主体の latent 一括変換。  
    * GPU 前提の高密度演算パイプライン。  
  * **Stage B acceptance criteria への追加（release gate 連携）:**  
    * `stage_b_readiness` の world-model minimum として、`transition_operator_coverage`、`operator_consistency`、`counterfactual_branch_viability` を必須項目に追加する。  
    * gate 判定は「予測が当たるか」だけでなく、「予測遷移の説明可能性（operator trace が欠落しないか）」を含める。  
    * 既存の `predictive_readiness` と energy 指標を維持し、operator trace 追加で消費コストが閾値を超えないことを同時に確認する。  
  * **`dflash-mlx` から採用した運用アイデア（SNN向け再解釈）:**  
    * speculative decode の `draft -> verify -> accept/reject` という観測フレームを、world-model の next-step prediction に適用し、`speculative_acceptance_ratio` と `speculative_rollback_observability` を継続計測する。  
    * full snapshot rollback ではなく「差分だけ追跡して可視化する」思想を採用し、runtime 上では rollback 要否と branch viability を軽量メトリクスとして扱う。  
    * 数値一貫性を守る dflash の設計思想を、SARA では「operator consistency（同条件で同系統の遷移演算子を再選択できるか）」の品質指標として導入する。  
  * **fluid-inspired supplementary dynamics の導入（実装済みの初期統合）:**  
    * 独立した `FluidFieldDynamics` を補助ダイナミクスとして追加し、主推論器を置き換えずに `predictor_state` の補助観測へ統合する。  
    * flow / diffusion / vortex の scalar-only 更新を `fluid_trace` として保存し、`bounded`、`support_score`、`active_columns`、`total_spikes` を predictive runtime から観測できるようにする。  
    * world-model の operator/speculative trace と並べて確認できるようにし、脳型・event-driven な補助場として段階的に評価する。  
    * `stage_b_readiness` の minimum check に `fluid_trace_integrity` と `fluid_support_integrity` を追加し、release gate / memory health / release summary でも補助場の最低健全性を確認できるようにする。  
  * **LoopLM系アイデアの限定採用（実装済み）:**  
    * full LoopLM / dense verification loop は採用せず、SNN制約に合う lightweight `adaptive refinement loop` のみを導入する。  
    * `refinement_trace`（`triggered`、`loop_count`、`selected_branch_before/after`、`score_gap_before/after`）を `predictor_state` に記録し、future-state benchmark / release summary で観測できるようにする。  
    * `stage_b_readiness` と release gate の world-model minimum に `future_state_refinement_loop_integrity` と `future_state_adaptive_refinement` を追加し、minimum条件として常時検証する。  
  * **Inference-time reasoning / verification の採用方針（SARA向け限定採用）:**  
    * Best-of-N は dense LLM の大量サンプリングではなく、`sparse_best_of_n` として少数の branch candidate（memory-prior / world-model-prior / low-energy-prior など）を生成し、event trace と verifier score で選ぶ lightweight 経路として採用する。  
    * Verifier は巨大な別モデルではなく、`sparse_verifier` として retrieval grounding、operator consistency、counterfactual branch viability、energy cost、uncertainty を採点する局所検証器から始める。  
    * MCTS は標準経路にそのまま導入せず、`bounded_tree_search` として depth / branch factor / event budget を固定した小さな探索木に限定する。rollout は dense simulation ではなく world-model の sparse transition trace を再利用する。  
    * 計算の深さ調整は、全入力で冗長推論を行うのではなく、uncertainty / retrieval conflict / verifier failure が閾値を超えた場合だけ `adaptive_depth_budget` を増やす。通常ケースは早期停止し、難問だけ `draft -> verify -> repair` を追加実行する。  
    * Hierarchical CoT は自然言語の長い思考文ではなく、`instruction_event -> execution_trace -> verification_trace` の階層化された sparse trace として採用する。plan-execution drift は operator trace と selected action の不一致で検出する。  
    * Forest-of-Thought は大規模並列推論ではなく、primary / counterfactual lane に加えて少数の `reasoning_forest_lane` を評価する限定機能にする。各 lane は state を汚染しない read-only snapshot を使い、最終選択理由を保存する。  
    * Self-Correction は回答生成中の `self_correction_trace` として導入する。ただし無制限の反省ループは禁止し、最大ループ数、改善幅、rollback reason、verifier failure reason を release summary で監査できるようにする。  
    * RAG 高度化は優先採用する。既存の SNN RAG / retrieval diagnostics の上に、`sparse_rag_rerank`、source agreement、contradiction flag、freshness / source reliability、query decomposition を追加候補とする。  
    * コード実行による検証は、数学・論理・構造化データ・benchmark 修復のような明確に実行可能な問題に限定して採用する。通常会話の全回答に Python 実行を必須化せず、`tool_verification_trace` と managed output policy を守る。  
  * **Inference-time reasoning で採用しない要素（本プロジェクト制約との不整合）:**  
    * 大量 Best-of-N、自己一致性サンプリング、大規模 MCTS rollout など、GPU / dense Transformer / 高トークン消費を前提にした test-time compute を標準経路にすること。  
    * Chain-of-Thought の長文自然言語ログを release artifact の必須出力にすること。代わりに、短い machine-readable trace と verifier summary を保存する。  
    * Verifier をブラックボックスの大規模外部モデルに依存させ、SARA の operator trace / retrieval evidence / energy budget と切り離して採点すること。  
    * 無制限の自己修正ループ、深い再帰探索、外部ツールの無差別実行を標準推論に組み込むこと。  
    * RAG の取得量を単に増やすだけで精度改善とみなすこと。取得量増加は rerank / contradiction detection / citation grounding とセットでのみ採用する。  
  * **Inference-time reasoning の追加実装候補:**  
    * DONE: `sparse_verifier` の最小 API を定義し、candidate answer / action branch に対して grounding、operator consistency、energy cost、uncertainty を返す observed-only benchmark を追加した。  
    * DONE: `adaptive_depth_budget` を world-model / agent runtime に接続し、uncertainty が高いケースだけ refinement loop を増やすかを `adaptive_depth_efficiency` として測定できるようにした。  
    * DONE: `sparse_best_of_n` を primary / counterfactual / retrieval-heavy の3候補程度から始め、Verifier が選択した branch と人間可読 summary が一致するかを Stage E observed metric で確認できるようにした。  
    * DONE: `sparse_rag_rerank` を SNN RAG に追加し、retrieval score だけでなく source agreement / contradiction / freshness を用いた再ランキングを real-data external validity benchmark へ接続した。  
    * DONE: `sparse_rag_rerank` の選択チャンクに `citation_id` を追加し、source / chunk index に grounded な引用可能結果だけを `sparse_rag_rerank_citation_grounding_observed` として real-data external validity で監査できるようにした。  
    * DONE: `sparse_rag_rerank` の source reliability を observed metric 化し、選択チャンクが匿名・出所不明の検索結果に偏っていないかを `sparse_rag_rerank_source_reliability_observed` として監査できるようにした。  
    * DONE: `tool_verification_trace` を導入し、コード実行で確認した結果だけを managed artifact / repair log に残し、通常推論の副作用や unmanaged output を避けられるようにした。  
    * DONE: `self_correction_trace` を bounded draft-verify-repair として追加し、最大loop数、改善幅、rollback reason、verifier failure reason を Stage E observed metric で監査できるようにした。  
    * DONE: `bounded_tree_search` を depth / branch factor / event budget 固定の小さな sparse transition tree として追加し、rollout 境界と Verifier 選択を Stage E observed metric で確認できるようにした。  
    * DONE: `reasoning_forest_lane` を少数 read-only snapshot lane として追加し、lane diversity、state 非汚染、Verifier 選択、選択理由の整合性を Stage E observed metric で監査できるようにした。  
    * DONE: `query decomposition` を SNN RAG に追加し、長い query を bounded subquery に分解して `sparse_rag_rerank` へ渡し、coverage / subquery hit / merged selection を real-data external validity で観測できるようにした。  
    * DONE: `query decomposition` の merged selection にも citation grounding / source reliability observed metric を追加し、subquery 統合後に出所付き evidence の品質が落ちていないかを監査できるようにした。  
    * DONE: RAG rerank / decomposed rerank に source diversity observed metric を追加し、複数 evidence 選択時に単一ソースへ偏りすぎていないかを監査できるようにした。  
    * DONE: `Hierarchical CoT` を自然言語の長文思考ログではなく `instruction_event -> execution_trace -> verification_trace` の sparse hierarchical trace として追加し、plan-execution alignment を Stage E observed metric で確認できるようにした。  
  * **World Model + lightweight reinforcement 観測の初期統合（今回実装）:**  
    * `future_state_runtime_state` に `rewarded_selection_ratio`、`policy_stability_ratio`、`energy_aware_preference_ratio` を追加し、行動選択が報酬イベントと方策更新でどう変化したかを runtime snapshot で監査できるようにした。  
    * `predictor_state` に `reward_trace`（`progress_score`、`risk_reduction_score`、`reversibility_score`、`energy_cost_proxy`、`user_feedback_signal`、`total_reward`）と `policy_trace`（`selected_branch`、`best_simulated_branch`、`policy_shift_applied`、`policy_stability`）を追加し、action selection の強化・抑制要因を可視化した。  
    * `future_state_consistency_benchmark` に `future_state_rewarded_action_selection_integrity`、`future_state_policy_update_stability`、`future_state_energy_aware_action_preference` を追加し、Phase 3 / predictive readiness で継続観測できるようにした。  
    * `release_soak` summary に新しい runtime ratio 行を追加し、release review 上でも reward/policy 系の挙動を直接確認できるようにした。  
    * `stage_b_readiness` に `promotion_candidate_ready` / `promotion_candidate_failure_count` / `promotion_candidate_checks` を追加し、上記3指標の minimum 昇格可否を機械判定できるようにした。Phase 3 summary / release soak summary / gate feedback にも同判定を露出し、昇格判断を運用レポート上で追跡できるようにした。  
    * `promotion_readiness`（`required_streak`、`consecutive_passes`、`recommended`）を追加し、履歴ベースで「minimum へ昇格提案してよいか」を自動判定できるようにした。現状は `required_streak=3` で連続達成時に `recommended=true` となる。  
    * `phase3_accuracy_suite` / `release_soak` に `--stage-b-promotion-required-streak` を追加し、昇格提案の連続達成条件を運用環境に合わせて調整できるようにした。  
    * `release_soak` の Gate フィードバックに `stage_b_promotion_next_step_hint` と `stage_b_promotion_actions` を追加し、`recommended=true` 時に契約更新と再検証の具体手順をそのまま実行できる形で提示できるようにした。  
    * `stage_b_contract` の minimum check に `future_state_rewarded_action_selection_integrity`、`future_state_policy_update_stability`、`future_state_energy_aware_action_preference` を正式追加し、reward / policy / energy-aware action preference を world-model prototype の出荷必須条件へ昇格した。昇格後は `promotion_candidate_promoted=true` として記録し、追加の promotion follow-up は出さない。  
    * `v1_release_gate` に `stage_b_reward_policy_minimum` を追加し、最終出荷判定でも Stage B の reward / policy / energy-aware minimum が明示 PASS でなければ v1.0 を通さないようにした。  
  * **RLM由来の long-context / branch consistency 観測（今回実装）:**  
    * full dense RLM は採用せず、SNN制約に合う軽量観測として `future_state_consistency_benchmark` に long noisy context case を追加した。長いアーカイブノイズがあっても、最新の goal/task に基づく action response へ集中できることを `future_state_focused_retrieval_hit_ratio` で計測する。  
    * `future_state_branch_level_decision_consistency` を追加し、branch ranking、lightweight simulation、最終選択が同じ preferred branch に揃うかを Phase 3 / predictive readiness で継続観測できるようにした。  
    * 上記2指標は `stage_b_readiness` と `release_soak` summary に observed check として露出し、連続達成後に Stage B minimum へ昇格した。  
    * `stage_b_contract` の minimum check に `future_state_focused_retrieval_hit_ratio` と `future_state_branch_level_decision_consistency` を正式追加し、長文ノイズ下の focused retrieval と branch-level decision consistency を world-model prototype の出荷必須条件にした。昇格後は `rlm_observation_candidate_promoted=true` として記録し、追加の promotion follow-up は出さない。  
    * `v1_release_gate` に `stage_b_rlm_observation_minimum` を追加し、最終出荷判定でも RLM 由来の long-context / branch consistency minimum が明示 PASS でなければ v1.0 を通さないようにした。  
  * **Stage B gate contract の運用強化（実装済み）:**  
    * Stage B minimum check の定義を `stage_b_contract` として共通化し、phase3 suite と release gate が同じ仕様を参照するように統一した。  
    * `stage_b_readiness` に `minimum_failure_count` / `minimum_failures`（check名・説明・実測値・閾値）を追加し、未達理由を機械可読に保持できるようにした。  
    * release gate のエラー文を value付き（`value=... required>=...`）に拡張し、release soak summary の Gate セクションでも failure件数と失敗checkを確認できるようにした。  
    * release gate / release soak feedback に `recovery_actions` を追加し、失敗内容に応じた再実行コマンド（phase3 suite 再計測、extended soak 再実行など）を自動提案できるようにした。  
    * `recovery_actions` に `priority`（high/medium/low）と `expected_effect` を追加し、Gate failure時に「どの手順を先に実行すべきか」と「何が回復するか」を summary 上で即判別できるようにした。  
    * `recovery_actions` に `affected_checks` を追加し、各提案がどの gate チェック（例: `stage_b.minimum_checks`、`soak.duration_seconds`、`release_gate.embedded_accuracy_present`）を回復対象にしているかを機械可読に追跡できるようにした。  
    * `recovery_actions` から `build_release_gate_repair_plan` を生成し、`repair_plan`（step順、coverage、未カバー項目）を release gate / release soak summary に表示できるようにした。  
    * `repair_plan` に `fallback_actions` を追加し、`uncovered_checks` が残るケースでも診断再収集→gate再実行の代替手順を自動提示できるようにした。未知エラーは `release_gate.unknown_error` として分類し、fallback planner の対象へ接続した。  
    * `build_iterative_release_gate_repair_plan` を追加し、`repair_execution_log`（success/failed + covered_checks）を反映して `remaining_checks` と `next_actions` を再計算する iterative repair loop を実装した。release soak summary の Gate セクションで iterative next step を可視化できる。  
    * iterative repair loop に `completed` / `auto_stopped` / `stop_reason` / `next_step_hint` を追加し、`remaining_checks=0` のときは自動停止して次アクションを出さない運用状態を明示できるようにした。  
    * release gate CLI に `--repair-log-path` / `--repair-plan-path` を追加し、外部実行ログ(JSON/JSONL)を入力に iterative plan を再計算しつつ、`workspace/release/release_gate_repair_plan.json` へ recovery/repair artifact を出力できるようにした。  
    * release soak CLI に `--repair-log-path` / `--repair-log-command` / `--repair-log-status` / `--repair-log-covered-checks` / `--append-iterative-next-actions` を追加し、手動実行ログ追記と iterative next action の pending 自動追記を同一フローで回せる半自動修復ランナーを実装した。  
    * release soak に `load/save/append` 系の repair-log helper を追加し、managed path 制約を守ったまま repair execution log の継続更新（manual + iterative）を安定運用できるようにした。  
    * pending な repair entry を `success/failed/skipped` へロールフォワード更新する `finalize_pending_repair_entries` を追加し、同一commandの重複追記を抑えつつ execution log を時系列で一貫管理できるようにした。  
    * pending entry の TTL 失効ガード（`expire_pending_repair_entries` + `--pending-ttl-seconds`）を追加し、長時間放置タスクを `timeout` へ自動遷移させて iterative repair loop の停滞を可視化できるようにした。  
    * `build_retry_queue_from_repair_log` と `--retry-max-attempts` を追加し、`failed/timeout` の最新状態を持つ command を再試行対象として抽出し、Gate summary で retry queue（attempt進捗付き）を監視できるようにした。  
    * `dispatch_retry_queue_to_pending` と `--auto-dispatch-retry` を追加し、retry queue の先頭N件を pending repair entry として自動投入し、再試行キューの運用を半自動で前進できるようにした。  
    * release soak report に `repair_auto_dispatch`（requested/dispatched）を追加し、summary の Gate セクションで auto-dispatch 実行結果を可視化できるようにした。  
    * `--retry-cooldown-seconds` と cooldown-blocked queue（`repair_retry_cooldown_blocked`）を追加し、直近失敗コマンドの過剰再投入を抑制しつつ、再試行待ちの残時間を Gate summary で追跡できるようにした。  
    * `prioritize_retry_queue` を追加し、`reason`（timeout/failed）・`remaining_checks` との重なり・attempt pressure を合成した priority score/tier で retry queue を並び替え、auto-dispatch が高価値な再試行から進むようにした。  
    * `dispatch_retry_queue_to_pending_with_report` を追加し、auto-dispatch の投入済み command と非投入理由（pending重複 / dispatch上限）を `repair_auto_dispatch` に保持して、Gate summary で運用判断に使える監査情報を残せるようにした。  
    * `--auto-dispatch-min-priority` と `select_retry_dispatch_batch` を追加し、priority tier（high/medium/low）の下限を指定して再試行投入を制御できるようにした。summary には eligible/selected/low-priority skip を表示し、dispatch方針の可視性を強化した。  
    * `--auto-dispatch-diversify-checks` を追加し、dispatch budget 内で covered check の重複を抑える greedy 多様化選択を導入した。`selected_unique_check_count` と `selection_mode` を summary へ表示し、再試行の探索効率を追跡できるようにした。  
    * `--auto-dispatch-max-per-check` を追加し、同一checkへの再試行集中を抑える quota 制御を導入した。summary に `skipped_check_quota` と command 明細を表示して、投入見送り理由を監査できるようにした。  
* **Stage C: Meta-Adaptation Experiments**
  * session memory や routing diagnostics を使い、短い対話で応答方針が改善される lightweight adaptation loop を追加する。  
  * readout, routing threshold, memory weighting, fallback gate などを、固定値ではなく task-conditioned に調整する。  
  * **現在の進捗:**  
    * `task_switch_adaptation_benchmark` に `meta_adaptation_parameter_integrity` を追加し、`response_mode/planning_confidence/memory_weight/fallback_relaxation` が整合した適応状態へ収束できるかを継続計測できるようにした。  
    * `phase3_accuracy_suite` の `adaptive_readiness` に同メトリクスを統合し、Stage C 系の品質を focus summary / trend で追跡できるようにした。  
    * phase3 accuracy summary / release soak summary に `adaptation_parameter_integrity` と trend 行を追加し、Stage C の適応品質を release gate 運用中にも直接監視できるようにした。  
    * release gate の recovery planner を Stage C に拡張し、`adaptive_readiness` / `meta_adaptation_parameter_integrity` 失敗時は `task_switch_adaptation_benchmark` の再実行を優先提案できるようにした。  
    * `stage_c_contract` と `stage_c_readiness` を追加し、`meta_adaptation_loop` / `meta_adaptation_parameter_integrity` を Stage C minimum check として機械可読に保持できるようにした。release gate でも Stage C minimum を必須検証し、未達時は value/threshold 付きエラーを返すようにした。  
    * `自己蒸留の効果と課題` の方針を反映し、`task_switch_adaptation_benchmark` に `temporal_self_distillation_stability` を導入した。時間差自己蒸留（teacher state → paraphrase応答後 student state）で適応状態ドリフトが過大化しないことを Stage C 指標として継続計測する。  
    * Stage C minimum gate を拡張し、`temporal_self_distillation_stability` を `stage_c_contract` の必須チェックへ昇格した。`phase3_accuracy_suite` / `release_gate` / `repair plan` / summary が同指標の未達を一貫して検知・復旧提案できるようにした。  
    * release soak の gate feedback / summary に Stage C フィールド（`stage_c_passed`、minimum failure count/details）を追加し、release運用中に Stage C minimum の失敗理由を直接追跡できるようにした。  
* **Stage D: Continual Consolidation**
  * replay data, memory upgrade, health diagnostics, release soak を接続し、継続学習の品質低下を継続監視できるようにする。  
  * active material curation と continual-learning replay をつなぎ、素材管理から学習ループまで監査可能にする。  
  * waking-time の実経験だけでなく、idle time / offline replay を利用した counterfactual consolidation を導入し、現実経験から派生した複数プランの仮想結果を比較学習できるようにする。  
  * **SNN構造的可塑性と知識転送まとめから採用する方針（段階導入）:**  
    * CNAF / AG-MSP / SGW-TAMC を巨大な一体アーキテクチャとして丸ごと採用するのではなく、既存の SNN 主経路に合う小さな runtime primitive と benchmark に分解して導入する。  
    * 第一段階として、スパイク統計に基づく gradient-free な `synaptic_tag` / `importance_score` を追加し、重要な結合を replay / pruning / consolidation の判断材料にする。候補指標は ISI-CV 由来の発火間隔ばらつき、pre/post spike correlation、weight persistence、recent replay usefulness とする。  
    * DONE: `src/sara_engine/learning/synaptic_tag.py` を追加し、local spike trace から `consolidate` / `replay` / `watch` / `prune` tag、`importance_score`、`replay_priority`、`pruning_candidate` を gradient-free に算出できるようにした。`continual_consolidation_benchmark` には observed-only の `synaptic_tag_integrity_observed`、`synaptic_tag_importance_score_observed`、`synaptic_tag_replay_priority_observed`、`synaptic_tag_pruning_candidate_observed`、`synaptic_tag_state_budget_observed` を追加し、Phase3 summary でも Stage D consolidation 観測値として表示する。現段階では release gate 必須条件へ昇格せず、第二段階の memory phase / 第三段階の metabolic budget へ渡す判断材料として扱う。  
    * 第二段階として、記憶状態を `liquid`（即時適応）、`glass`（一時保護）、`crystal`（長期固定）の三相で表現し、短期記憶から長期記憶へ移す際の可塑性・保護・固定化を明示する。  
    * DONE: `src/sara_engine/learning/memory_phase.py` を追加し、local consolidation signal（stability / replay success / interference）から `liquid -> glass -> crystal` の phase path、plasticity、retention を gradient-free に観測できるようにした。`continual_consolidation_benchmark` には observed-only の `memory_phase_transition_integrity_observed`、`memory_phase_retention_protection_observed`、`memory_phase_plasticity_guard_observed`、`memory_phase_overfixation_guard_observed`、`memory_phase_state_budget_observed` を追加し、Phase3 summary でも Stage D consolidation 観測値として表示する。現段階では重要記憶の保護と noisy/fresh context の過剰固定化抑止を観測する段階に留め、release gate 必須条件への昇格は metabolic budget / sleep consolidation 接続後に判断する。  
    * 第三段階として、structural plasticity に `metabolic_budget` / `plasticity_reserve` を導入し、シナプス生成・再配線・刈り込みが無制限に増えないようにする。release summary では resource pressure と pruning reason を監査可能にする。  
    * DONE: `src/sara_engine/learning/metabolic_budget.py` を追加し、structural grow / rewire / prune 候補を `max_synapses`、`event_budget`、`plasticity_reserve`、重要度、resource pressure で bounded に評価できるようにした。`continual_consolidation_benchmark` には observed-only の `metabolic_budget_integrity_observed`、`plasticity_reserve_integrity_observed`、`structural_growth_bounded_observed`、`pruning_reason_trace_observed`、`resource_pressure_observed` を追加し、Phase3 summary でも Stage D consolidation 観測値として表示する。これにより、低重要度の成長を高圧下で拒否し、刈り込み理由を trace として残す最低限の監査導線が入った。  
    * 第四段階として、`sleep_consolidation` / `latent_replay` を Stage D benchmark に追加し、idle/offline replay 中に noise resilience、retention、memory health、energy cost が改善するかを観測する。  
    * DONE: `src/sara_engine/learning/sleep_consolidation.py` を追加し、offline replay trace から retention delta、noise delta、memory health delta、latent branch selection、total event cost を gradient-free に観測できるようにした。`continual_consolidation_benchmark` には observed-only の `sleep_consolidation_retention_observed`、`latent_replay_noise_resilience_observed`、`sleep_consolidation_memory_health_observed`、`latent_replay_counterfactual_branch_observed`、`sleep_consolidation_energy_budget_observed` を追加し、Phase3 summary でも Stage D consolidation 観測値として表示する。現段階では idle/offline replay が記憶品質を悪化させず event budget 内に収まることの監査に留め、次段の astro/world-model 接続で unlock/lock policy と統合する。  
    * 第五段階として、既存の `astro_modulator` と world-model replay を接続し、予測誤差が高い期間だけ構造可塑性をアンロックし、安定後は構造変更をロックして bounded STDP のみを許可する。  
    * DONE: `src/sara_engine/learning/astro_structural_gate.py` を追加し、`AstroReplayModulator` の slow-timescale state と world-model replay trace（prediction error / recovery event）から structural plasticity の `unlock_structural_plasticity` / `lock_to_bounded_stdp` / `bounded_stdp_only` policy を observed-only で監査できるようにした。`continual_consolidation_benchmark` には `astro_structural_unlock_observed`、`astro_structural_lock_observed`、`astro_bounded_stdp_fallback_observed`、`world_model_replay_policy_trace_observed`、`astro_policy_state_budget_observed` を追加し、Phase3 summary でも Stage D consolidation 観測値として表示する。これで Stage D の構造可塑性段階導入は、重要度タグ・三相記憶・代謝予算・sleep replay・astro unlock/lock policy まで一通り観測可能になった。  
    * DONE: Stage D の新規 observed metrics（synaptic tag / memory phase / metabolic budget / sleep consolidation / astro structural gate）を release soak gate feedback と operational readiness の `stage_d_readiness` snapshot / summary に伝播し、Phase3 summary だけでなく運用レポート上でも resource pressure、pruning reason、sleep replay、unlock/lock policy を監査できるようにした。  
  * **δ-mem由来の online associative memory 方針（段階導入）:**  
    * Stage D では、全履歴を replay buffer や prompt text として保持するのではなく、短期履歴を小さな `delta_associative_state` に圧縮し、必要な時だけ memory controller へ steering event を返す仕組みを追加候補にする。  
    * delta update は `predicted_value` と `observed_value` の残差だけを書き込み、既に予測できている経験を繰り返し強化しない。これにより replay energy cost と干渉を抑える。  
    * retention/forget gate は `astro_modulator` / memory phase（liquid/glass/crystal）と接続し、安定記憶は保持、揮発的な文脈は徐々に減衰させる。  
    * DONE: `src/sara_engine/learning/delta_retention_policy.py` を追加し、memory phase と astro stability を使って δ-mem の retention / forget gate を observed-only で評価できるようにした。crystal 記憶は保持し、liquid 文脈は低保持へ倒し、astro が不安定な時は構造保持を過剰に固定しない。`continual_consolidation_benchmark` には `delta_memory_phase_retention_policy_observed`、`delta_memory_crystal_retention_observed`、`delta_memory_liquid_forget_observed`、`delta_memory_astro_gate_alignment_observed`、`delta_memory_policy_state_budget_observed` を追加し、Phase3 summary / release soak / operational readiness へ伝播した。現段階では Stage D minimum へは昇格せず、履歴圧縮が replay recovery、noise resilience、memory health、manifold recall を悪化させないことを追加確認するための監査指標として扱う。  
    * DONE: δ-mem retention policy に multi-history stress 評価を追加し、複数の crystal/glass/liquid 履歴を同じ bounded associative state へ流した後でも、文脈別 recall、liquid noise resilience、state health、cross-branch leak guard が維持されるかを observed-only で確認できるようにした。`delta_memory_multi_history_recall_observed`、`delta_memory_multi_history_noise_resilience_observed`、`delta_memory_multi_history_health_observed`、`delta_memory_multi_history_manifold_guard_observed` を Phase3 summary / release soak / operational readiness へ伝播し、Stage D minimum 昇格前の複数履歴監査を一段具体化した。  
    * DONE: Gated DeltaNet-2（arXiv:2605.22791）の erase/write 分離を参考に、δ-mem retention policy の次段として `delta_memory_erase_write_decoupling_observed`、`delta_memory_erase_preserves_stable_memory_observed`、`delta_memory_write_commits_residual_observed` を observed-only 候補に追加した。実装は dense linear attention ではなく、memory phase / astro stability / residual magnitude に基づく bounded event gate とし、既存記憶の保護と新規 residual commit を別々に監査する。  
    * DONE: δ-mem 系 observed metrics を Stage D minimum へ即時昇格せず、`delta_memory_candidate_ready` / `delta_memory_promotion_readiness` として連続達成数、required streak、promotion recommended を追跡できるようにした。Phase3 summary / release soak gate feedback / operational readiness summary で候補状態と次アクションを確認でき、十分な履歴が貯まった後に `stage_d_contract` へ昇格するかを判断する。  
    * DONE: `delta_memory_promotion_readiness.recommended=true` の場合は operational readiness の `recovery_actions` に `stage_d_delta_memory_promotion` を追加し、`release_soak.py --record-repair-source stage_d_delta_memory_promotion` で契約昇格作業を repair log へ pending 登録できるようにした。Stage B promotion と同じ運用導線で、昇格判断と手作業の追跡を一貫して扱える。  
    * DONE: δ-mem promotion candidate の各チェックに説明文と structured failure details（check / metric / description / value / threshold）を追加し、候補未達時にどの保持・忘却・複数履歴・manifold guard が不足しているかを Phase3 / release soak / operational readiness で追跡できるようにした。  
    * DONE: release soak summary と operational readiness summary に `stage_d_delta_memory_candidate_failure` 行を追加し、候補未達時の metric/value/threshold を人間向けレポートでも直接確認できるようにした。  
    * 初期実装は `continual_consolidation_benchmark` の observed-only ケースとして、短い履歴を消した後でも delta state から関連 action / target / correction event を復元できるかを確認する。  
    * Stage D minimum へ昇格するのは、履歴圧縮が replay recovery、noise resilience、memory health、manifold recall を悪化させないことを複数履歴で確認してからにする。  
  * **採用しない要素（本プロジェクト制約との不整合）:**  
    * backpropagation / surrogate-gradient / BPTT を前提にした重要度推定を continual consolidation の中核に置くこと。  
    * GPU 前提の大規模 sparse simulation や dense matrix replay を必須化すること。  
    * quantum SNN や Transformer attention 相当の高コストな global workspace を標準 runtime として導入すること。  
    * 構造可塑性を常時オンにし、release gate で resource budget / pruning trace / retention trace を監査できない状態にすること。  
    * δ-mem を理由に dense attention correction / LLM hidden-state adapter / gradient-trained projection を SARA runtime の必須経路へ入れること。  
    * Gated DeltaNet-2 を理由に、chunkwise dense training、gate-aware backward pass、線形 attention kernel を SARA runtime の必須経路へ入れること。採用するのは erase/write 分離の設計原理と評価観点に限定する。  
  * **Stage D acceptance criteria への追加候補:**  
    * `synaptic_tag_integrity`: 重要度タグが replay 後も安定し、低重要度結合が優先的に pruning されること。  
    * `memory_phase_transition_integrity`: `liquid -> glass -> crystal` の遷移が過剰固定化や過剰忘却を起こさないこと。  
    * `metabolic_budget_integrity`: synapse count / active route / event cost が上限内に収まり、resource pressure が summary に出ること。  
    * `sleep_consolidation_retention`: sleep/latent replay 後に retention と noise resilience が悪化せず、energy-per-success が閾値を超えないこと。  
    * DONE: 上記に δ-mem retention / multi-history stress 系を加え、`STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES` / `STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS` として `stage_d_contract` に機械可読化した。Phase3 の `stage_d_readiness` では `acceptance_candidates`、`acceptance_candidate_count`、`acceptance_candidate_ready_count` を出し、minimum 昇格前でもどの候補が達成済みかを監査できる。  
    * DONE: Stage D acceptance candidate の候補数・達成数を release soak gate feedback と operational readiness snapshot / summary に伝播し、Phase3を開かなくても運用レポート上で minimum 昇格候補の準備状況を確認できるようにした。  
    * DONE: Stage D acceptance candidates 全体の `acceptance_candidates_ready` / `acceptance_candidate_failure_count` を追加し、Phase3 / release soak / operational readiness の全層で「minimum 昇格候補が全て達成済みか」を単一フィールドで判定できるようにした。これで Stage D 候補管理は、列挙・達成数・未達詳細・ready判定・promotion action まで一通り閉じた。  
    * DONE: Stage D acceptance candidates 全体に `acceptance_candidate_stability`（`consecutive_passes` / `required_streak` / `recommended`）を追加し、単発の ready ではなく履歴ベースの連続達成で minimum 昇格候補の安定性を判定できるようにした。Phase3 summary / release soak gate feedback / operational readiness summary にも `stage_d_acceptance_candidate_consecutive_passes`、`stage_d_acceptance_candidate_required_streak`、`stage_d_acceptance_candidate_stability_recommended` を出し、δ-mem 個別候補と Stage D 候補群全体の両方を同じ運用導線で確認できる。  
    * DONE: `acceptance_candidate_stability.recommended=true` の場合に、release soak gate feedback へ `stage_d_acceptance_candidate_next_step_hint` / `stage_d_acceptance_candidate_actions` を出し、operational readiness の `recovery_actions` に `stage_d_acceptance_candidate_stability` を pending 登録できる導線を追加した。これで Stage D 候補群全体の安定達成から、minimum 昇格範囲レビューまでを repair log 上で追跡できる。  
    * DONE: release soak summary / operational readiness summary に `stage_d_acceptance_candidate_action` 行を追加し、Stage D 候補群全体の安定達成後に必要なレビュー作業を人間向けレポート上でも直接確認できるようにした。  
    * DONE: Stage D acceptance candidate stability の next step / action 文言を `stage_d_contract` の共通定数へ集約し、release soak と operational readiness の間で repair 導線の文言がずれないようにした。  
    * DONE: `tests/test_stage_d_contract.py` を追加し、Stage D acceptance candidate / minimum check / stability action 契約が、重複なし・説明文あり・非空 action で保たれることを単体で固定した。  
    * DONE: release soak gate feedback / summary と operational readiness snapshot / summary に `stage_d_acceptance_candidate_action_count` を追加し、Stage D 候補群全体の安定達成後に必要なレビュー作業数を機械的に拾えるようにした。  
    * DONE: Stage D acceptance candidate 全体にも structured failure details を追加し、`stage_d_acceptance_candidate_failure` として Phase3 / release soak / operational readiness summary へ表示できるようにした。δ-mem 個別候補以外の未達や、erase/write 分離候補の未達も value/threshold 付きで追跡できる。  
    * DONE: Stage D candidate failure summary に `description` を追加し、metric 名だけでなく「どの性質が未達か」を Phase3 / release soak / operational readiness の人間向けレポートから直接読めるようにした。  
    * DONE: release soak / operational readiness では古い snapshot に `description` が無い場合でも、`stage_d_contract` の check description から `stage_d_acceptance_candidate_failure` / `stage_d_delta_memory_candidate_failure` の説明文を補完できるようにした。  
    * DONE: Stage D acceptance candidate failure が残る場合、operational readiness の `recovery_actions` に `stage_d_acceptance_candidate_repair` を追加し、失敗 metric 名付きで release repair log へ pending 登録できるようにした。これで acceptance candidate の未達検知から修復作業の追跡までが閉じた。  
  * **導入予定（Astrocyte-inspired modulation）:**  
    * ANN/astrocyte の全面置換ではなく、SNN主経路に対する lightweight な `astro_modulator`（遅い時間スケールの閾値・可塑性ゲート調整）として限定導入する。  
    * 適用対象は Stage D の replay/consolidation loop を優先し、過活動抑制・忘却抑制・安定化の補助信号として使う。  
    * 初期評価は `continual_consolidation_benchmark` と `consolidation_readiness` に追加し、`catastrophic drift` 抑制と retention 改善を release gate で追跡できる形にする。  
    * backprop 依存・dense matrix 依存・GPU 前提の astrocyte 実装は採用しない。  
  * **現在の進捗:**  
    * `continual_consolidation_benchmark` を追加し、`replay_recovery_integrity`、`long_horizon_consolidation_retention`、`counterfactual_replay_selection_integrity` を CPU-only で継続計測できるようにした。  
    * 同 benchmark に `replay -> upgrade-memory -> inspect-memory` の実行ケースを追加し、`replay_upgrade_reindex_integrity` と `memory_health_index_integrity` を計測可能にした。replay data / memory upgrade / health diagnostics の導線を Stage D 評価へ直接接続した。  
    * `phase3_accuracy_suite` に `continual_consolidation` component と `consolidation_readiness` focus を統合し、summary/trend で replay 固定化品質を追跡できるようにした。  
  * `astro_modulator`（astrocyte-inspired slow-timescale modulation）を Stage D benchmark に限定統合し、`astro_modulation_stability` を継続観測できるようにした。SNN主経路を置換せず、replay干渉時の保持安定化を補助する形で段階導入した。  
  * `replay_noise_resilience_integrity` を Stage D の minimum gate に昇格し、`continual_consolidation_benchmark` / `phase3_accuracy_suite` / `release_gate` / `release_soak` / operational readiness で、ノイズ混入 replay による記憶崩壊リスクを出荷前に必ず検知できるようにした。  
  * `operational_readiness` の top-level snapshot に `stage_d_readiness` を追加し、Stage D minimum metrics（replay recovery / long-horizon retention / counterfactual replay / reindex / memory health / replay noise resilience / astro modulation）を最上位運用レポートから直接監査できるようにした。  
  * `release_soak` summary の Phase 3 Focus に `consolidation_readiness` 詳細（replay recovery / reindex / memory health / replay noise resilience / astro modulation と trend）を追加し、Stage D の観測値を release 運用レポートで直接レビューできるようにした。  
  * `release_soak` の `collect_release_gate_feedback` に Stage D 観測メトリクス（readiness score と主要 integrity 指標）を機械可読フィールドとして追加し、運用自動化側からも Stage D の状態を直接参照できるようにした。  
  * `release_gate` artifact に `error_details`（minimum threshold failure / metric threshold drop の構造化抽出）を追加し、Stage D を含む gate 失敗理由を文字列解析なしで機械処理できるようにした。  
  * `release_soak` summary でも `release_gate.error_details` を表示し、Stage D を含む失敗理由（type/category/metric/value/required）を人間レビューと運用自動化で同時に追跡できるようにした。  
  * `release_gate` / `release_soak` に `error_details_summary`（type/category/metric の件数集計）を追加し、失敗傾向の俯瞰（例: Stage D由来の失敗がどの型で増えているか）を軽量に監視できるようにした。  
  * `release_soak` の `error_details_summary` 集計を `release_gate` の共通実装へ統合し、gate artifact と soak summary 間で集計ロジックの一貫性を保てるようにした。  
  * `error_details_summary` に `top_types` / `top_categories` / `top_metrics` を追加し、失敗傾向の上位要因を運用レポートでそのまま表示できるようにした。  
  * `release_soak` Gate summary に `error_detail_category_count` を追加し、Stage D を含む失敗カテゴリ（check系統）の偏りを人間レビューで即把握できるようにした。  
  * `release_gate` / `release_soak` に `failure_focus`（primary/secondary category・primary metric・primary action）を追加し、復旧時に「最初に潰すべき失敗軸」を即時提示できるようにした。  
  * `failure_focus` に `confidence` を追加し、失敗カテゴリの集中度と即応アクション有無をもとに復旧優先軸の明確さを定量表示できるようにした。  
  * `release_soak` の gate feedback 生成を `collect_release_gate_artifacts` に統合し、recovery plan / error details / failure focus の計算経路を単一化して運用時のドリフトを抑制した。  
  * Stage D minimum gate を拡張し、`astro_modulation_stability` を `stage_d_contract` の必須チェックへ昇格した。`phase3_accuracy_suite` / `release_gate` / `release_soak` が同指標の未達を一貫して検知・監査できるようにした。  
    * Stage D minimum gate をさらに拡張し、`replay_noise_resilience_integrity` も `stage_d_contract` の必須チェックへ昇格した。ノイズ混入 replay で保持安定性が落ちた場合は Phase 3 completion / release gate / operational readiness が失敗する。  
    * `v1_release_gate` に `stage_d_consolidation_minimum` と `operational_stage_d_snapshot` を追加し、最終出荷判定でも Stage D の Phase 3 minimum と operational 伝播 snapshot が両方 PASS でなければ v1.0 を通さないようにした。  
    * Stage D minimum gate をさらに拡張し、`replay_upgrade_reindex_integrity` と `memory_health_index_integrity` も `stage_d_contract` の必須チェックへ昇格した。replay -> upgrade-memory -> inspect-memory の運用導線まで release gate の minimum 条件として常時検証できるようにした。  
    * `stage_d_contract` と `stage_d_readiness` を追加し、Stage D minimum を機械可読に保持できるようにした。release gate でも Stage D minimum を必須検証し、未達時は value/threshold 付きエラーで返すようにした。  
    * release gate の recovery planner / failed-check inference を Stage D に拡張し、`continual_consolidation_benchmark` の再実行提案と affected check 追跡を自動化した。  
    * release soak の gate feedback / summary に Stage D フィールド（`stage_d_passed`、minimum failure count/details）を追加し、運用時に Stage D 失敗理由を直接監査できるようにした。  
* **Stage E: Modular Cognitive Runtime**
  * planner, memory controller, world model, actor を明示的に分け、非同期に協調させる。  
  * 長期的には single-agent chat runtime から、cognitive module orchestration runtime へ進化させる。  
  * planner / world model 間に counterfactual simulation lane を追加し、「実行中の plan」と「未実行の代替 plan」を並列比較できる runtime へ発展させる。  
  * **方針強化（Brain-like distributed submodel architecture）:**  
    * 単一巨大モデルを拡張するより、小さな world model / memory / value / body control / language / math / self-monitor などの専門的・可塑的サブモデルを sparse event で動的に接続・切断・再学習する構造を優先する。  
    * 継続学習、局所学習、STDP / 可塑性、液体状態機械、因果モデル形成、自己組織化、低電力イベント駆動計算を Stage E-F の研究・実装ポリシーとして維持する。  
    * 解釈可能性を必須研究軸とし、内部概念、計画候補、価値判断、counterfactual lane、submodel route を人間が監査できる trace として残す。  
  * **導入予定（High-order reasoning / multimodal spike integration）:**  
    * `common_spike_space` を cognitive module 間の共通インターフェースとして定義し、memory / world model / planner が同じ sparse event を参照できるようにする。  
    * `modality_temporal_scale` を runtime policy に追加し、text / structured state / future sensor adapter ごとに bounded timestep budget を割り当てる。  
    * `dendritic_context_gate` を planner と world model の間に置き、短期文脈・長期文脈・prediction error を軽量に分離して保持する。  
    * `event_relation_trace` と `reverse_reasoning_trace` を action candidate の説明情報として保存し、順方向予測だけでなく「なぜその原因候補に戻ったか」を release summary で監査できるようにする。  
    * 初期評価は Stage E benchmark に限定し、`common_spike_space_integrity`、`temporal_compression_efficiency`、`dendritic_context_gate_stability`、`reverse_reasoning_trace_integrity` を観測枠として追加する。  
  * **δ-style memory steering の導入候補:**  
    * `ModularCognitiveRuntime` の memory controller に、text retrieval ではなく compact online state からの `memory_steering_event` を追加候補とする。  
    * world model / planner は、この steering event を action candidate の prior として使うが、選択理由には `residual_write_trace`、`retention_gate_trace`、`state_budget_trace` を残す。  
    * primary / counterfactual lane の両方で同じ delta state を読む場合、lane ごとの read-only snapshot を使い、counterfactual simulation が本番 state を汚染しないことを必須観測にする。  
    * Stage E の初期観測候補は `delta_memory_steering_integrity`、`delta_memory_counterfactual_isolation`、`delta_memory_trace_observability` とする。  
  * **Interaction Models から採用する方針（限定採用 / Stage E-F bridge）:**  
    * Thinking Machines Lab の Interaction Models で示された `time-aligned micro-turn`、foreground interaction model と asynchronous background model の分離、同時入力・同時出力・割り込み・沈黙・視覚変化を model context に含める方針は、SARA の身体性 / 継続学習 / 分散サブモデル統合に強く整合するため採用候補にする。  
    * ただし、巨大な full-duplex Transformer を中核にするのではなく、`micro_turn_event_stream` として 100-250ms 程度の小さな time bucket を sparse event 化し、foreground は低遅延の event router / safety gate / backchannel policy、background は `ModularCognitiveRuntime` / hypothesis bank / scientific model trace を担当する二層構造として設計する。  
    * foreground runtime は `audio_tick` / `visual_change` / `text_delta` / `silence` / `overlap` / `interrupt` / `tool_result_delta` を共通 spike event に変換し、background runtime はそれらをまとめて world model / memory / value / self-monitor / language / math サブモデルへ渡す。  
    * 初期観測候補は `micro_turn_event_budget_observed`、`foreground_background_context_handoff_observed`、`interrupt_recovery_trace_observed`、`simultaneous_stream_route_integrity_observed`、`time_aligned_backchannel_policy_observed` とし、最初は Stage E observed-only に留める。  
    * ユーザーが作業中に割り込む / 追加指示する / 視覚的変化が起きる状況でも、selected action、counterfactual lane、hypothesis bank、submodel credit assignment の trace が途切れないことを長期目標にする。  
  * **DiffusionBlocks から採用する方針（設計原理のみ / Stage E-D bridge）:**  
    * DiffusionBlocks の「residual update を dynamical process と見なし、block ごとに独立した役割を持たせる」発想は、SARA の専門サブモデル群、memory phase、prediction error、forward-only local update に有用な設計原理として採用候補にする。  
    * ただし、score matching / Transformer block-wise gradient training / GPU 前提の training recipe は採用しない。SARA では `noise_level`、`uncertainty`、`prediction_error`、`memory_phase` を sparse event の phase tag として扱い、各 phase に対応する専門サブモデルや local correction route を選ぶ。  
    * 初期 primitive 候補は `phase_assigned_submodel_block`、`uncertainty_bucket_route`、`denoising_like_correction_trace`、`block_independent_local_credit_trace` とする。これらは backprop ではなく、既存の `PlasticSubmodelRegistry.apply_route_credit`、`adapt_route_edges`、hypothesis bank、memory phase / astro gate と接続する。  
    * 初期観測候補は `phase_assigned_submodel_route_observed`、`uncertainty_bucket_specialization_observed`、`denoising_correction_trace_integrity_observed`、`block_independent_local_update_budget_observed` とし、Stage D consolidation / Stage E runtime の observed-only に流す。  
    * recurrent-depth / BPTT 削減の観点は参考にするが、実装は single-pass forward trace、局所 credit、反証可能 hypothesis revision として扱う。  
  * **LeJEPA から採用する方針（限定採用 / Stage E world-model latent health）:**  
    * LeJEPA の latent を線形に読み出せる状態へ保つ思想、positive pair の latent alignment、collapse 回避、factor disentanglement、latent 上での planning consistency は、SARA の Spiking H-JEPA / common spike space / causal branch runtime の健全性監査に有用なため採用する。  
    * ただし、dense Gaussian SIGReg、end-to-end backprop、GPU 前提の latent whitening、画像再構成や巨大 encoder training は採用しない。SARA では sparse event id、transition fingerprint、prediction error、correction coverage、primary / counterfactual branch separation から proxy を読む。  
    * 初期観測候補は `lejepa_linear_identifiability_proxy_observed`、`lejepa_latent_whitening_health_observed`、`lejepa_factor_disentanglement_observed`、`lejepa_latent_planning_consistency_observed`、`lejepa_positive_pair_alignment_observed` とし、Stage E observed-only に留める。  
    * 目的は「latent が賢そうに見える」ことではなく、専門サブモデル群が使う world-model state が collapse せず、反実仮想 lane と分離され、局所 correction によって計画へ使える形を保っているかを継続監査することに置く。  
  * **現在の進捗:**  
    * Stage E の初期観測枠として `cognitive_runtime_benchmark` を追加し、common spike space / temporal compression / modality budget / dendritic context gate / reverse reasoning trace の全項目が PASS する状態にした。  
    * `ModularCognitiveRuntime` を実装し、encoder -> memory controller -> world model -> planner -> actor の module order と、primary / counterfactual lane の分岐比較を sparse event trace として保存できるようにした。  
    * Stage E の追加観測枠として `module_orchestration_integrity`、`counterfactual_lane_integrity`、`action_trace_observability` を追加し、release readiness の summary / trend に流れる状態にした。  
    * 上記3指標を Stage E minimum gate へ昇格し、初期 cognitive module orchestration が壊れた場合は Phase 3 completion / release gate が失敗する状態にした。  
    * `release_soak` summary の Gate セクションを Stage E 対応に拡張し、出荷前レビューで modular cognitive runtime の readiness と minimum failure を Stage B-D と同じ粒度で確認できるようにした。  
    * `operational_readiness` summary に Stage E snapshot を追加し、v1/production promotion の最上位レポートからも Stage E の pass/fail、minimum failure count、module orchestration / counterfactual lane / action trace を直接確認できるようにした。  
    * `build_runtime_trace_digest` / `compare_runtime_trace_digests` を追加し、full event payload を保存し続けなくても Stage E runtime trace の再現性を監査できるようにした。`cognitive_runtime_benchmark`、Phase 3 summary、release soak summary、operational readiness summary に `runtime_trace_replay_consistency` を流す。  
    * δ-mem の `memory_steering_event` を observed-only trace として `cognitive_runtime_benchmark` に接続し、primary / counterfactual lane の steering id 分離、text reinjection 不使用、trace observability を `delta_memory_steering_integrity_observed` / `delta_memory_counterfactual_isolation_observed` / `delta_memory_trace_observability_observed` で確認できるようにした。  
    * δ-mem の observed-only 指標を Phase 3 summary、release soak gate feedback、operational readiness summary へ接続し、単発 benchmark だけでなく運用レポート上でも Stage E の compact memory steering 状態を監査できるようにした。  
    * Stage D の `delta_memory_residual_write_integrity_observed` / `delta_memory_retention_gate_stability_observed` / `delta_memory_state_budget_integrity_observed` と、Stage F の `edge_delta_state_*_observed` を Phase 3 summary に追加し、δ-mem のオンライン学習状態と edge 永続化状態を同時に確認できるようにした。  
    * `PlasticSubmodelRegistry` を追加し、world model / memory / value / body control / language / math / self-monitor の専門サブモデルが、bounded route edge、接続・切断 event、局所再学習 trace、解釈可能な concept trace を持つことを observed-only で検証できるようにした。  
    * `plastic_submodel_registry_integrity_observed`、`dynamic_submodel_route_integrity_observed`、`submodel_relearning_trace_integrity_observed`、`interpretable_submodel_concept_trace_observed` を `cognitive_runtime_benchmark` / Phase 3 summary / release soak / operational readiness に接続し、分散サブモデル統合の研究方針が退行した場合に運用レポートで検知できるようにした。  
    * `PlasticSubmodelRegistry` を `ModularCognitiveRuntime` の world-model action candidate 生成へ接続し、selected action / counterfactual action のそれぞれに `submodel_route` と `support_submodels` を残すようにした。これにより、planner が選んだ行動がどの専門サブモデル経路に支えられたかを action trace から監査できる。  
    * `runtime_submodel_route_action_grounding_observed`、`runtime_submodel_counterfactual_route_separation_observed`、`runtime_submodel_concept_trace_observed` を追加し、分散サブモデル統合が単独 benchmark だけでなく runtime の実行経路・反実仮想 lane・concept trace に実際に接続されているかを Phase 3 / release soak / operational readiness で観測できるようにした。  
    * `PlasticSubmodelRegistry` に `set_active` を追加し、特定の専門サブモデルを一時的に deactivate / activate する解釈可能性介入 trace を実装した。`memory_system` を ablation した時に route support から消え、再活性化で復帰することを observed-only で検証できる。  
    * `submodel_intervention_trace_integrity_observed`、`submodel_ablation_effect_observed`、`submodel_reactivation_recovery_observed` を追加し、内部概念・専門サブモデル経路を操作した時の影響を Phase 3 / release soak / operational readiness で監査できるようにした。これは Claude などの内部表現操作研究を SARA の sparse / local / interpretable runtime 方針へ翻訳する初期足場として扱う。  
    * `PlasticSubmodelRegistry.apply_route_credit` と `ModularCognitiveRuntime.run(..., action_feedback=...)` を追加し、selected / counterfactual branch の結果 credit を、その branch を支えた専門サブモデル route だけへ局所的に返せるようにした。これは backpropagation ではなく、route support に対する forward-only な local credit assignment として扱う。  
    * `submodel_credit_assignment_trace_integrity_observed`、`submodel_credit_selectivity_observed`、`submodel_credit_state_budget_observed`、`runtime_submodel_local_credit_assignment_observed`、`runtime_submodel_feedback_trace_observed` を追加し、行動結果が bounded state 内で専門サブモデルの局所再学習 trace に反映されるかを Phase 3 / release soak / operational readiness で観測できるようにした。  
    * `PlasticSubmodelRegistry.adapt_route_edges` を追加し、成功した専門サブモデル route には bounded な support edge を追加し、失敗した route からは弱い edge を1本だけ pruning する自己組織化 primitive を実装した。構造変更は `route_edge_adaptation` trace として残し、edge budget を超えないことを必須観測にする。  
    * `submodel_structural_adaptation_trace_integrity_observed`、`submodel_structural_growth_bounded_observed`、`submodel_structural_pruning_observed` を追加し、open-ended な構造変化を無制限に許すのではなく、低電力・イベント駆動・解釈可能な範囲で成長と刈り込みを監査できるようにした。  
    * `evaluate_plastic_submodel_scientific_model_trace` を追加し、専門サブモデル route から `hypothesis -> prediction -> counterexample -> revised hypothesis` の小さな科学的モデル形成 trace を作れるようにした。反証が入ると confidence を下げ、guard condition を追加し、必要なら route edge adaptation で過信した経路を弱める。  
    * `submodel_scientific_hypothesis_trace_integrity_observed`、`submodel_counterexample_revision_observed`、`submodel_scientific_model_budget_observed` を追加し、因果モデル形成・反証・修正が bounded state と trace の中で維持されるかを Phase 3 / release soak / operational readiness で観測できるようにした。  
    * `evaluate_plastic_submodel_open_ended_hypothesis_bank_trace` を追加し、複数の専門サブモデル由来 hypothesis を confidence / novelty / counterexample の観点で bounded bank に保持・選別・prune できるようにした。保持された仮説には local credit を返し、prune された仮説は route edge adaptation で弱める。  
    * `submodel_hypothesis_bank_integrity_observed`、`submodel_open_ended_selection_observed`、`submodel_hypothesis_bank_budget_observed` を追加し、自己組織化・オープンエンド進化を無制限な増殖ではなく、反証可能で予算管理された hypothesis ecology として監査できるようにした。  
    * `evaluate_lejepa_sparse_latent_health_trace` を追加し、LeJEPA 由来の latent health を dense regularizer ではなく Spiking H-JEPA の sparse transition trace から監査できるようにした。linear identifiability proxy、latent whitening health、factor disentanglement、latent planning consistency、positive pair alignment は `cognitive_runtime_benchmark` / Phase 3 summary / release soak / operational readiness へ observed-only として流す。  
    * `evaluate_micro_turn_interaction_trace` を追加し、Interaction Models 由来の micro-turn / foreground-background handoff / interrupt recovery / simultaneous stream routing / time-aligned backchannel を、巨大 full-duplex model ではなく bounded sparse interaction trace として監査できるようにした。`micro_turn_event_budget_observed`、`foreground_background_context_handoff_observed`、`interrupt_recovery_trace_observed`、`simultaneous_stream_route_integrity_observed`、`time_aligned_backchannel_policy_observed` は `cognitive_runtime_benchmark` / Phase 3 summary / release soak / operational readiness へ observed-only として流す。  
    * `evaluate_phase_assigned_submodel_block_trace` を追加し、DiffusionBlocks 由来の phase / uncertainty ごとの専門ブロック化を、score matching や block-wise backprop ではなく sparse phase tag、uncertainty bucket、correction event、independent local update として監査できるようにした。`phase_assigned_submodel_route_observed`、`uncertainty_bucket_specialization_observed`、`denoising_correction_trace_integrity_observed`、`block_independent_local_update_budget_observed` は `cognitive_runtime_benchmark` / Phase 3 summary / release soak / operational readiness へ observed-only として流す。  
    * `research_automation_benchmark` に `stage_e_architecture_integration` signal を追加し、Interaction Models / DiffusionBlocks 由来の observed-only 指標が退行した場合に `stage_e_architecture_integration_metric_recovery` として next hypothesis / negative result / roadmap patch suggestion に回せるようにした。これにより、新規アーキテクチャ導入が単発の summary 表示に留まらず、研究自動化の安定・退行・再測定ループに入った。  
    * `stage_e_architecture_integration_observed_trend` を Phase 3 report に追加し、前回 run と比較して micro-turn / phase-block 指標の退行を `regressions` / `regression_count` として検出できるようにした。この trend は release soak / operational readiness summary にも伝播し、退行があっても observed-only / release-gate 非ブロッキングとして扱いつつ、research automation の `stage_e_architecture_integration_observed_regression` watchlist に回せる。  
    * operational readiness の runbook action 生成に `stage_e_architecture_integration_observed_trend` を接続し、履歴未取得または退行検出時に `observed_trend_long_run_validation` の affected checks へ自動追加できるようにした。これにより、Stage E の新規アーキテクチャ群は Phase3 での退行検知、research automation の仮説化、operational runbook の再測定アクションまで一続きで扱える。  
  * **採用しない要素（本プロジェクト制約との不整合）:**  
    * dense Transformer / MLLM を中核 runtime として導入すること。  
    * cross-attention を dense matrix 計算として標準経路に置くこと。  
    * GPU 前提の multimodal training や大規模 RL/CoT pipeline を release gate の必須要件にすること。  
    * 高次推論をブラックボックス化し、event relation / causal candidate / reverse reasoning の trace なしで action を選ぶこと。  
    * Interaction Models を理由に、リアルタイム full-duplex Transformer / 常時音声動画学習 / GPU resident streaming session を SARA の必須中核にすること。採用するのは micro-turn、foreground/background 分離、同時 stream を event として扱う設計原理に限定する。  
    * DiffusionBlocks を理由に、score matching、block-wise backprop、Transformer residual block training を SARA runtime の必須経路へ入れること。採用するのは phase / uncertainty ごとの専門ブロック化、独立更新、dynamical update 解釈に限定する。  
    * LeJEPA を理由に、dense SIGReg / global whitening loss / backprop encoder training を SARA の標準学習経路へ入れること。採用するのは sparse latent health の観測 proxy と world-model trace の健全性監査に限定する。  
* **Stage F: Hardware-Aware Optimization**
  * Rust core と serialization を、future neuromorphic execution を見据えたデータ構造へ最適化する。  
  * multilevel weights, event compression, sparse routing tables, low-precision state persistence を優先対象にする。  
  * **Neuromorphic backend 方針（Loihi 2 / Lava、SpiNNaker 2、BrainChip Akida など）:**  
    * 特定チップ専用実装へ早期に寄せず、まず SARA 内部の sparse event / spike / delay / routing / low-precision state を表す `spike_event_ir` を定義する。  
    * `spike_event_ir` は event id、timestep、channel、weight、delay、routing hint、state budget、online update policy を持つ lightweight 中間表現とし、dense matrix や GPU 前提の projection を必須にしない。  
    * edge payload の `format_capabilities` / `edge_manifest` に `neuromorphic_capabilities` を追加し、event routing、delay support、low precision weights、state persistence、online update support、backend compatibility を検証対象にする。  
    * Lava / SpiNNaker / Akida へは直接 runtime を依存させず、まず `lava_profile` / `spinnaker_profile` / `akida_profile` のような exporter profile と compatibility report を出す。  
    * 初期 benchmark 候補は `neuromorphic_ir_schema_integrity_observed`、`neuromorphic_capability_manifest_integrity_observed`、`neuromorphic_backend_profile_compatibility_observed`、`neuromorphic_sparse_event_budget_observed` とし、実機なしでも CI で変換可能性を確認する。  
    * 実機 adapter は Lava を第一候補にする。Python研究環境と接続しやすく、Loihi 2 向けの検証導線を作りやすいため。ただし SARA 本体は Lava 専用APIに依存しない。  
    * SpiNNaker 2 / Akida は、同じ `spike_event_ir` から profile 変換できるかを先に検証し、チップ固有の制約は adapter 層で吸収する。  
    * 目標は「特定neuromorphic chipでだけ動く実装」ではなく、「SARAの省電力・疎イベント・継続学習モデルが複数のneuromorphic backendへ移植可能であることを manifest と benchmark で示す」こととする。  
  * **δ-style online state persistence 方針:**  
    * δ-mem 由来の online associative state は edge payload 上で `delta_associative_state` として保存候補にし、`format_capabilities` には `delta_associative_memory_state` を追加する方針とする。  
    * state は full precision matrix ではなく、既存の `compact_int` / `multilevel_weight_profile` / `active_row_readout_storage` と同じ低精度・疎保存方針に揃える。  
    * `edge_manifest` digest と `validate_edge_model_file` の対象に delta state profile を含め、状態破損・未知 capability・budget 超過を strict validation で検出できるようにする。  
    * 初期 benchmark 候補は `edge_delta_state_persistence_observed`、`edge_delta_state_budget_observed`、`edge_delta_state_manifest_integrity_observed` とし、まず observed-only に留める。  
  * **現在の進捗:**  
    * edge exporter に `compact_quantized` を追加し、readout synapse を float JSON ではなく compact int weight + per-row range として永続化できるようにした。既存の float 形式も互換維持する。  
    * edge runtime が compact int 形式をロード時に復元できるようにし、低精度永続化後も既存の sparse event readout 経路で推論できる状態にした。  
    * `sparse_routing_table` を export payload に追加し、active row と row count を保存して neuromorphic routing / Rust core 側の疎ルーティング最適化へ接続しやすくした。  
    * edge exporter に `compress_events` を追加し、active row event を delta encoding で保存できるようにした。edge runtime は `active_row_deltas` を `active_rows` へ復元し、既存の推論経路と benchmark 観測を維持する。  
    * edge exporter に `sparse_readout` を追加し、空の readout row を payload から省いて active row だけを保存できるようにした。edge runtime は `readout_storage="active_rows"` を検出して従来の row index 空間へ復元するため、既存推論互換を維持したまま serialization を小型化できる。  
    * `edge_storage_profile` を export payload に追加し、row count / stored row count / empty row count / row reduction ratio / compact weight count を benchmark から観測できるようにした。Stage F の serialization 改善を主観的な軽量化ではなく、削減率つきで追跡する。  
    * `multilevel_weight_profile` を export payload に追加し、quantization bits / levels / quantized row count / quantized weight count / flat row count を runtime と benchmark から監査できるようにした。multilevel weights を単なる保存形式ではなく、Stage F の観測可能な低精度重み契約として扱う。  
    * edge payload に `format_version` と `format_capabilities` を追加し、runtime 側で対応 capability と未知 capability を監査できるようにした。`edge_format_compatibility_observed` を benchmark に追加し、将来の Rust core / neuromorphic backend 向け形式拡張が互換性を壊していないかを継続観測する。  
    * edge payload に `edge_manifest` を追加し、readout / routing / compression / profile 群の stable digest を runtime で再計算できるようにした。`edge_manifest_integrity_observed` を benchmark に流し、低精度・圧縮済み payload の欠損や破損を軽量に検知する。  
    * edge runtime に `strict_format` を追加し、未知 capability または manifest mismatch を検出した場合にロードを拒否できるようにした。`edge_strict_format_validation_observed` を benchmark に追加し、通常運用の互換ロードと CI/出荷前の厳格ロードを使い分けられるようにした。  
    * `validate_edge_model_file` を追加し、runtime を通常起動しなくても format compatibility / manifest integrity / row reduction / multilevel weight profile を report として取得できるようにした。`edge_payload_validation_report_observed` を benchmark に追加し、CI・Rust core・neuromorphic backend 側から同じ検証結果を参照できる足場にした。  
    * `delta_associative_state` を edge payload / `format_capabilities` / manifest digest / runtime strict validation の対象に追加し、`edge_delta_state_persistence_observed`、`edge_delta_state_budget_observed`、`edge_delta_state_manifest_integrity_observed` を energy benchmark から観測できるようにした。  
    * edge exporter に `neuromorphic_profile` を追加し、指定時に `spike_event_ir` と `neuromorphic_capabilities` を payload / `format_capabilities` / manifest digest / strict validation の対象へ含められるようにした。初期 profile は `lava` を fixture として扱い、SARA 本体は Lava 専用APIへ依存しない。  
    * energy benchmark に `neuromorphic_ir_schema_integrity_observed`、`neuromorphic_capability_manifest_integrity_observed`、`neuromorphic_backend_profile_compatibility_observed`、`neuromorphic_sparse_event_budget_observed` を追加し、実機なしでも neuromorphic backend 変換可能性を継続観測できるようにした。  
    * `neuromorphic_profile_report` を payload / manifest / strict validation に追加し、`lava_profile` / `spinnaker_profile` / `akida_profile` の adapter、event budget、delay support、low precision weight、online update policy を実機なしで確認できるようにした。Akida のように online update を直接扱わない profile は `freeze_state_for_inference_profile` として adapter 層で差分吸収する。  
    * energy benchmark に `neuromorphic_profile_report_integrity_observed` を追加し、複数profileの compatibility report が欠けたり互換性を失った場合に summary 上で検出できるようにした。  
    * `src/sara_engine/edge/neuromorphic.py` を追加し、profile 正規化、`spike_event_ir`、`neuromorphic_capabilities`、`neuromorphic_profile_report` の生成を exporter から分離した。将来の Lava / SpiNNaker / Akida 実 adapter はこの module 境界へ追加し、edge exporter は chip-neutral payload assembly に集中させる。  
    * energy benchmark に `neuromorphic_profile_trend` と `neuromorphic_profile_history_regression_observed` を追加し、前回 report と比較して profile 欠落、compatibility 低下、event budget / delay / low precision / online update policy check の退行を observed-only で検出できるようにした。CLI では `--history-path` / `--no-history-update` を追加し、managed `workspace/evaluation` 配下で履歴を更新できる。  
    * Phase 3 summary / release soak summary / operational readiness summary に `neuromorphic_profile_history_regression_observed`、`neuromorphic_profile_trend_regression_count`、`neuromorphic_profile_trend_policy_change_count` を表示し、単体 energy benchmark report を開かなくても long-run bundle から neuromorphic profile regression を確認できるようにした。  
    * operational readiness に neuromorphic profile regression 用の `recovery_hint` と recovery action を追加し、profile 欠落 / compatibility 低下時は `energy_efficiency_benchmark.py --no-history-update` で履歴を更新せずに再検査し、その後 Phase 3 summary を再生成する導線を出せるようにした。policy change のみの場合は release promotion 前の adapter policy review として中優先度 action にする。  
    * operational readiness summary に `neuromorphic_profile_trend_regression_details` と `neuromorphic_profile_trend_policy_change_details` を追加し、`akida:check_regression:low_precision_weight_ok` のような compact detail で壊れた profile / check / policy を report から直接読めるようにした。  
    * release soak summary にも `neuromorphic_profile_trend_regression_details` と `neuromorphic_profile_trend_policy_change_details` を追加し、release bundle 単体で profile regression の内訳を確認できるようにした。  
    * Stage E の micro-turn interaction / foreground-background handoff / phase-assigned submodel block / denoising correction を `neuromorphic_state_trace_ir` の state trace event として edge fixture に接続し、`neuromorphic_stage_e_state_trace_ir_observed`、`neuromorphic_stage_e_routing_hint_coverage_observed`、`neuromorphic_stage_e_online_update_policy_observed`、`neuromorphic_stage_e_event_budget_observed` で Lava / SpiNNaker / Akida profile が新しいサブモデル統合 trace を壊さず表現できるかを observed-only で監査できるようにした。  
    * Phase 3 summary / release soak summary / operational readiness summary に Stage E neuromorphic compatibility metrics を表示し、設計方針で重視する sparse event / local update / neuromorphic-ready integration が長時間 run の report bundle から確認できるようにした。  

#### **非目標 / 注意点**

* dense Transformer の大規模再学習を主軸に戻さない。  
* GPU 必須の訓練系へ設計を寄せない。  
* 生成品質だけを追って、世界モデル・継続学習・省電力性を犠牲にしない。  
* 「論文の流行そのもの」を追うのではなく、SARA Engine の中核原則に整合する要素だけを採用する。  
* hub formation, counterfactual replay, analogy mapping はいずれも導入候補だが、必ず lightweight benchmark と observability を先行させ、いきなり巨大な汎用実装へ飛ばない。  

#### **完成イメージ**

* 少量の経験から task に適応できる。  
* 長期運用で忘れにくい。  
* 潜在状態の予測を使って、応答や行動の一貫性を上げられる。  
* 複数ドメインの知識を hub-like concept node で短い経路に束ねられる。  
* 実行していない選択肢も offline に比較し、counterfactual simulation で plan を洗練できる。  
* オブジェクト表現と関係表現を分離した binding により、構造的類推と zero-shot generalization を強化できる。  
* world model, memory, planning, action が疎結合に連携する。  
* CPU-first / low-power / event-driven の強みを維持したまま高度化できる。  

## **5\. 実用化に向けての課題 (Challenges for Practical Application)**

SARA Engineを商用レベルや既存のディープラーニング（ANN）の代替として実用化するためには、現在以下の課題を克服する必要があり、これらは上記ロードマップの各フェーズの目標に組み込まれています。

1. **リリース品質の継続的担保 (Phase 2):**  
   * 実験コードと出荷候補コードを混在させない運用、CLI 回帰、soak run、配布メタデータ整合性の維持が必要です。  
2. **従来型ANNに対する学習効率と精度の客観的実証 (Phase 3):**  
   * SNNはエネルギー効率が極めて高い反面、複雑なタスクではDNNやTransformerなどのANNに精度（Accuracy）で劣りやすいという一般的な課題があります。まず限定タスクで「省エネルギー込みで見劣りしない」ことを示す必要があります。  
3. **大規模モデルへのスケーラビリティと分散処理の確立 (Phase 4):**  
   * 現在の基礎的なシミュレータ基盤から、実用的な巨大モデルを動かすための「数千万〜億単位のニューロン規模」へ安定してスケールアップさせる必要があります。マルチコアや分散環境におけるイベントルーティングの最適化が必須です。  
4. **継続的学習（Continual Learning）と記憶の安定化 (Phase 4):**  
   * SNNの強みである「破局的忘却のないオンライン学習」を実証するため、ネットワーク構造を動的に変更する機能（Structural Plasticity）の安定化と、短期・長期記憶間の知識転送アルゴリズムの確立が急務です。  
5. **高次推論とマルチモーダル統合の実現 (Phase 5):**  
   * 視覚・聴覚・言語など複数のモダリティを共通のスパイク空間に統合する仕組みや、連続的な動画・音声といった時空間予測ストリームへ対応する高度なアーキテクチャ（Spiking H-JEPA等）の具現化が必要です。  
6. **エッジ環境での完全な自律動作と生涯学習 (Phase 6):**  
   * エージェント自身の行動計画を組み込んだ能動的推論（Active Inference）を実現し、多様な環境（多言語や物理環境）に適応しながら、オンデバイスで低消費電力に自己成長を続ける「生涯学習（Lifelong Learning）」を完全にデプロイすることが最終的な壁となります。  

# **SARA Engine: Documentation Hub**

SARA Engine (Spiking Architecture for Reasoning and Adaptation) は、従来の人工ニューラルネットワーク (ANN) の限界を打破し、次世代の脳型認知アーキテクチャを実現するためのフレームワークです。

このディレクトリには、プロジェクトの思想、設計ドキュメント、そして今後のロードマップがまとめられています。

## **1\. コア設計ポリシー (Design Philosophy)**

SARA Engineは、以下の厳格なポリシー (doc/policy.md) に基づいて設計されています。

1. **Biological Plausibility (生物学的妥当性):** \* 誤差逆伝播法（バックプロパゲーション）に依存しません。  
   * 密な行列演算（Dense Matrix Multiplication）を排し、イベント駆動のスパースなスパイク伝達を利用します。  
   * STDP（スパイクタイミング依存シナプス可塑性）、報酬変調型STDP、遅延可塑性などの局所的な学習則を用います。  
2. **Self-Organization & Homeostasis (自己組織化と恒常性):** \* ネットワークトポロジは静的ではなく、学習の過程で動的にシナプスが生成・消滅（Structural Plasticity）し、最適な構造を自己組織化します。  
   * ホメオスタシス（恒常性）機構により、発火率を安定させ、一部のニューロンへの負荷集中を防ぎます。  
3. **Hardware Efficiency (ハードウェア効率):** \* GPUによる並列計算を必須とせず、RustコアによるCPU上での超高速・低消費電力な推論と学習を実現します。エッジデバイスでの実行を第一級の市民として扱います。

## **2\. コアモジュールとドキュメント**

* [Architecture Design](http://docs.google.com/idea/SNN-based_AI_architecture_design_document.md): SNNベースのAIアーキテクチャの詳細設計。  
* [Stateful SNN Theory](http://docs.google.com/idea/stateful_snn_theory.md): 状態を持つSNNの理論的背景。  
* [Self-Organization](http://docs.google.com/idea/Self-organization_and_homeostasis.md): 自己組織化と恒常性のメカニズム。  
* [Tools & Setup (EN)](http://docs.google.com/bout-Tools-EN.md) / [(JP)](http://docs.google.com/bout-Tools-JP.md): プロジェクト内で使用する解析・可視化ツールの説明。

### **マイルストーン**

* [Detailed Roadmap](http://docs.google.com/ROADMAP.md): 詳細な実装ロードマップと進捗状況。  
* [Release Strategy](http://docs.google.com/idea/SARA%20Engine%20PyPI%20Release%20Strategy.pdf): PyPIを通じたパッケージリリースの戦略。  
* [Competitive Analysis](http://docs.google.com/COMPETITIVE_ANALYSIS.md): 他のアーキテクチャとの比較分析。

## **4\. 開発スケジュール・ロードマップ (Master Schedule)**

今後の工程は、まず **リリース優先** で実用上の信頼性を固め、その後に **学習精度・推論精度の強化** へ進む方針とします。  
ANN系AIに正面から追随するのではなく、まず「CPU中心・低消費電力・バックプロパゲーション不要」というSNNの強みを保ったまま、出荷可能な品質を確立することを優先します。

### **Phase 1: Foundation & Rust Core Acceleration (完了/最適化中)**

* **目標:** 生物学的学習則（STDP等）の確立と、Rustによる超高速なイベント駆動シミュレータの統合。  
* **成果:** `sara_rust_core` によるバックエンド統合、SNN Transformer、Spatiotemporal STDPモジュールの完成。CPU上での高速推論の達成。  
* **位置づけ:** 以後のリリース安定化と精度改善の土台フェーズ。

### **Phase 2: Release Readiness & Practical Stability (最優先 / 直近)**

* **目標:** 現行のSNN基盤を「試作」から「安全に配布できるプレリリース品質」へ引き上げる。  
* **完了済み/進行中の項目:**  
  * direct memory の保存・復元経路の統一と unsafe `eval()` の除去。  
  * `SaraAgent` の runtime diagnostics、セッション永続化、CLI 診断表示の追加。  
  * FORCE artifact の入力検証、`UnifiedSNNModel` の二重更新バグ修正。  
  * 軽量 soak test、CLI dispatch test、release metadata test の追加。  
  * `scripts/old/` の legacy 扱いの明文化。  
* **残タスク:**  
  * wall-clock を長めに取った soak run の運用基準策定と `extended` profile の運用定着。  
  * 配布前チェックリストに基づく最終確認フローの固定。  
  * README / release notes / packaging metadata の継続整備。  
  * 主要 CLI コマンドの end-to-end カバレッジ拡張。  
* **完了条件:**  
  * リリースチェックリストの主要項目を継続的に満たせる。  
  * CPU-only 環境で回帰テストと soak run が安定完走する。  
  * モデル保存・復元・CLI 導線・診断導線に致命的不整合がない。

### **Phase 3: Accuracy Uplift for Learning & Inference (baseline implementation complete)**

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
* **現在の到達点:**  
  * lightweight benchmark と aggregated suite が通る状態まで実装済みで、Phase 3 の baseline instrumentation と品質 gate は一段完了。  
  * より広い ANN 比較、タスク拡張、長期 continual learning の大規模検証は Phase 4 以降の拡張テーマとして継続する。  

### **Phase 4: Scale-out & Continuous Learning (中期)**

* **目標:** 数千万〜億単位のニューロン規模へのスケールアップと、破局的忘却のないオンライン学習の実証。  
* **タスク:**  
  * 動的構造変更（Structural Plasticity）の安定化とスケーラビリティ向上。  
  * LTM (Long-Term Memory) と海馬モジュール間の知識転送・記憶固定化のアルゴリズム強化。  
  * マルチコア・分散環境でのイベントルーティングの最適化と、Rust側の非同期処理の強化。  
  * 限定タスクで確立した高精度化手法を、大規模構成へ移植する。  
* **完了条件:**  
  * 精度改善とスケール改善が両立し、継続学習時の品質劣化が許容範囲に収まる。

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

### **Phase 6: Autonomous General Intelligence (長期 / 最終目標)**

* **目標:** 完全に自律的で、多言語環境や物理環境と相互作用しながら自己成長を続ける強いAIの実現。  
* **タスク:**  
  * Spiking H-JEPA で獲得した世界モデルを基盤に、リアルタイム意思決定と Reward-Modulated STDP の実運用を確立する。  
  * 外部環境の言語（英語・日本語・フランス語等）に適応し、自発的に語彙や概念を獲得する多言語基盤を自己組織化する。  
  * エッジデバイスへの完全デプロイと、オンデバイスでの低消費電力な生涯学習（Lifelong Learning）を達成する。

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

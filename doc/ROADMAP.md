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

ANNを乗り越え、強力な自律型エージェントを実現するための開発フェーズです。

### **Phase 1: Foundation & Rust Core Acceleration (完了/最適化中)**

* **目標:** 生物学的学習則（STDP等）の確立と、Rustによる超高速なイベント駆動シミュレータの統合。  
* **成果:** sara\_rust\_core によるバックエンド統合、SNN Transformer、Spatiotemporal STDPモジュールの完成。CPU上での高速推論の達成。

### **Phase 2: Scale-out & Continuous Learning (現在〜中期)**

* **目標:** 数千万〜億単位のニューロン規模へのスケールアップと、破局的忘却のないオンライン学習の実証。  
* **タスク:** \* 動的構造変更（Structural Plasticity）の安定化とスケーラビリティ向上。  
  * LTM (Long-Term Memory) と海馬モジュール間の知識転送・記憶固定化のアルゴリズム強化。  
  * NLPやVisionタスクにおいて、少数のデータからのOne-shot学習がANNを上回ることをベンチマークで実証。  
  * マルチコア・分散環境でのイベントルーティングの最適化と、Rust側の非同期処理の強化。

### **Phase 3: Spiking H-JEPA & Advanced Predictive Coding (中期〜後期)**

* **目標:** 自己教師あり学習による抽象的な潜在空間表現の獲得と、誤差逆伝播に頼らない高次推論（Hierarchical Joint Embedding Predictive Architecture）のSNN上での実現。  
* **背景と意義:** 従来の生成モデルが陥りがちな「ピクセルレベルの厳密な再構成」を避け、SDR（Sparse Distributed Representation: スパース分散表現）を用いた意味的・抽象的な潜在空間での未来予測世界モデルを確立します。  
* **Spiking H-JEPA 進化のステップ (Evolutionary Steps):**  
  * **Step 1: 基礎的JEPAモジュールの確立 (Foundation)**  
    * オンラインネットワーク（スパイクによる現在の状態表現）とターゲットネットワーク（遅延可塑性を用いた目標表現）間の予測誤差を、STDPによるスパイクタイミングの差として定式化し、自律的に重みを更新するバックプロパゲーションレスな基本アーキテクチャの実装。  
    * メタ可塑性（BCM則によるスライディング閾値）や恒常性機能と組み合わせ、ノイズの多い環境下でも概念学習が発散せずに安定稼働することを評価する。  
  * **Step 2: 階層化と抽象化 (Hierarchical Predictive Coding)**  
    * 複数層にわたる階層的予測符号化の実装。上位層からの「トップダウン予測スパイク」と、下位層からの「ボトムアップ観測スパイク」が局所的に相互作用し、予測が外れた（Surprise）部分のニューロンのみを発火させるエネルギー最小化モジュールの構築。  
    * 階層を上がるにつれて、空間的・時間的な受容野が広がり、より抽象度の高い概念をスパイク群として表現できるようにする。  
  * **Step 3: 時空間予測ストリームへの拡張 (Spatiotemporal Prediction)**  
    * 静的なデータだけでなく、連続的な動画や音声ストリームデータに対応。時間軸における「未来の潜在状態」予測を、スパイクの到達タイミングと発火頻度の双方で表現する。  
    * 長期依存性を捉えるため、リカレント結合（SpatioTemporal STDP）による自己回帰的な推論メカニズムと融合させる。  
  * **Step 4: マルチモーダル統合と能動的推論 (Multimodal & Active Inference)**  
    * 視覚、聴覚、言語などの異なる入力モダリティを共通のスパイク潜在空間（Joint Embedding）にマッピングし、「画像からテキストの未来状態を予測する」といった異種データ間での強力な連想推論を実現する。  
    * エージェント自身の「行動計画」が未来の予測状態に与える影響を組み込む能動的推論（Active Inference）の導入。報酬修飾型STDP（R-STDP）と連動し、行動による予測誤差の最小化を自律的に学習させる。

### **Phase 4: Autonomous General Intelligence (最終目標)**

* **目標:** 完全に自律的で、多言語環境や物理環境と相互作用しながら自己成長を続ける強いAIの実現。  
* **タスク:**  
  * Spiking H-JEPAで獲得した世界モデルを基盤とした、リアルタイムの自律的意思決定と強化学習（Reward-Modulated STDPの実運用）。  
  * 外部環境の言語（英語・日本語・フランス語等）に適応し、自発的に語彙や概念を獲得する多言語基盤の自己組織化。  
  * エッジデバイスへの完全デプロイと、オンデバイスでの低消費電力な生涯学習（Lifelong Learning）の達成。
# **SARA Engine: Documentation Hub**

SARA Engine (Spiking Architecture for Reasoning and Adaptation) は、従来の人工ニューラルネットワーク (ANN) の限界を打破し、次世代の脳型認知アーキテクチャを実現するためのフレームワークです。

このディレクトリには、プロジェクトの思想、設計ドキュメント、そして今後のロードマップがまとめられています。

## **1\. コア設計ポリシー (Design Philosophy)**

SARA Engineは、以下の厳格なポリシー (doc/policy.md) に基づいて設計されています。

1. **Biological Plausibility (生物学的妥当性):**  
   * 誤差逆伝播法（バックプロパゲーション）に依存しません。  
   * 密な行列演算（Dense Matrix Multiplication）を排し、イベント駆動のスパースなスパイク伝達を利用します。  
   * STDP（スパイクタイミング依存シナプス可塑性）、報酬変調型STDP、遅延可塑性などの局所的な学習則を用います。  
2. **Self-Organization & Homeostasis (自己組織化と恒常性):**  
   * ネットワークトポロジは静的ではなく、学習の過程で動的にシナプスが生成・消滅（Structural Plasticity）し、最適な構造を自己組織化します。  
   * ホメオスタシス（恒常性）機構により、発火率を安定させ、一部のニューロンへの負荷集中を防ぎます。  
3. **Hardware Efficiency (ハードウェア効率):**  
   * GPUによる並列計算を必須とせず、RustコアによるCPU上での超高速・低消費電力な推論と学習を実現します。エッジデバイスでの実行を第一級の市民として扱います。

## **2\. ANN系AIを乗り越えるための戦略 (How to Surpass ANNs)**

現在のLLMをはじめとするANNは、巨大な計算資源による「力技」の極みですが、消費電力の増大、破局的忘却、コンテキスト長の限界という構造的な弱点を持っています。SARA Engineは以下の戦略でANNを乗り越えます。

* **Continuous Online Learning (連続的なオンライン学習):**  
  ANNのように「事前学習」と「推論」のフェーズを明確に分けません。環境と相互作用しながら、破局的忘却を起こさずにリアルタイムに新しい概念を獲得し続けます。  
* **Infinite Context via Stateful Dynamics (状態ダイナミクスによる無限の文脈):**  
  Attention行列サイズによる制限をなくし、ニューロンの膜電位やシナプス遅延という「状態」を利用することで、時間的な流れを自然にエンコードし、原理的に無限の文脈を扱います。  
* **Dynamic Resource Allocation (動的リソース割り当て):**  
  計算リソースを全体に使うのではなく、必要なモジュール（Cortical Column）だけがスパイク発火する「スパース・ルーティング」により、ANNでは不可能なレベルの省電力を実現します。

## **3\. ドキュメント一覧**

### **コンセプト・アーキテクチャ**

* [SNN-based AI architecture design document](http://docs.google.com/idea/SNN-based_AI_architecture_design_document.md): SNNベースの全体アーキテクチャ構想。  
* [Next-generation brain-like cognitive architecture](http://docs.google.com/idea/Next-generation_brain-like_cognitive_architecture_based_on_self-organized_criticality_and_dendritic_computation.md): 樹状突起計算と自己組織化臨界現象を利用した次世代認知モデル。  
* [Self-organization and homeostasis](http://docs.google.com/idea/Self-organization_and_homeostasis.md): ネットワークの自律的成長と安定化のメカニズム。  
* [Stateful SNN Theory](http://docs.google.com/idea/stateful_snn_theory.md): 状態保持型SNNによる系列処理と時間ダイナミクス理論。  
* [Core Policy](http://docs.google.com/policy.md): SARA Engineの絶対的な設計原則。

### **マニュアル・ツール**

* [Training Manual](http://docs.google.com/SARA-Engine_Training_Manual.md): 学習の実行、パラメータチューニングのガイド。  
* [Tools (English)](http://docs.google.com/bout-Tools-EN.md) / [Tools (Japanese)](http://docs.google.com/bout-Tools-JP.md): プロジェクト内で使用する解析・可視化ツールの説明。

### **マイルストーン**

* [Detailed Roadmap](http://docs.google.com/ROADMAP.md): 詳細な実装ロードマップと進捗状況。  
* [Release Strategy](http://docs.google.com/idea/SARA%20Engine%20PyPI%20Release%20Strategy.pdf): PyPIを通じたパッケージリリースの戦略。

## **4\. 開発スケジュール・ロードマップ (Master Schedule)**

ANNを乗り越え、強力な自律型エージェントを実現するための開発フェーズです。

### **Phase 1: Foundation & Rust Core Acceleration (完了/最適化中)**

* **目標:** 生物学的学習則（STDP等）の確立と、Rustによる超高速なイベント駆動シミュレータの統合。  
* **成果:** sara\_rust\_core によるバックエンド統合、SNN Transformer、Spatiotemporal STDPモジュールの完成。CPU上での高速推論の達成。

### **Phase 2: Scale-out & Continuous Learning (現在〜中期)**

* **目標:** 数千万〜億単位のニューロン規模へのスケールアップと、破局的忘却のないオンライン学習の実証。  
* **タスク:**  
  * 動的構造変更（Structural Plasticity）の安定化とスケーラビリティ向上。  
  * LTM (Long-Term Memory) と海馬モジュール間の知識転送・記憶固定化のアルゴリズム強化。  
  * NLPやVisionタスクにおいて、少数のデータからのOne-shot学習がANNを上回ることをベンチマークで実証。  
  * マルチコア・クラスタ環境でのRustノード分散処理プロトコルの実装。

### **Phase 3: Infinite Context & Multimodal Autonomous Agent (長期)**

* **目標:** コンテキスト長の概念を取り払い、視覚・聴覚・言語を同時に処理する完全自律型の知能の完成。  
* **タスク:**  
  * グローバルワークスペース（Global Workspace）を通じた、複数モダリティ間の意識的な情報統合。  
  * 遅延可塑性とスパイキング位相コーディングによる、長期的な時間依存性の完全な掌握。  
  * 対話、推論、行動決定をリアルタイムに実行する sara\_agent の汎用人工知能化。  
  * ANNのフラグシップモデル（GPTクラス）と同等の推論能力を、1/100以下の消費電力（エッジデバイス）で実現。

### **Phase 4: Beyond & Ecosystem Evolution (最終形態)**

* **目標:** SARA Engineが新たなAI開発のデファクトスタンダードとなるエコシステムの構築。  
* **タスク:**  
  * 自己組織化によりタスク固有のアーキテクチャを自動設計するメタ学習機能。  
  * 多様なIoT、ロボティクスエッジデバイスへの展開。  
  * PyPI及びオープンソースコミュニティによるモジュールの拡張エコシステム形成。
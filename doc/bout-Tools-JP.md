# **ツールとサンプルスクリプトの詳細**

SARA Engine プロジェクトに含まれる各種サンプルスクリプト、ベンチマーク、テストツールの概要です。

## **1\. Examples (examples/)**

このディレクトリには、エンジンの各機能を示すデモスクリプトやパフォーマンス測定用のベンチマークが含まれています（全59ファイル）。

### **デモンストレーション (Demos)**

SARAの主要なコンポーネントやユースケースを示します。

* **エージェント・チャット**  
  * demo\_agent\_chat.py: 自律エージェントによる対話デモ。  
  * demo\_interactive\_chat.py: インタラクティブなチャットインターフェース。  
  * demo\_million\_token\_agent.py: 長大なコンテキストを扱うエージェントのデモ。  
* **Spiking Neural Networks (SNN)**  
  * demo\_advanced\_snn.py: 高度なSNN構成のデモ。  
  * demo\_mnist\_snn.py: SNNを用いたMNIST数値認識。  
  * demo\_snn\_classification.py: 一般的な分類タスクのデモ。  
  * demo\_snn\_learning.py: オンライン学習機能のデモ。  
  * demo\_snn\_feature\_extraction.py: 特徴抽出パイプライン。  
* **Transformer & LLM**  
  * demo\_bio\_transformer.py: 生物学的に妥当なトランスフォーマーのデモ。  
  * demo\_snn\_transformer.py: SNNベースのトランスフォーマー実装。  
  * demo\_snn\_transformer\_multipath.py: マルチパス構成のトランスフォーマー。  
  * demo\_spiking\_llm.py: スパイクベースの言語モデルデモ。  
  * demo\_spiking\_llm\_text.py: テキスト生成に特化したスパイクLLM。  
  * demo\_spiking\_llm\_save\_load.py: モデルの保存と読み込みの例。  
* **マルチモーダル & パイプライン**  
  * demo\_multimodal\_memory.py: 複数モダリティにまたがる記憶システム。  
  * demo\_multimodal\_pipeline.py: マルチモーダル処理パイプラインの構築。  
  * demo\_snn\_pipelines.py: 推論パイプラインの一括デモ。  
* **特定タスク・応用**  
  * demo\_snn\_audio\_classification.py: 音声データ認識。  
  * demo\_snn\_image\_classification.py: 画像データ認識。  
  * demo\_snn\_text\_classification.py: テキスト分類。  
  * demo\_snn\_text\_generation.py: スパイクベースのテキスト生成。  
  * demo\_snn\_token\_classification.py: トークン単位の分類。  
  * demo\_snn\_rag.py: RAG（検索拡張生成）の実装例。  
  * demo\_snn\_rag\_persistent.py: 永続化ストレージを使用したRAG。  
  * demo\_rl\_training.py: 強化学習のトレーニングループ。  
  * demo\_predictive\_coding.py: 予測符号化（Predictive Coding）の実装。  
  * demo\_predictive\_lm.py: 予測言語モデル。  
  * demo\_semantic\_spike\_routing.py: セマンティック・スパイク・ルーティング。  
  * demo\_spike\_attention.py: スパイクアテンション機構。  
  * demo\_spike\_dataloader.py: スパイクデータ専用のローダー。  
  * demo\_spike\_stream\_processing.py: ストリームデータ処理。  
  * demo\_stream\_learning.py: リアルタイムストリーム学習。  
* **ハードウェア・エッジ・Rust**  
  * demo\_sara\_board.py: SARA Boardインターフェースのデモ。  
  * demo\_sara\_edge.py: エッジデバイス向けデプロイ例。  
  * demo\_saraboard\_and\_loader.py: ボードとデータローダーの連携。  
  * demo\_rust\_snn\_no\_numpy.py: NumPyに依存しないRustコアによるSNN。  
  * demo\_nn\_module.py: 新しいNNモジュール構成のデモ。

### **ベンチマーク (Benchmarks)**

パフォーマンスと精度の測定を行います。

* benchmark\_hal.py: ハードウェア抽象化レイヤー（HAL）の効率測定。  
* benchmark\_long\_context.py: 長文処理時のメモリと速度の評価。  
* benchmark\_memory\_retention.py: 記憶の保持能力（忘却耐性）の評価。  
* benchmark\_multicore.py: マルチコア並列処理のスケール性能。  
* benchmark\_multimodal\_associative.py: マルチモーダル連想記憶の性能。  
* benchmark\_rl\_stdp.py: STDPを用いた強化学習の収束性。  
* benchmark\_rust.py: Python実装とRustコア実装の速度比較。  
* benchmark\_rust\_acceleration.py: Rustによる加速効果の詳細測定。  
* benchmark\_snn\_transformer.py: SNNトランスフォーマーの計算コスト評価。

### **インタラクティブ & ユーティリティ**

* interactive\_demo.py: 総合的なインタラクティブGUIデモ。  
* interactive\_snn.py: SNNの挙動をリアルタイムで確認するツール。  
* visualize\_stdp.py: STDP（スパイクタイミング依存可塑性）の視覚化。  
* utils.py: サンプルスクリプト共通のユーティリティ関数。  
* test\_knowledge\_recall.py: 知識想起の精度評価（サンプル内テスト）。  
* test\_spike\_dataloader.py: データローダーの機能検証（サンプル内テスト）。  
* test\_transformer\_components.py: トランスフォーマー各部位の動作確認。

## **2\. Tests (tests/)**

システムの信頼性を担保するためのユニットテストおよび統合テストです（全10ファイル）。

* test\_crossmodal\_association.py: モダリティ間連想機能のテスト。  
* test\_event\_driven\_snn.py: イベント駆動型SNNエンジンの動作検証。  
* test\_hippocampal\_system.py: 海馬を模した短期・長期記憶システムのテスト。  
* test\_million\_token\_snn.py: 超大規模コンテキスト処理の安定性テスト。  
* test\_neurofem.py: NeuroFEM（神経有限要素法）の基本演算テスト。  
* test\_neurofem\_2d.py: 2次元空間におけるNeuroFEMのシミュレーション。  
* test\_neurofem\_integration.py: NeuroFEMとSNNの統合動作テスト。  
* test\_neurofem\_visualize.py: NeuroFEMの計算結果の視覚化テスト。  
* test\_new\_features.py: 新しく追加された各機能の総合検証。  
* test\_spatiotemporal\_stdp.py: 時空間STDPルールの動作検証。

## **3\. Scripts (scripts/)**

メンテナンスや運用のためのスクリプトです。

* health\_check.py: インストール環境、依存ライブラリ、Rustコアの接続状態を確認する診断ツール。
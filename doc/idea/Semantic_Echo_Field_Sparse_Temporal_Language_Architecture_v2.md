# **Semantic Echo Field v2**

## **イベント駆動型世界モデルとSNN言語アーキテクチャの統合**

**Event-Centric World Model Extension for SARA Engine**

**Document type:** Concept paper and falsifiable research proposal

**Project:** SARA Engine

**Version:** 2.0 (Integrated World Model Edition)

## **概要: 言語アーキテクチャから世界モデルへの転換**

本稿は、SNN（スパイキングニューラルネットワーク）を用いた自然言語処理アーキテクチャ「Semantic Echo Field」を拡張し、システムの根幹を「イベント駆動型世界モデル（Event-Centric World Model）」へと再定義する。

最大のパラダイムシフトは、「言語は第一義（Primary）ではない」という認識にある。自然言語の文（例：「猫が魚を食べた」）は、孤立した記号の羅列ではなく、現実世界で観測された「状態遷移（State \-\> Event \-\> State）」の圧縮されたラベル（描写）に過ぎない。

したがって、本システムはテキストの係り受けを解くことを最終目的とせず、多様なモダリティ（視覚、聴覚、触覚、言語）から抽出されたイベントをEvent Memory（イベント記憶）**として蓄積し、それらを束ねる**Temporal Relation Graph（時間的関係グラフ）**を構築する。かつて言語処理の中心であった**Semantic Echo Field（意味残響場）は、この巨大なイベント記憶に対して文脈的な共鳴を起こし、曖昧性解消や予測支援を行う「大脳皮質的なサブシステム（Cortical Subsystem）」として再配置される。

## **1\. 第一原理: 世界モデルとイベント記憶 (World Model First Principle)**

### **1.1 State \-\> Event \-\> State の基本ループ**

世界モデルの最小構成単位は、トークンではなく経験（Experience）である。システムは世界のダイナミクスを以下の遷移として理解し、保存する。

* **状態（State Before）:** 空腹である (Hungry)  
* **イベント（Event）:** 魚を発見し、近づき、食べる (Fish Appears \-\> Approach \-\> Eat)  
* **状態（State After）:** 満腹になる (Full)

### **1.2 状態特徴 (State Features)**

状態は固定のラベルではなく、疎な特徴量の集合として表現される。

*例: hungry, full, visible, reachable, moving, stationary, safe, dangerous*

モデルは孤立した単語の意味を学習するのではなく、これらの「状態がどのように遷移したか」を学習する。

### **1.3 動的意味の再定義 (Dynamic Meaning Redefinition)**

意味（Meaning）とは、静的なベクトルではなく「文脈に応じた状態遷移のパターン」として操作的に定義される。

「食べる（Eat）」という意味は、辞書的な定義ではなく、「空腹（Hungry）→ 食物を発見（FoodFound）→ 摂取（Consume）→ 満腹（Full）」という一連の予測可能な状態遷移のダイナミクスそのものである。

## **2\. アーキテクチャ構成: Event Memory First Design**

システム全体のデータフローは、言語から入るのではなく、多感覚なイベントの統合から始まる。

Multi-Modal Inputs (Vision, Audio, Touch, Proprioception, Language)  
  |  
  v  
Event Extraction & Bounded Decomposition  
  |  
  v  
Event Memory (基盤となる一次記憶)  
  |  
  v  
Episodic Memory (エピソードの形成と統合)  
  |  
  v  
Temporal Relation Graph (因果と時間関係のネットワーク)  
  |  
  v  
Semantic Echo Field (共鳴場: 文脈復元・推論・言語バインディング)  
  |  
  v  
State Prediction & Concept Crystallization (次状態の予測と概念の結晶化)

### **2.1 Event Memory (イベント記憶層)**

システムの主要な記憶基盤。入力されたトークンではなく、構造化された経験を保持する。

{  
  "episode\_id": "ep\_1024",  
  "state\_before": \["hungry", "alert"\],  
  "event\_sequence": \["fish\_found", "approach"\],  
  "state\_after": \["eating", "engaged"\],  
  "reward": 0.8,  
  "surprise": 0.65,  
  "modality\_source": \["vision", "proprioception"\]  
}

### **2.2 Episodic Event Memory (エピソード記憶)**

Event Memoryに蓄積された個々の遷移を、時間的・空間的なまとまり（エピソード）として編成する。

* **責務:** エピソードの形成、文脈に応じた検索（Retrieval）、オフラインでの再生（Replay）、記憶の固定化（Consolidation）、矛盾の追跡（Contradiction tracking）。

### **2.3 Temporal Relation Graph Extension (時間的関係グラフ)**

イベント間の関係性を疎なグラフ（エッジ）として保持する。単なる「共起」ではなく、意味のある関係性を定義する。

* **Relations:** before, after, predicts, overlaps, enables, prevents, requires, supports, contradicts

### **2.4 分離を保ったイベント束ね (Bind, Do Not Collapse)**

共感覚的な近接性を導入する場合でも、各モダリティの表現そのものを1つの混合表現へ溶かし込むべきではない。SARA Engine では、**「混ぜる」のではなく「同じ出来事として束ねる」**ことを原則とする。

* **Modality-specific store:** 視覚、聴覚、触覚、言語、内的状態は、それぞれ独立した sparse event records として保持する。  
* **Shared binding identity:** 同一の出来事に属すると検証された records には、共通の `event_id` と `time_chunk_id` を付与する。  
* **Binding metadata:** 各 record / relation には `modality_id`, `confidence`, `binding_strength`, `uncertainty` を保持し、近さの理由を失わない。  
* **No payload collapse:** 共通IDを持つことは、payloadの完全統合を意味しない。視覚情報は視覚として、音情報は音情報として追跡可能でなければならない。  

この構造により、SARA は「雷」という言語イベントが音・光・驚きと近いことを扱えても、それらを1つの密なベクトルへ潰すことなく、監査可能な形で束ねられる。

## **3\. Semantic Echo Field: 言語レイヤーと共鳴場**

旧版では言語処理の中心であったSemantic Echo Field（意味残響場）は、再定義された世界モデルの上で、言語（テキストや音声）を内部のイベント記憶と結びつけるインターフェースおよび推論バッファとして機能する。

### **3.1 再配置された責務 (Repositioning)**

* **Contextual Resonance (文脈的共鳴):** 入力された言語（例：「猫」「食べる」）に対応する過去のイベント記憶を短期的に活性化させる。  
* **Retrieval Guidance (検索誘導):** 曖昧な入力から、最も確からしいエピソードをグラフから引き出す。  
* **Dependency Reconstruction (係り受けの再構築):** 言語特有の局所的な役割（主語・目的語など）を、イベントの「主体・対象」などの物理的役割とバインディングする。  
* **Prediction Support (予測支援):** 次に発話されるべき単語ではなく、「次に世界で何が起きるか」の予測を言語チャネルにフィードバックする。

### **3.2 多軸非同期言語表現 (Language as a Modality)**

言語入力は、密なテンソルではなく、複数の時間解像度を持つ疎な「イベント軸」に分解され、Event Memoryへ投入される。

* **Orthographic (表層形):** 文字・形態素  
* **Phonological (音韻):** 音素・アクセント（任意経路）  
* **Semantic (意味素):** 事前学習やグラフから引かれた概念特徴  
* **Role (役割仮説):** 物理的イベントにおける動作主や対象との一時的な位相バインディング (Phase-Role Slot)

重要なイベントだけが、Fast (局所), Medium (文脈), Slow (長期) の異なる時間スケールを持つ「残響（Echo）」として場に残り、時間的関係グラフ上のノードと共鳴（Resonance）を起こす。

### **3.3 Event Hub と抽象概念層**

モダリティ間をすべて直接結合すると混線しやすいため、SARA では次の3層を保つ。

1. **Modality-specific sparse spaces:** 各モダリティ固有の近傍表現。  
2. **Event hub / binding layer:** 同期・反復・検証により、同じ出来事に属する records を束ねる層。  
3. **Abstract concept layer:** 複数のエピソードと反証テストを通過した durable concepts のみを昇格させる層。  

Semantic Echo Field は主に 1 と 2 の間で働く短期的な共鳴機構であり、概念そのものを即座に固定する装置ではない。

## **4\. 学習メカニズムとSARA Engine統合仕様**

### **4.1 局所可塑性と Learning Priority Signal**

SNNの強みである誤差逆伝播に依存しない局所学習（STDP等）を維持する。ただし、学習の更新（シナプス荷重や遅延の調整）は、単なる時間的近接性だけでなく、以下の**学習優先度信号 (Learning Priority Signal)** によって変調される。

Priority \=  
    PredictionError (サプライズ・予測誤差)  
  \+ Novelty (新奇性)  
  \+ Reward (内的/外的報酬)  
  \+ Coverage (通常状態の網羅性)  
  \- Redundancy (過剰な重複のペナルティ)

### **4.2 SARA Engine との統合仕様**

本アーキテクチャは、SARA Engineの各フェーズと以下のように統合される。

| 提案要素 | SARA Engine 該当モジュール | 統合の役割 |
| :---- | :---- | :---- |
| **Event Extraction** | Phase 16 (Modality Adapter) | 各センサー入力を State-\>Event-\>State の疎なイベント形式に変換 |
| **Semantic Echo Field** | Phase 19 (Temporal Binder) / Dendritic Feedback | 多軸イベントの非同期バインディング、および一時的な文脈保持 |
| **Event / Episodic Memory** | Phase 18 (Event-State Cache) | 短期残響から選別された「経験」の長期保存層（海馬相当） |
| **Concept Crystallization** | Phase 17 (Resonance Credit) | 複数エピソードで検証・反証テストを通過した不変法則の概念化 |

## **5\. 外部連携: WordPress Event Memory CMS 統合**

SARA Engineが内部で構築する「Event Memory」や「Temporal Relation Graph」はブラックボックスになりがちである。これを人間が監査・編集・アノテーションし、またシステムへの教師データとして還元するためのUI基盤として、**WordPressをHeadless CMSとして活用する**。

### **5.1 WordPress 連携仕様 (SARA Memory Dashboard)**

WordPressのカスタム投稿タイプ（CPT）とタクソノミーを用いて、SARAの世界モデルを直接マッピングする。

* **CPT State:** システムが認識した世界の状態（例：「空腹」「雨が降っている」）。  
* **CPT Event:** 観測された事象。REST API経由でSARAから非同期にPOSTされる。JSONペイロードとして予測誤差や優先度スコアをメタデータに保持。  
* **CPT Episode:** 複数のEventを時間軸でまとめたコンテナ。  
* **CPT Concept (概念結晶):** 十分な検証を経て固定化されたルールや知識。  
* **Taxonomy Modality:** 視覚、聴覚、言語などの情報源。  
* **Taxonomy Relation:** Temporal Relation Graph のエッジ（predicts, contradicts 等）を投稿間のリレーション（acf-relationship等）として表現。

### **5.2 連携のメリット**

1. **監査と介入:** 開発者やユーザーが、WordPressの管理画面から直感的に「システムが世界をどう解釈したか（どのイベントにサプライズを感じたか）」を閲覧・修正できる。  
2. **Continuous Learningの基盤:** 人間がWordPress上で矛盾（Contradicts）フラグを立てたり、正しいRelationを手動で紐付け直すことで、SARAの再学習フック（Webhook）をトリガーできる。  
3. **記憶の外部化:** SNN内部の限られたリソース（Bounded Memory）から溢れた古いエピソードをCMS側に退避（コールドストレージ化）し、必要に応じてベクトル検索等で再活性化（Reactivation Hint）させる。

## **6\. 新規性と反証可能な仮説 (Hypotheses)**

旧版の仮説（H1〜H9: 長距離依存、多軸同期、遅延学習の有効性など）に加え、世界モデルベースへの転換に伴う新たな仮説を提示する。

* **H10: World Model Transfer (モダリティ間の知識転移)**  
  言語を通じて学習した因果関係（知識）は視覚や行動の予測精度を向上させ、逆に身体行動（Embodied Action）から得た物理法則の経験は、言語タスクにおける物理的推論（常識推論）のZero-shot精度を向上させる。  
* **H11: State Prediction (次状態の予測)**  
  LLMのように「次にくるトークン」を予測するのではなく、「次の世界の状態（Future States）」を予測し、それを後処理として言語化するアプローチは、未知のシナリオにおいて単なる言語モデルの自己回帰予測よりも高い矛盾検知率と妥当性を示す。

## **7\. 長期ロードマップ (Revised Long-Term Goal)**

Event-Centric World Model の実現に向け、以下の段階的な導入ステージを定義する。

* **Stage 1: Language Resonance:** テキスト入力を疎なイベントに分解し、固定された文脈内で共鳴・係り受けを解決する。（旧Semantic Echo Fieldの基本実装）  
* **Stage 2: Event Memory:** 言語入力から State \-\> Event \-\> State の遷移を抽出し、経験としてエピソード記憶に蓄積・検索できる仕組みを構築。  
* **Stage 3: Cross-Modal World Model:** 音声、視覚など他のモダリティからのイベント抽出を統合し、Temporal Relation Graph 上でモダリティ間の紐付けを行う。（WordPress CMS連携の実装）  
* **Stage 4: Embodied Prediction:** 単なる記憶の蓄積から、次に起きる状態や必要な行動を予測し、外界へ作用するシステムへの進化。  
* **Stage 5: Self-Improving Sparse Cognitive System:** 予測誤差と内的報酬に基づき、自律的に不要な記憶を忘却し、重要な概念を結晶化し続ける、真のエネルギー効率の高い認知システムの完成。

## **結論**

Semantic Echo Field v2 は、「言語をどう処理するか」という局所的な課題から、「世界をどう解釈し、その結果として言語をどう発出・理解するか」という根源的な認知モデルへとスコープを拡張した。

本アーキテクチャにおいて、自然言語はシステムが世界を理解するための「窓」の一つに過ぎない。事象（イベント）と状態遷移を第一の構成要素（First Principle）とし、多時間尺度の残響と局所的な共鳴（Echo & Resonance）を推論のエンジンとすることで、既存の巨大な密行列演算（LLM）とは根本的に異なる、真にSNNの利点（局所性、低遅延、エネルギー効率）を活かした継続学習可能なAI基盤を提供する。

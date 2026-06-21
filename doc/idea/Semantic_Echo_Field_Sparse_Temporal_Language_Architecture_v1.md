# Semantic Echo Field

## 疎な時間的残響と局所可塑性によるSNN言語アーキテクチャ

**English title:** Semantic Echo Field: A Sparse Temporal Language Architecture for Spiking Neural Networks  
**Document type:** Concept paper and falsifiable research proposal  
**Project:** SARA Engine  
**Status:** Proposed; no empirical superiority claim  
**Date:** 2026-06-12

## 概要

本稿は、テキストを離散的なトークンID列としてのみ扱うのではなく、表層形、音韻、意味素、局所統語役割、句境界、予測誤差などから構成される疎な時間イベント列として処理するSNN言語アーキテクチャを提案する。これらは密な4次元または5次元テンソルとして積み上げず、それぞれ異なる時間解像度、信頼度、発火条件を持つ非同期イベント軸として扱う。中心機構は **Semantic Echo Field（意味残響場）** である。入力中の重要な意味イベントは、全文を保存したり全トークン対を比較したりする代わりに、複数の有限時間尺度を持つ減衰状態として残る。後続イベントが到着すると、意味的一致、役割整合、遅延適合、検証状態に応じて局所的な共鳴が生じ、係り受け候補、照応候補、予測、記憶再活性化を形成する。

本方式はTransformerのAttentionをスパイクで模倣することを目的としない。全対全比較、密な埋め込み行列、誤差逆伝播、GPU依存を避け、疎イベントルーティング、局所可塑性、有限状態、明示的忘却を維持する。音韻化と位相結合は必須経路ではなく、曖昧性解消や一時的役割bindingを支援する任意チャネルとする。また、短期の流動的残響と、検証後にのみ固定される概念結晶を分離する。

提案の価値は、言語精度だけでは判断しない。固定時定数SNN、単純な多時間尺度SNN、外部ANNベースラインと同一条件で比較し、精度、棄権、イベント数、状態量、遅延、継続適応、`joule_per_success`を同時に評価する。SNNの疎性、局所性、エネルギー効率を失う場合、本方式は採用しない。

## Abstract

This paper proposes a sparse temporal language architecture for spiking neural networks in which text is represented as bounded, asynchronous multi-axis event streams rather than a flat sequence of token identifiers or a dense multimodal tensor. Orthographic, phonological, semantic, predictive, and causal-hypothesis events may operate at different temporal resolutions and confidence levels. Its central mechanism, the Semantic Echo Field, maintains only salient semantic events as decaying local traces over multiple finite timescales. Later events resonate with compatible traces through sparse semantic, temporal, role, and verification links, enabling dependency recovery and contextual binding without dense all-pairs attention. Optional phonological and phase-binding channels support ambiguity resolution but are not mandatory for comprehension. A verified crystallization path separates transient language dynamics from durable concepts. The proposal is explicitly SNN-first: runtime backpropagation, dense recurrent matrices, general-purpose ODE solvers, and GPU requirements are excluded. The hypothesis is accepted only if independently sourced evaluations show temporal-language gains over simpler fixed-SNN controls while preserving bounded state, sparse event cost, and non-regressing measured joules per successful task.

## 1. 問題設定

SNNは時間、遅延、局所因果、オンライン適応を自然に表現できる一方、テキスト処理には次の難点がある。

1. 単語IDや既成embeddingをスパイク化するだけでは、SNN固有の時間計算を十分に利用できない。
2. STDPは近接する時間因果を学びやすいが、長距離依存、照応、否定の作用域、入れ子構造には単独では不十分である。
3. 全履歴を保持すると状態とエネルギーが増大し、SNNの利点が失われる。
4. 外部の構文解析器や大規模言語モデルが完成済みの意味構造を与えると、SNN自身の能力を評価できない。
5. 単一の固定減衰時間では、局所的な語順と長い係り受けを同時に扱いにくい。
6. 文字を音素列へ展開すると時間解像度は増えるが、系列長、イベント数、同音異義性も増えるため、常に有利とは限らない。
7. 意味、予測、因果を同じ確度の軸として扱うと、観測事実とモデル仮説が混同される。

本研究の問いは次の通りである。

> 重要な言語イベントだけを有限の時間的残響として保持し、後続イベントとの疎な局所共鳴で構造を回復することにより、密なAttentionを使わずにSNNの言語精度を改善できるか。

## 2. 設計原則

提案方式は以下を非交渉条件とする。

- runtime learningは誤差逆伝播に依存しない。
- 密行列演算を主要なruntime機構にしない。
- GPUを正しさや通常動作の要件にしない。
- 各イベント、リンク、残響、役割スロット、更新回数に上限を設ける。
- 観測、予測、外部解析、検証済み概念を別の型として保持する。
- 精度向上を、イベント数、遅延、状態量、消費電力の無制限な増加で購入しない。
- 言語を他モダリティより上位の表現ハブにしない。
- より単純な固定SNNで同じ改善が得られる場合は、複雑な機構を採用しない。

## 3. 提案アーキテクチャ

```text
Text
  |
  v
Bounded Language Event Decomposition
  |- surface-form events
  |- optional phonological events
  |- learned semantic-feature events
  |- provisional role events
  |- boundary and timing events
  `- prediction-error events
  |
  v
Sparse Event Gate
  |
  v
Semantic Echo Field
  |- fast echoes
  |- medium echoes
  `- slow echoes
  |
  +--> Optional Phase/Role Binding
  |
  v
Verified Sparse Resonance
  |
  +--> transient prediction / abstention
  |
  `--> verified concept crystallization
          |
          v
     bounded hierarchical event-state memory
```

### 3.1 言語イベントの分解

入力語または部分語 `x_k` を、単一IDではなく疎なイベント集合へ変換する。

```text
E(x_k) = {
  surface events,
  phonological events,
  semantic-feature events,
  role-hypothesis events,
  boundary events,
  prediction-error events
}
```

すべてのチャネルを常時生成してはならない。表層形と順序は基本チャネルとし、それ以外は不確実性、課題、利用可能な根拠、イベント予算に応じて開く。

#### 表層形チャネル

- 文字、部分語、形態境界、句読点を疎な署名へ変換する。
- 未知語でも文字n-gramや部分語の共有によって完全な未知状態を避ける。
- 語彙表と署名幅に上限を設ける。

#### 音韻チャネル

- 読み、音節、モーラ、アクセント境界などを時間イベントへ変換する。
- 常時必須とせず、同形異音語、韻律、句境界、音声との結合に限定して利用する。
- コード、数式、URL、未知の固有名詞では無理に音声化せず、表層形へフォールバックする。

#### 意味素チャネル

- 「動作」「主体候補」「摂取」「生物」「否定」などの疎な特徴を表す。
- 初期の辞書由来特徴、局所共起から自己獲得した特徴、外部モデル由来特徴を区別する。
- 外部ANNが生成した特徴は `external_proposal` とし、SARAが検証するまで学習済み事実にしない。
- 固定された普遍意味素集合を前提とせず、Phase 14型のown-latent clusterと併用する。

#### 役割仮説チャネル

- 主語、目的語、述語、修飾、否定作用域などを確定ラベルではなく競合する仮説イベントとして表す。
- 語順、助詞、形態、共起、残響との整合から局所的に更新する。
- 日本語の省略、自由語順、引用、埋め込み節を扱うため、単一役割への早期確定を避ける。

### 3.2 多軸非同期言語表現

入力トークン列には元から順序があるため、音韻化によって時間が初めて生じるわけではない。音韻化の価値は、語または部分語単位の粗い順序を、音素、モーラ、音節、アクセント、境界の細粒度時間構造へ展開できる点にある。例えば「猫」は一つの語イベントであると同時に、読みが確定できる条件では `/n/ -> /e/ -> /k/ -> /o/` の局所系列を持ち得る。

この展開は密な座標空間ではなく、次の疎なイベント軸として実装する。

| 軸 | 型 | 時間解像度 | 信頼性 |
|---|---|---|---|
| Orthographic | observed | 文字、部分語、語 | 入力から直接観測 |
| Phonological | observed/proposed | 音素、モーラ、音節 | 読みの根拠に依存 |
| Semantic | learned/proposed | 概念特徴、own-latent | sourceと学習履歴に依存 |
| Predictive | predicted | 次イベント候補 | 不確実性付き |
| Causal | causal_hypothesis | 状態遷移候補 | 検証前は事実ではない |

各イベントは少なくとも次の情報を持つ。

```text
LanguageAxisEvent {
  axis,
  event_id,
  local_time,
  source_ref,
  sparse_signature,
  confidence,
  uncertainty,
  evidence_type,
  expiry,
  event_cost
}
```

軸同士は共有クロックへ強制的に密集させない。局所的な時間窓で同時または予測可能に発火したイベントだけをbinding候補とする。例えば表層形「猫」、音韻系列 `/n e k o/`、self-learnedな動物概念が繰り返し整合すれば、STDPまたは局所共起により疎な双方向リンク候補を形成できる。

```text
orthographic: 猫
phonological: /n/ -> /e/ -> /k/ -> /o/
semantic: animal, animate, agent_candidate
```

ただし、次の区別を守る。

- 音韻は意味へ到達する唯一の経路ではない。
- 文字形状、音韻、意味は同時に利用できるが、全チャネルの常時発火は避ける。
- 予測軸は次入力の候補であり、観測済み入力ではない。
- 因果軸は時間的先行だけでは確定せず、反例、介入相当の変化、source evidence、検証結果を必要とする。
- 感情や運動特徴は課題と根拠がある場合だけ追加し、普遍的な語表現の必須軸にしない。

この表現は「単語が存在しない」と断定するものではない。単語や部分語を唯一の計算単位にせず、複数時間尺度のイベントから再構成可能な安定ラベルとして扱う。語境界が有用な課題では語イベントを維持し、未知語、連続音声、形態変化ではより細かいイベントを利用する。

### 3.3 Semantic Echo Field

意味残響場は、重要イベントだけを複数の有限時間尺度で保持する疎な状態集合である。イベント `i` の残響強度を `e_i(t)` とすると、入力がない区間では次のように減衰する。

```text
e_i(t + dt) = e_i(t) * decay_i(dt)
```

新しいイベント `j` が到着したとき、残響候補 `i` との共鳴スコアを次の局所項から計算する。

```text
R(i, j) =
    semantic_overlap(i, j)
  + role_compatibility(i, j)
  + delay_compatibility(i, j)
  + source_reliability(i, j)
  + verified_history(i, j)
  - contradiction(i, j)
  - event_cost_penalty(i, j)
```

これは密なsoftmax Attentionではない。比較対象は以下で制限する。

- 現在活動中の残響だけ
- 同一または許可された局所近傍だけ
- 残響ごとの最大リンク数
- 時間尺度ごとの最大保持数
- 1入力あたりの最大共鳴評価数
- 最小活性値と有効期限

共鳴が閾値を超えると、次のいずれかを出力する。

- 係り受け候補
- 照応候補
- 予測イベント
- 記憶再活性化hint
- 役割binding
- 矛盾または棄権

### 3.4 多時間尺度と遅延

残響は少なくとも高速、中速、低速の有限tierへ分ける。

| Tier | 主な役割 | 例 |
|---|---|---|
| Fast | 局所順序と形態 | 助詞と直後の語、短い修飾 |
| Medium | 句と節 | 主語と述語、否定作用域 |
| Slow | 長距離文脈 | 埋め込み節、照応、話題 |

最初の実装では固定された複数時定数を用いる。可変時定数は、固定多時間尺度では解けない失敗が観測され、Phase 19の条件を満たす場合だけ試す。汎用ODEソルバーは使用しない。

シナプス遅延は、係り受けの正解を直接格納するものではない。共起と予測成功によって、後続イベントと再会しやすい時間帯を局所的に調整する。遅延値、遅延候補数、更新幅には上限を設ける。

### 3.5 位相と役割binding

固定的に「主語は0度、目的語は120度」と割り当てる方式は採用しない。自然言語では役割が重複し、節ごとに再利用され、語順も変化するためである。

代わりに、短命な **phase-role slot** を用いる。

- 句または節ごとに少数の局所位相スロットを生成する。
- 役割仮説はスロットへ一時的に同期する。
- 新しい節境界、矛盾、timeoutで解除する。
- 同一位相だけで意味を決定せず、意味残響、形態根拠、予測整合を必要とする。
- 位相同期を使わない対照実験を必須にする。

位相は構文木の代替と主張しない。疎な一時bindingを低コストで維持できるかを検証する補助仮説である。

### 3.6 内的音声化

内的音声化は、テキスト理解の必須パイプラインではなく任意の再符号化経路とする。

```text
surface events
  -> optional phonological event proposal
  -> audio-like temporal adapter
  -> semantic resonance
```

期待される用途は以下である。

- 読点や句境界の推定
- 音韻的曖昧性の処理
- 音声と文字の共有概念学習
- 未知表記の読み候補比較

常時音声化はイベント数と遅延を増やし、表記固有情報を失う可能性がある。そのため、音韻経路を開く条件と閉じる条件をtraceへ記録する。

### 3.7 動的意味モード

「意味を発火波として表す」という着想は、固定ベクトルを完全に廃止する宣言ではなく、意味を再現可能な **sparse dynamic mode** として操作的に定義する。

動的意味モードは次の要素からなる。

- 少数の概念イベント集合
- 発火順序または許容される部分順序
- 有限の遅延分布
- 局所位相または共鳴窓
- 再活性化時の許容変形
- source、confidence、verification状態

例えば「猫」「犬」「馬」は完全に同じ波形を共有するのではなく、`animate`、`animal`、`agent_candidate`など一部の動的部分構造を共有し、それぞれ固有の表層、音韻、経験リンクを持つ。概念近接性は、密ベクトル間距離だけでなく、共有イベント、再活性化順序、共鳴可能な遅延、予測結果の重なりとして測定できる。

動的モードが概念表現として認められるには、次を満たす必要がある。

- 同じ概念の異なる表記や文脈から部分的に再現される。
- 近縁概念では共有部分が増え、無関係概念では誤共鳴が抑えられる。
- 入力速度や小さな時間ずれに対して許容範囲内で安定する。
- 語順を変えると意味が変わる課題では、過剰な時間不変性を示さない。
- 活動を停止した後も、必要な情報だけがPhase 18型記憶から疎に再活性化できる。

周期振動を常時維持するとidle costが増えるため、動的意味モードは永続的な発振器ではなく、入力または再活性化hintで短時間だけ発生する減衰モードを基本とする。振動周波数そのものを意味ラベルに固定せず、イベント集合と相対タイミングの組合せを用いる。

### 3.8 流動的残響と概念結晶

一度の共鳴を永続知識にしてはならない。記憶を二段階に分離する。

#### 流動層

- 短命な残響、仮説、予測を保持する。
- 文脈に応じて素早く変化する。
- timeout、矛盾、低効用で消える。
- durable stateを直接変更しない。

#### 概念結晶層

- 複数の独立した根拠で再現された構造だけを保持する。
- source backing、検証、共鳴信用、代謝予算を必要とする。
- 類似概念を統合し、矛盾した概念を隔離する。
- 使用されない、誤りが増えた、費用が高い結晶を忘却する。

SARAでは、流動層をPhase 15、16、19の局所状態、概念結晶層をPhase 17のverified resonance creditとPhase 18のevent-state cacheへ接続できる。

## 4. 学習方式

### 4.1 第1段階: 局所時間学習

STDPまたは局所共起更新により、次を学ぶ。

- 表層形の近接
- 文字と音韻の対応
- 局所語順
- 句境界
- 短い遅延関係

更新はweight、delay、短期促通状態へ限定し、すべてclipする。

### 4.2 第2段階: 結果変調付き局所学習

予測成功、役割整合、検索成功、矛盾、棄権の正否を第三因子として用いる。

```text
local_update =
    eligibility_trace
  * verified_outcome_signal
  * metabolic_headroom
```

報酬だけでは更新しない。source integrity、矛盾検査、複数信号の一致を要求する。

### 4.3 第3段階: 検証付き構造安定化

繰り返し有用な残響パターンを概念結晶候補とする。候補は以下を通過した場合だけ永続化する。

1. 独立したsourceまたはrevisionで再現される。
2. 予測または検索成功に寄与する。
3. negative queryとcontrastive caseを壊さない。
4. 矛盾率と棄権精度が許容範囲内である。
5. event/state budgetに余裕がある。
6. 単純な既存構造との重複が小さい。

## 5. SARA Engineへの対応

| 提案要素 | 既存SARA機構 | 新規に必要なもの |
|---|---|---|
| 多チャネル言語イベント | Phase 16 modality adapter | bounded language event adapter |
| 多軸非同期binding | temporal binder, phase trace | axis-aware sparse binder |
| 意味残響 | dendritic feedback, temporal state | sparse echo field |
| 長距離再活性化 | Phase 18 event-state cache | language-aware reactivation policy |
| 意味素 | Phase 14 own-latent | source-labeled feature proposals |
| 動的意味モード | own-latent, oscillation traces | bounded mode signature and stability evaluator |
| 学習許可 | Phase 17 resonance credit | language-specific evidence bridge |
| 多時間尺度 | fixed decay and memory tiers | fixed echo tiers; optional Phase 19 |
| 位相binding | existing phase metrics | short-lived local role slots |
| 概念結晶 | verified cache promotion | concept utility and contradiction policy |

## 6. 新規性

本提案の新規性は、個々の要素を最初に発明したという主張ではない。SNNの言語処理、遅延、位相表現、semantic pointer、eligibility trace、階層記憶には先行研究がある。

提案する組合せの特徴は次の通りである。

1. Attentionの直接的なスパイク近似ではなく、有限の意味残響と局所共鳴を中心に据える。
2. 表層、音韻、意味、役割、予測誤差を同一の疎イベントIRで扱う。
3. 位相を固定文法ラベルではなく、短命な局所role bindingとして限定する。
4. 流動的言語状態と検証済み概念結晶を明確に分離する。
5. 外部ANN解析を提案値として隔離し、SNN単独能力との混同を防ぐ。
6. 精度改善と物理エネルギー非退行を同じ採用条件にする。
7. 多次元言語表現を密テンソルではなく、異なる時間解像度と証拠型を持つ非同期イベント軸として構成する。
8. 意味を永続発振ではなく、入力依存で再現される有限の疎動的モードとして検証する。

## 7. 反証可能な仮説

### H1: 長距離依存

Semantic Echo Fieldは、同じ状態上限を持つ単一減衰SNNより、長距離の係り受け、照応、否定作用域で高い正解率を示す。

### H2: 単純対照との差

Semantic Echo Fieldは、複数の固定時定数を持つだけのSNNより、同等以下のevent/state budgetで高い精度または棄権品質を示す。

### H3: 位相bindingの限定的有効性

phase-role slotは、埋め込み節や役割競合を含む課題で改善を示すが、単純文では不要な費用を増やさない。

### H4: 音韻経路の選択性

任意音韻経路は、曖昧性や句境界課題では改善するが、コード、数式、URLでは自動的に閉じ、表層経路より悪化しない。

### H5: 概念結晶化

検証付き結晶化は、継続学習後の保持率を改善し、誤情報、単発共起、自己生成データの誤固定を抑制する。

### H6: SNN優位性の維持

精度改善後も、外部ANN比較と同じ成功基準における`joule_per_success`、peak RSS、idle costの少なくとも主要なSNN優位性を維持する。

### H7: 多軸同期の有効性

表層、音韻、意味イベントの選択的な同期は、同じイベント予算の表層単独または単純連結表現より、未知語、形態変化、読み曖昧性、speech-text対応で高い品質を示す。

### H8: 動的意味モード

有限のsparse dynamic modeは、静的な疎署名だけの対照より、概念再活性化、近縁概念の共有、文脈依存の意味変化を改善し、idle時には追加発火を維持しない。

### H9: 因果仮説の完全性

causal-hypothesis軸は、単なる時間的共起より高い因果候補精度を示し、未検証候補を観測事実またはdurable conceptへ昇格させない。

## 8. 実験計画

### 8.1 比較方式

- Current SARA language path
- Single-decay fixed SNN
- Multi-timescale fixed SNN
- Semantic Echo Field without phase binding
- Semantic Echo Field with optional phase binding
- Semantic Echo Field with optional phonological route
- Semantic Echo Field with asynchronous orthographic/phonological/semantic binding
- Semantic Echo Field with bounded dynamic semantic modes
- BM25 retrieval baseline
- Lightweight pretrained embedding baseline
- Tiny Transformer or recurrent language baseline

Phase 19のliquid time constantは、固定多時間尺度対照が失敗した後だけ追加する。

### 8.2 課題

- 局所語順
- 長距離主述関係
- 目的語と述語の対応
- 埋め込み節
- 照応
- 否定と数量表現の作用域
- 語順変更
- 省略
- 同音異義と表記曖昧性
- 未知語、形態変化、読み速度変化
- speech-text alignment and mismatch
- semantic-neighbor versus unrelated-concept resonance
- temporal co-occurrence versus supported causal hypothesis
- noisy text
- adversarial near-miss
- unsupported queryと棄権
- delayed recall
- continual adaptation後の保持
- source revisionによる矛盾更新

### 8.3 時間同期マルチモーダルデータの収集

#### 基本仮説

テキスト量だけでなく、互いに独立した情報を持つ複数チャネルが正確な時間対応を持つことが、局所時間学習のsample efficiencyを高める可能性がある。音声、文字、映像、行動、反応を同一時刻周辺の疎イベントとして表せれば、STDP、遅延学習、Semantic Echo Field、Phase 16 cross-modal bindingを同じ経験から更新できる。

ただし、`同期チャネル数`だけをデータ価値と見なしてはならない。同じ内容を複製したチャネル、不正確な自動字幕、対象と無関係な映像、遅延した字幕は情報を増やさず、誤結合を増やす。提案する指標は **verified synchronization density** である。

```text
verified_synchronization_density =
    verified informative cross-channel bindings
    / observed duration
```

各bindingは、重複でない情報、許容範囲内の時刻整合、source provenance、信頼度、negative controlへの耐性を必要とする。

#### 収集順序

1. 自作または明示同意付きの音声、転写、タイムスタンプ
2. public domainまたは用途適合ライセンスが明確な音声・映像データセット
3. creatorまたは研究partnerから明示的に提供されたsource media
4. platform mediaは、platform規約と権利者許諾の両方が取得・保存・変換・学習を許す場合だけ
5. ロボットまたはセンサー経験は、同意、プライバシー、event schema、削除伝播が成熟した後

YouTubeは同期データの存在例としては有用だが、SARAでは汎用scraping sourceとして採用しない。YouTubeの公式caption download APIは動画を編集する権限を持つ利用者を要求し、Developer PoliciesはYouTubeの事前の書面承認なしにYouTube audiovisual contentをdownload、cache、storeすることを禁じている。標準YouTubeライセンスが既定であり、CC BY表示または公開URLだけで、取得手段と学習利用のすべてが自動的に許可されるとはみなさない。

#### 最小同期イベント形式

```text
SynchronizedExperienceEvent {
  session_id,
  source_clock_anchor,
  timestamp_ms,
  relative_delay_ms,
  duration_ms,
  modality,
  track_id,
  source_ref,
  source_hash,
  payload_ref,
  confidence,
  alignment_uncertainty_ms,
  evidence_type,
  license_id,
  consent_state,
  expiry_or_deletion_ref
}
```

イベント型の例:

- `observed_audio`
- `creator_caption`
- `automatic_caption`
- `local_asr_hypothesis`
- `observed_visual_event`
- `inferred_visual_label`
- `speaker_turn`
- `pause`
- `overlap`
- `reaction_event`
- `action_event`
- `causal_hypothesis`

自動字幕、視覚ラベル、感情、笑い、因果は観測された意味事実ではない。それぞれ出自と不確実性を保持し、検証前にdurable conceptへ昇格させない。

#### 階層イベント圧縮

SNN向けデータ管理の目的は、すべての動画フレームや音声サンプルを学習IRへ保存することではない。一方、上位概念だけを残して生データを即座に破棄すると、特徴抽出器の誤りを監査できず、将来のより良いeventizerで再処理できない。そこで次の段階的表現を用いる。

```text
Level 0: bounded raw evidence
  -> Level 1: sensor/change events
  -> Level 2: provisional entities and actions
  -> Level 3: bounded episodes
  -> Level 4: temporal/causal hypotheses
  -> Level 5: verified reusable invariants
```

**Level 0: bounded raw evidence**

- 学習時の主表現ではなく、監査、校正、誤検出確認、限定的再処理のために保持する。
- 全量無期限保存は避け、pre/post event window、uniform audit sample、calibration sample、rights-cleared exemplarへ限定する。
- retention期限後もsource hash、license/consent、extractor version、削除状態を残す。

**Level 1: sensor/change events**

- 視覚の輝度・edge・motion変化、音響onset・周波数変化、触覚・力覚変化、字幕境界などを保存する。
- threshold、hysteresis、refractory periodを用いてノイズと連続重複を抑える。
- 安定区間はframe列ではなく、開始、終了、要約状態を持つ`state_continues`イベントへ変換する。

**Level 2: provisional entities and actions**

- `cat_present`、`speaker_changed`、`hand_reached`などをconfidence付き候補として保存する。
- extractorとmodel versionを保持し、観測事実と分類結果を混同しない。
- 低confidence候補はdurable learningへ直接使用しない。

**Level 3: bounded episodes**

- 同じsource、局所時間、対象、行動に関係するイベントを有限のepisodeへまとめる。
- episodeには開始/終了、参加event、代表event、予測、action、outcome、uncertaintyを持たせる。
- episode境界の誤りを評価できるよう、隣接候補と分割理由をtraceする。

**Level 4: temporal and causal hypotheses**

- `before`、`overlaps`、`predicts`、`action_precedes_result`、`causal_hypothesis`などの型付きedgeを構成する。
- edge delayは単一時刻ではなく許容区間または小さな分布として保持する。
- 時間的先行と因果を区別し、反例、異なるsource、action/result変化、検証なしにcausal factへ昇格させない。

**Level 5: verified reusable invariants**

- 繰り返し観測され、negative caseとcontradiction検査を通過した関係だけを保存する。
- 適用範囲、例外、source count、最終検証時刻、contradiction history、expiryを持つ。
- 「猫は魚を食べる」のような文は普遍法則ではなく、条件付き傾向として表現する。

上位levelは下位levelを置き換えるだけではなく、source hashとevent IDへのlineageを保持する。source削除、同意撤回、extractor更新、矛盾発見時には依存するepisode、relation、invariantを再評価または削除する。

#### 二重時間表現

相対時間はSTDPと遅延学習に有用だが、絶対時刻またはsource clock anchorを完全に捨ててはならない。複数センサー同期、再現、clock drift補正、source削除、監査に必要だからである。

各eventは次の二つを持つ。

- `source_clock_anchor + timestamp_ms`: source内での位置と再現用
- `relative_delay_ms`またはdelay interval: 局所学習とtemporal graph用

wall-clockの個人情報性が不要な場合、外部の実日時刻はsession-scoped opaque anchorへ変換できる。重要なのは、学習時に相対時間を使いながら、データ管理上の再現性を失わないことである。

#### Balanced Surprise Retention

予測誤差は重要な保持信号だが、`予想通りならすべて捨てる`方式は採用しない。通常例を失うとnormal baseline、頻度、calibration、no-change predictionを学べず、ノイズや異常だけに偏る。また、初期の未熟なモデルはほぼすべてをsurprisingと判断する。

保持優先度を次の有界な複合信号とする。

```text
retention_priority =
    w1 * bounded_surprise
  + w2 * expected_learning_gain
  + w3 * uncertainty
  + w4 * safety_or_rarity
  + w5 * representative_coverage
  + w6 * contradiction_value
  + w7 * source_reliability
  - w8 * redundancy
  - w9 * storage_and_event_cost
```

各項をclipし、次のquotaを分離する。

- routine representative states
- no-change and negative examples
- novel or high prediction-error events
- rare or safety-critical events
- contradictions and failed predictions
- recovery and successful adaptation outcomes

高surpriseでも、sensor glitch、字幕ずれ、adversarial novelty、低信頼sourceならquarantineする。学習後にsurpriseが下がったeventは、coverage、safety、再評価用途がなければ優先度を減衰させる。

#### Sparse Temporal Relation Graph

動画DBではなく経験DBとして、eventをnode、時間関係をtyped sparse edgeとして保存する。

```text
TemporalRelationEdge {
  source_event_id,
  relation_type,
  target_event_id,
  min_delay_ms,
  max_delay_ms,
  confidence,
  evidence_count,
  counterexample_count,
  verification_state,
  expiry
}
```

許可するrelation例:

- `before`
- `after`
- `overlaps`
- `same_episode`
- `predicts`
- `action_precedes_result`
- `co_occurs`
- `causal_hypothesis`

node数、outgoing edge数、episode depth、query expansionをhard capする。Phase 17のverified resonance creditとPhase 18の階層cacheを使い、contradiction、低utility、重複、期限切れedgeを忘却する。

#### 圧縮評価

圧縮率だけでは成功としない。次を同時に比較する。

- raw bytesからevent bytesへの削減率
- raw samplesからprocessed eventsへの削減率
- uniform sparse samplingとの比較
- missed salient event rate
- false event and false episode rate
- downstream task quality loss
- baseline/routine coverage
- rare/safety event recall
- relation precision and causal false-promotion rate
- reprocessing reproducibility
- extraction and learning joules
- `joule_per_success`

#### 段階的導入

**Stage A: Audio + transcript + timestamps**

- 音声区間、語または字幕区間、pause、speaker turnを保存する。
- transcript-only、true timestamps、shuffled timestampsを比較する。
- 人手字幕、creator caption、自動字幕、local ASRを分離する。

**Stage B: Sparse visual events**

- 全フレームを学習IRへ入れず、scene change、motion、object-presence候補、speaker visibilityを疎イベント化する。
- 発話時に対象が映っているとは限らないため、off-screen narration、montage、複数対象、字幕遅延をnegative caseにする。
- 一度の同期だけで「映像対象と単語が同一」と固定しない。

**Stage C: Conversation and response timing**

- turn onset、pause duration、overlap、interruption、repair、laughter/no-laughter、response latencyを別イベントとして扱う。
- 漫才や会話の「間」は有用な予測信号になり得るが、笑いまたは拍手を正しさや報酬と同一視しない。
- true timing、shuffled timing、transcript-onlyを比較する。

**Stage D: Embodied experience**

- 視覚、音声、触覚、運動、位置、力覚、行動結果を共通イベントIRへ変換する。
- ロボットのactionとresultは因果推定に有用だが、一回の遷移だけでは因果を確定しない。
- 安全、同意、bystander privacy、hardware event clock、sensor calibrationを先に満たす。

#### 収集品質と反証条件

- 時刻付きデータがtimestamp-shuffled controlを超えなければ、同期学習の利点を主張しない。
- 追加チャネルがtext/audio-only対照を超えなければ、そのチャネルを既定経路にしない。
- raw channel countではなく、informative bindings、alignment error、contradiction、false bindingを報告する。
- surprise-only retentionを使わず、routine、negative/no-change、rare/safety、contradiction、recovery strataを維持する。
- 生データ全量を学習IRにせず、bounded raw audit windowsとuniform calibration samplesを保持する。
- 相対遅延とsource clock anchorを併存させる。
- 上位episode、relation、invariantからsource hashと下位eventへのlineageを保持する。
- 同一動画のclip、alternate upload、字幕版、音声抽出版をtrain/evaluationへ分散させない。
- speaker、creator、series、session、source hash、collection timeで分離する。
- source削除、同意撤回、ライセンス変更を派生eventとconcept manifestへ伝播する。
- 個人識別情報、声紋、顔、位置、子供、bystander dataを必要最小限にし、許諾のないbiometric learningを行わない。

### 8.4 データ分離

- train/evaluationをsource、hash、revision、domain、timeで分ける。
- 自動生成教材を主要なheld-out評価に使用しない。
- 同一文の言い換えや形態変化をnear-duplicateとして検査する。
- 外部解析付きデータとraw text onlyデータを分ける。
- 日本語だけでなく、語順特性の異なる少なくとも1言語を含める。

### 8.5 評価指標

- exactまたはtask-specific accuracy
- macro F1
- dependency/role binding precision and recall
- abstention precision and recall
- contradiction detection
- delayed recall
- continual-learning retention
- harmful crystallization rate
- events per successful task
- resonance comparisons per input
- active echo count
- events and bindings per axis
- cross-axis binding precision and false resonance rate
- dynamic-mode reactivation stability
- idle spike count
- unverified causal promotion count
- informative channel count
- verified synchronization density
- alignment uncertainty and timestamp error
- timestamp-shuffle performance delta
- cross-modal contradiction and false-binding rate
- raw-to-event byte reduction
- processed-event reduction
- missed salient event and false event rate
- routine baseline coverage
- rare/safety event recall
- temporal relation precision
- causal false-promotion rate
- raw-to-invariant lineage integrity
- reprocessing reproducibility
- serialized state bytes
- latency
- peak RSS
- measured joules
- `joule_per_success`

### 8.6 Ablation

- phonological channel off
- transcript-only versus timestamp-aware
- true timestamps versus shuffled timestamps
- audio-text versus audio-text-vision
- creator caption versus automatic caption
- true conversation pauses versus shuffled pauses
- change-triggered eventization versus uniform sparse sampling
- balanced retention versus surprise-only retention
- dual anchored/relative time versus relative-time-only
- bounded raw audit windows versus immediate raw deletion
- temporal relation graph versus flat event replay
- asynchronous axis binding off
- axis-specific confidence and evidence typing off
- dynamic semantic mode replaced by a static sparse signature
- semantic-feature channel off
- phase-role slot off
- learned delays off
- resonance verification off
- crystallization off
- single decay versus fixed multi-timescale decay
- external proposals off
- source verification off

## 9. 採用基準

以下をすべて満たす場合だけproduction候補へ昇格する。

- independently sourced held-out tasksで固定SNN対照より改善する。
- 単純な多時間尺度SNNより改善するか、同等品質を明確に低い費用で達成する。
- negative query、contrastive case、棄権が悪化しない。
- active echo、リンク数、更新数、state bytesがhard limit内にある。
- runtime learningが局所的で、backpropagationを必要としない。
- dense all-pairs comparisonを導入しない。
- CPU-only動作を維持する。
- Phase 6の公平比較条件で物理エネルギーが非退行、または品質向上を考慮した`joule_per_success`が改善する。

一つでも重大条件を満たさない場合、固定SNN経路を維持し、不採用結果を記録する。

## 10. 主要なリスク

### 前処理器への能力移転

高性能な形態素解析器、構文解析器、LLMから正解に近い特徴を受け取ると、SNNの能力ではなく前処理器を評価することになる。raw、dictionary-assisted、external-assistedの三条件を分ける。

### イベント爆発

一語を多数チャネルへ展開すると疎性を失う。チャネルgate、最大署名幅、最大残響数、最大共鳴比較数を設定する。

### 人工的な時間引き延ばし

音素やモーラへ展開するだけで系列長が増え、学習が容易になったように見える可能性がある。同一情報量、同一イベント予算、同一wall-clock境界の対照を用意し、単なる反復符号化との差を測る。

### 軸間同期の誤結合

同じ時間窓に存在するだけのイベントを同一概念として結ぶ危険がある。source identity、反復整合、negative pair、競合候補、信頼度を用い、単発同期ではdurable linkを作らない。

### プラットフォーム、著作権、同意

公開視聴可能であることは、media download、長期保存、再配布、学習利用、顔や声の処理への包括的許諾を意味しない。sourceごとに取得方法、license、attribution、consent、retention、deletionを記録し、不明なsourceはmetadata-onlyまたは拒否とする。

### 字幕と映像の擬似対応

字幕の発話内容と画面上の対象が一致しない場合が多い。off-screen narration、編集映像、字幕遅延、複数対象を含むnegative alignmentを用意し、単一windowの共起を正例にしない。

### 過剰圧縮と再解析不能

高levelイベントだけを残すと、抽出器のbiasや誤認識を後から検証できない。bounded raw audit windows、uniform calibration sample、extractor version、raw-to-event lineageを保持し、抽出器変更時に再評価できるようにする。

### Surprise bias

高予測誤差だけを保存すると、通常状態、class frequency、no-change、成功例が失われる。またsensor noiseやadversarial noveltyがmemoryを占有し得る。surpriseを有界な一要素とし、coverage、safety、source reliability、negative example、costを含むbalanced quotaを使う。

### 時刻情報の欠落

相対delayだけでは複数センサーのclock drift、再現、source segmentの特定、削除伝播を扱えない。学習にはrelative delayを使いながら、source-scoped anchorを保持する。

### 反応信号の誤用

笑い、拍手、再生数、engagementは文化、編集、観客構成、演出に左右される。反応タイミングは予測対象にはできるが、意味の正しさや学習報酬として無条件に使用しない。

### 相関と因果の混同

STDPは時間的順序を学習できるが、それだけで因果を証明しない。因果イベントは `causal_hypothesis` として保持し、反例や検証なしにobservedへ変換しない。

### 動的モードの常時発振

概念を振動として保持し続けるとidle energyが増える。入力駆動の短命な減衰モード、発火停止条件、再活性化hintを用い、idle spike countを採用指標にする。

### 位相衝突

長文や入れ子節で位相スロットが衝突する。節ごとの局所スロット、timeout、競合検出、位相なしfallbackを用いる。

### 誤った概念結晶

頻出する誤情報や自己生成データを固定する危険がある。source-aware verification、複数revision、contradiction quarantine、忘却を必須にする。

### 生物学的比喩の過剰解釈

内的音声、共鳴、位相、結晶化は工学的機構の名称であり、人間脳との同一性を意味しない。

### 省電力性の消失

精度向上のために常時多チャネル化、長い残響、過剰な比較を行うとSNNの意義が失われる。物理計測前に優位性を主張しない。

## 11. 実装順序

1. raw surface eventだけを使う固定3-tier echo fieldを作る。
2. 単一減衰と固定多時間尺度SNNを対照にする。
3. 表層、音韻、意味の非同期軸bindingを、単純連結表現との比較として追加する。
4. source-backed own-latent意味特徴を任意チャネルとして追加する。
5. role hypothesisと係り受け候補を追加する。
6. verified crystallizationをPhase 17、18へ接続する。
7. 入力駆動のbounded dynamic semantic modeを静的疎署名とのablationとして試す。
8. 位相bindingを独立ablationとして試す。
9. 音韻経路を曖昧性課題だけに追加する。
10. 因果軸を未検証仮説型として追加し、誤昇格率を測る。
11. rights-clearedなaudio + transcript + timestamp pilotを作り、transcript-onlyとtimestamp shuffleを比較する。
12. change event、bounded raw audit window、uniform baseline reservoirを持つeventizerを比較する。
13. balanced surprise retentionをsurprise-onlyとuniform retentionに対して評価する。
14. bounded episodeとsparse temporal relation graphを追加し、causal false-promotionを測る。
15. pilotが通過した後だけsparse visual eventsとconversation timingを追加する。
16. 固定時定数の限界が確認された場合だけPhase 19と接続する。
17. 機能評価を通過した後、Phase 6条件で物理エネルギーを測定する。

## 12. 結論

Semantic Echo Fieldは、テキストを単なる記号列から疎な時間イベントへ再構成し、SNNが得意とする遅延、減衰、局所共鳴、オンライン適応を言語処理へ利用する提案である。最も重要な設計判断は、Attentionを模倣しないこと、全文を活動状態として保持しないこと、外部解析済み構造をSNN自身の能力と混同しないこと、流動状態と永続概念を分離することである。

この方式が価値を持つのは、言語精度を上げるだけでなく、固定SNNより優れ、単純な多時間尺度対照を超え、疎性と物理エネルギー効率を維持した場合に限られる。失敗時には、複雑性を正当化せず固定SNNへ戻る。この反証可能性を含めて、SARA Engineの独自研究方向として検討する価値がある。

## 参考文献

1. Zhu, R.-J., Zhao, Q., Li, G., and Eshraghian, J. K. "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks." arXiv:2302.13939, 2023. <https://arxiv.org/abs/2302.13939>
2. Bellec, G., Salaj, D., Subramoney, A., Legenstein, R., and Maass, W. "Long short-term memory and learning-to-learn in networks of spiking neurons." arXiv:1803.09574, 2018. <https://arxiv.org/abs/1803.09574>
3. Bellec, G. et al. "A solution to the learning dilemma for recurrent networks of spiking neurons." Nature Communications 11, 3625, 2020. <https://doi.org/10.1038/s41467-020-17236-y>
4. Orchard, J. and Jarvis, R. "Hyperdimensional Computing with Spiking-Phasor Neurons." arXiv:2303.00066, 2023. <https://arxiv.org/abs/2303.00066>
5. Ding, N., Melloni, L., Zhang, H., Tian, X., and Poeppel, D. "Cortical entrainment reflects hierarchical structure building in speech comprehension." Nature Neuroscience 19, 158-164, 2016. <https://doi.org/10.1038/nn.4186>
6. Izhikevich, E. M. "Polychronization: Computation with Spikes." Neural Computation 18(2), 245-282, 2006. <https://doi.org/10.1162/089976606775093882>
7. Eliasmith, C. et al. "A Large-Scale Model of the Functioning Brain." Science 338(6111), 1202-1205, 2012. <https://doi.org/10.1126/science.1225266>
8. Hasani, R. et al. "Liquid Time-constant Networks." arXiv:2006.04439, 2020. <https://arxiv.org/abs/2006.04439>
9. Perrinet, L. U. "Working Memory in a Recurrent Spiking Neural Network With Heterogeneous Synaptic Delays." arXiv:2604.14096, 2026. <https://arxiv.org/abs/2604.14096>
10. YouTube Data API. "Captions: download." Accessed 2026-06-12. <https://developers.google.com/youtube/v3/docs/captions/download>
11. YouTube API Services. "Developer Policies." Accessed 2026-06-12. <https://developers.google.com/youtube/terms/developer-policies>
12. YouTube Help. "License types on YouTube." Accessed 2026-06-12. <https://support.google.com/youtube/answer/2797468>
13. Schaul, T., Quan, J., Antonoglou, I., and Silver, D. "Prioritized Experience Replay." arXiv:1511.05952, 2015. <https://arxiv.org/abs/1511.05952>
14. Vitale, A., Renner, A., Nauer, C., Scaramuzza, D., and Sandamirskaya, Y. "Event-driven Vision and Control for UAVs on a Neuromorphic Chip." arXiv:2108.03694, 2021. <https://arxiv.org/abs/2108.03694>

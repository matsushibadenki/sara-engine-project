##Adaptive Modular Hybrid Architecture（AMHA）


**役割ごとに最適な方式を割り当てる**  

```text
Input
  │
  ├─ Token Embedding
  └─ n-gram / Local Pattern Memory
            │
            ▼
┌─────────────────────────────────┐
│      Dynamic Computation Router │
└───────────────┬─────────────────┘
                │
      ┌─────────┼──────────┐
      ▼         ▼          ▼
  DeltaNet   Sparse      Structural
  /Linear    Attention   Reasoning
  Memory     /MLA        Module
   60%        30%        10%
      └─────────┼──────────┘
                ▼
       Multi-path Residual
                │
                ▼
        Hierarchical MoE
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
     Expert   Expert    Shared
       │        │       Expert
       └────────┼─────────┘
                ▼
        Abstraction Layer
                │
        ┌───────┴────────┐
        ▼                ▼
   reusable          ordinary
   modules            features
        │
        ▼
       Output
```

核心は6点です。

**① 普段の処理 → Delta/Linear Attention**

Qwen系の方向です。文章の大部分は厳密な全トークン比較を必要としません。ここをO(N²) Attentionで処理するのは無駄なので、全計算の60〜70%程度を高速な recurrent/linear memory に担当させます。

**② 本当に検索が必要な場所 → Sparse Attention + MLA**

長距離参照、正確な引用、変数の対応関係などだけAttentionを呼びます。

つまり、

> Attentionを「常時動く思考装置」から「必要な時に呼ぶ検索装置」にする。

KV cache圧縮にはMLA系の考え方を利用します。

**③ 知識・技能 → Hierarchical Sparse MoE**

巨大FFNを毎回全部動かさず、

```text
Router
 ├─ Science
 │   ├─ Physics
 │   └─ Biology
 ├─ Language
 └─ Programming
```

のような階層的Expertにします。

重要なのは、固定カテゴリーを人間が決めるのではなく、**学習によってExpertの専門性を自己形成させること**です。

これによって、

**総パラメータ ≫ Active parameter**

にできます。

**④ 現行LLMに不足している部分 → Structural Reasoning Module**

ここが追加したい最大の要素です。

Transformerは、

「AならB、BならC」

をベクトル表現として学習できますが、

```text
A
↓
B
↓
C
```

という**明示的な構造そのものを保持・再利用すること**は得意ではありません。

そこでDLGN的な小さな構造演算系を追加します。

例えば、

```text
AND
OR
XOR
COMPARE
CAUSE
SEQUENCE
LOOP
SELECT
BIND
RECALL
```

のようなprimitiveから始める。

ただし、これを大量に並べません。

**⑤ 最も重要な追加 → 自己抽象化**

primitiveの組み合わせが頻繁に出現したら、

```text
A → B → C → D
```

を、

```text
[M1]
```

という新しい中間モジュールへ圧縮します。

さらに、

```text
M1 + M4 + M7
```

が頻出すれば、

```text
[M23]
```

へ抽象化する。

つまり、

```text
Primitive
   ↓
Micro Module
   ↓
Concept Module
   ↓
Abstract Module
   ↓
Reusable Skill
```

という**階層が学習中に自然形成される**ようにします。

これは現在のTransformer/MoEにかなり不足している部分です。

**⑥ Residualも一本道にしない**

GLM/Kimi系で見え始めているMulti-path / Hyper-connection方向をさらに進めます。

情報を全部同じ層へ押し込まず、

```text
             ┌─ semantic stream
Input ───────┼─ memory stream
             ├─ structural stream
             └─ exact-information stream
                       │
                       ▼
                     Merge
```

とします。

これによって深層化した際の情報消失を減らします。

最終的には、モデル内部を一枚岩のニューラルネットワークではなく、

**高速直感系 + 精密検索系 + 専門知識系 + 構造推論系 + 抽象化系**

に分けるわけです。

非常に簡潔に表現すると、

> **Linear/Delta Memory + Sparse MLA Attention + Hierarchical MoE + Multi-path Residual + DLGN Structural Reasoning + Self-Abstraction**

です。

そして、私は最後の **Self-Abstraction** が一番重要だと考えています。

現在のLLMはモデルが100Bから1Tになっても、基本的には「同じ種類の層をたくさん積む」方向です。しかし知能の高度化には、単純な演算要素を増やすだけでなく、

**演算 → パターン → モジュール → 概念 → 概念を使った新しい概念**

という階層形成が必要なのではないか、という仮説です。

これが成功すれば、以前話していた「低ビット化で失った自由度を構造的自由度へ変換する」ことにもつながります。**重みの精度やDense parameterを増やす代わりに、経路・専門化・構造・抽象階層を増やして知能を上げる**アーキテクチャです。

# **SARA Engine: 自己組織化臨界と樹状突起計算に基づく次世代脳型認知アーキテクチャ**

## **概要 (Abstract)**

現在の人工知能研究は、Transformerに代表される大規模言語モデル（LLM）が主流となっている。しかし、これらのモデルは静的な行列演算と誤差逆伝播法（Backpropagation）に依存し、膨大な計算資源（GPU）を必要とする。本稿で提案する「SARA Engine」は、行列演算と誤差逆伝播を完全に排除し、局所的なスパイク伝達（イベント駆動型メッセージパッシング）のみで駆動する生体模倣アーキテクチャである。Liquid State Machine (LSM) を基盤とし、樹状突起計算、多階層可塑性（STP/STDP/構造可塑性）、自己組織化臨界（Edge of Chaos）、神経振動（Oscillation）、そして Global Workspace Theory を統合することで、自律的な概念形成と意思決定が可能な認知エージェントの設計図を提示する。

## **1\. はじめに (Introduction)**

SARA Engine の根底にある哲学は、「記憶は静的な重み（Weight）に宿るのではなく、システムの動的な状態空間（Dynamics）に宿る」という Liquid Computing の思想である。

本プロジェクトは以下の制約のもとに設計されている。

* **脱・誤差逆伝播法**: 全てローカルな学習ルール（STDPや三因子学習）で更新する。  
* **脱・行列演算 (![][image1] 計算)**: ![][image2] のAttentionを廃し、スパースなグラフ上のメッセージパッシングで実行する。  
* **脱・GPU依存**: 純粋なPythonによるイベント駆動で軽量に動作する。

## **2\. コア・ダイナミクス：樹状突起を持つLiquid Reservoir**

SARA Engine の中核は、連続時間で状態が変化する Liquid State Machine (LSM) である。

### **2.1 樹状突起計算 (Dendritic Computation)**

従来の点(Point)ニューロンモデルとは異なり、各ニューロンは複数の「樹状突起枝 (Dendritic Branches)」を持つ。

シナプス入力は特定の枝に集約され、局所的な閾値を超えた場合のみ超線形に増幅される（Dendritic Spike）。

これにより、1つのニューロンが「AND/ORゲート」のような非線形特徴検出器として機能し、ニューロン数の数倍に相当する実質的な計算ユニット（MLP相当）を獲得する。

### **2.2 シナプス・クラスタリング**

学習過程で、同時期に活動するシナプスが同じ枝に集まるように配線が自己組織化される。これにより、文脈（Context）の多義性が物理的な枝の違いとして表現される。

## **3\. 多階層可塑性 (Metaplasticity)**

ネットワークは複数の異なる時間スケールで学習・適応を行う。

### **3.1 短期記憶：STP (Short-Term Plasticity)**

Tsodyks-Markramモデルに基づき、ミリ秒スケールの記憶を担う。

* **Facilitation (促通 ![][image3])**: 連続発火で放出確率が上がる。文脈の保持（ワーキングメモリ）。  
* **Depression (抑圧 ![][image4])**: 神経伝達物質が枯渇し、応答が弱まる。暴走の防止と変化への適応。

### **3.2 長期記憶：STDP (Spike-Timing-Dependent Plasticity)**

シナプス前と後の発火タイミング（因果関係）に基づき、秒〜分スケールで重みを更新する局所的なヘブ則（Hebbian Learning）。

### **3.3 シナプス遅延可塑性 (Synaptic Delay Plasticity)**

スパイクの伝播遅延（Delay）自体を学習パラメータとする。到着タイミングを同期させることで、「ポリクロナイゼーション (Polychronization)」を引き起こし、時間的な発火パターンを記憶する。

### **3.4 構造的可塑性 (Structural Plasticity)**

数時間スケールで、弱いシナプスを刈り込み（Pruning）、新たな接続を生成する（Growth）。これにより、脳特有の「スモールワールド・ネットワーク」トポロジが自然に形成される。

## **4\. 安定性と計算の極大化：自己組織化臨界 (Self-Organized Criticality)**

ネットワークが「カオス（暴走）」と「秩序（沈黙）」の境界、すなわち「カオスの縁 (Edge of Chaos)」にある時、記憶容量 (Memory Capacity) と情報伝播効率が最大化される。

### **4.1 E/I バランスと Daleの法則**

興奮性ニューロン(80%)と抑制性ニューロン(20%)を配置し、Daleの法則（1つのニューロンは1種類の神経伝達物質しか出さない）を厳密に適用する。

### **4.2 恒常性可塑性 (Homeostatic Plasticity)**

各ニューロンが自身の指数移動平均(EMA)発火率を監視し、目標発火率から逸脱した場合に、発火閾値（または入力シナプス全体）を動的にスケーリングする。これにより、システムは自律的に分岐比（Branching Ratio: ![][image5]）を保ち、臨界状態へと自己組織化する。

## **5\. 概念形成と時間遷移**

### **5.1 ニューラル・アセンブリ (Cell Assembly)**

Hebb則により、頻繁に同時発火するニューロン群の相互結合が強まり、1つのアトラクター（安定点）を形成する。これがTransformerの静的Tokenに代わる、SARA Engine における「概念（Concept）」の物理的実体である。部分的な入力からでも概念全体が発火する「パターン補完 (Pattern Completion)」が可能になる。

### **5.2 神経シーケンス (Neural Sequence Dynamics)**

非対称なSTDPと遅延可塑性により、アセンブリ間の時間的な遷移（A → B → C）が回路として彫り込まれる（Synfire Chain）。これにより、モデルは自発的に「未来の予測（Priming）」や「思考の再生（Replay）」を行うことができる。

## **6\. 神経振動による時間的ゲーティング (Neural Oscillations)**

TransformerのAttention機構の代替となるのが、脳波（Theta, Alpha, Beta, Gamma）の同期による情報のゲーティングである。

* **Theta-Gamma Coupling**: ゆっくりとしたTheta波（文脈）の特定の位相に、高速なGamma波（単語/特徴）が乗ることで、情報を時間的にチャンキング（Chunking）する。  
* 周期的な閾値の変動により、必要なタイミングでのみ情報が層を跨いで伝達される（Communication Through Coherence）。

## **7\. 認知アーキテクチャの統合**

無意識のダイナミクスから、明確な意思決定と行動を生み出すループ構造。

### **7.1 グローバル・ワークスペース (Global Workspace)**

LSMが生成する無数の「思考候補（Assemblies）」が、相互抑制ネットワーク（Winner-Take-All）で競争する。勝者となった1つの情報だけが「意識」に上り、行動のトリガーとなったり、システム全体にブロードキャストされたりする。

### **7.2 ドーパミン強化学習 (Dopamine RL / Three-Factor Learning)**

シナプスはSTDPの痕跡である「Eligibility Trace（資格トレース）」を保持する。行動の結果として環境から報酬予測誤差（RPE: ドーパミン信号）を受け取ると、過去のトレースを遡ってシナプスが強化される。これにより、遅延報酬に基づく目標指向の学習が実現する。

### **7.3 予測符号化 (Predictive Coding)**

究極の世界モデル（World Model）の基盤。

モデルは常に「次の入力」を内部で予測（Top-down）し、実際の入力（Bottom-up）との「予測誤差（Surprisal）」を計算する。学習は、この誤差を最小化する方向（Active Inference）でのみ進行する。

## **8\. 結論 (Conclusion)**

SARA Engine は、微分方程式とスパイク・メッセージパッシングによって「時間の流れ」そのものを計算資源として利用する Dynamical System である。

巨大な静的パラメータに頼る現在のAIから脱却し、\*\*「小さくてもカオスの縁で自己組織化し、環境と相互作用して概念を獲得する」\*\*という、真の意味での生物型・自律型AGI（汎用人工知能）の実現に向けた強力な基盤となる。

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAYCAYAAAC8/X7cAAAEo0lEQVR4Xp2XS4hcRRSGu5NojO+gw5h+VN07D0YnuJn2OYqbbJQgRPGBYIyJShIEFReiKDGLgAgRJepGXGh8ohtJQPCxEA1RMYqKBIyJEsmImsVIRMJETfzOrbq3z61b3fbkhzN96z//OXXqVN3qnlrNo54/BM9Vph4T9EFV3yt8vrxHNXs0QPNRgYNz6b8nAR04UJLqGnqRAfrMFNU79HHNC/PLM6DaGLO01WovCfkogvXH2iH5Go3G6bECqkyAsfGxxWmaXmOtXYkt074iWGVpt9oNJvx8ZGTknC5bG2CmWrz6WraASWPtruHh4TO6bD8QTNETSZLsoOi9fG4myRM8H7HGvIdvOAwRjI2NLUbzGfq1YcX4zsb3PfYL9qtxeU/TGvht2EFrzQ98HsB2K99TSZLu4HGBY9x2RftC4keySYy5g+GinKdwaPsF9tPQ0NCZRYBPBL9V/I4J4Blyb0DzB3bCLbSsoUErZG528hJdn2/Ab9R0TyEuwY+NsU9bY+co9jLN52DSO2VyEj3uGCcYdxMcwabD1Bl8NcS9maTJGsmBfRXKyH8t/PMhL4BfZ6z5efnk5KkZEU5E8CqfOJLAqdvtdiPTGLtH83TufvjvcrVylZ7RfCMjPnf7uaa1gPEWFnlzpTiHhcx7jON3Q04UMnnpCJ7BjkqRhSNINDExcRaa49isdjH+gInfUVQGp3F/E/devSVDPm/zC3hdK+E/5rgMdRntzfz7iHlOuQR1Omgf9AlfCCODQqe97oDW0JX9cFvDBWukSbKeAjbKc6fTOcW6F/oY3AXCtVqtJYy/zPWxVPjftW4XK44PfWF3lxxBFiZbn+mM3anoha6QdIPiKi1E8wZ2UU7Je+RymU1+LC/ws90gQbkA/Nto1uGSy3djTpJR4OVduYZTo9npdQ/knpTbyWQvtl0R7pgGcV/rsXReFo7NdDpTUsMW7EatCcEi70Pzb624TmvF/S3n+njpegxA8HKvO5SoO5zxlCyK+3tK6/Vi0F+YyPkvoS6xr/mG3MrnJ81m87xKC9SQW2i11DApN5FulnVfHCfkBe7Ky0jS5GXRsJDVMs6D4eQLQviblFyhLrEb0dzrhwXgrnSLt3vEup44mGMTuz2Tj/UCtvsibik5/DPney3+f2QLlSuDv5mkiw93A8td5Iy9zUt6cYkU1LO5pXiJf6Ygg03I4evcVRFALsNmsU+bDdlGDzfBXdjfpW/BIJ6uHELzYpl1kMJZwGya/QTRse6Zwtf4BazScTFY9/2xPeQzcHwuJcm3CPZLp3l+EtuXJsn7jK8O9RrcJC/xDnykiyP2XHL9iB319hf2WDfKif07eJA5luq+xDYB3e/oHgp5jUUIOtjtiFfSNRsKYpDzj/5wEvxA03AFVXegQHVjSxgdHW0zx1zivzd6oF8KhapsATuwlwke1WRV1gf/I7buxno15OeFfnOwC9czwZ+c+WbORfV6B/TfQKyH5L6KBs2SezwqKI1DRwW9BXLGsVdCPkTvDB5lAbsrP+PNdSW2C92JAdFHzBndzL+A54d8L+hUsbTku4IFrAv5MgZv/0kj2qYBp4vKPPkfoo4DydTZR44AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAAFHUlEQVR4XrWXe4gVVRzH75SRQi+ppdx77zzuXVy1F7i9kf4RIpBAeiJktWWkRBZBUBStfwhRSE/6JyrK6EFCVEJgSoaFFWmkhNBDw9AI+mPDELEtt89vzjkzZ845d+5V6gu/nTnf3/t3Zs7cbTQ0InNjwyb7GvRiFBTfS2uj3rIX3wd9OxkQx+tbYx/Vav8vDJLSt6nfk4bVTNgmy7KL4zie4HrN2NjYKT3MLPeKgW9dl8zjnSWFzG61WrNc3kaNqoCxSZLkBmLehyyIk3gT67dsO3I15RruzUZAMTIycipTu5qgS5A5IZscmm+328MU8lWn0zkzaHACIO+WNE2flXtiL2I9TU0XWfoVyIulh0LtHhJglKAf4riH6xoCP8H9Ia6bsjQ7tzCMSl8ZBjZfpmk27gZEdwa675Ffkd903Jm2GdzzyH7kR2Qvsl14GTCDu0Du2a0LpUHq6FiuEdzXyEMW58DKROJHpIg4Tm5jOcPwJEpiFejnoaGh00oPBfh1kqhhtW3/FRB7JTZ/EGea+/FCoW0ofLHkpqFLCtICuqfx+8Dl4a5AN8V1nqw9RwMSPIPhUZq5rCCrzd+hJzhh83qHDiVJfJWsgwmiPP47xLidSU1j/41rmWbptUngcRPITqL7SHbe5k0EdOslvq2rAMelUnyvBAJ5x7TNjpzQ0Ql8P9x3lmmwSWx2cZFHaruOkw/E0q8l1k02p/n5yKvSXLfbbXc73bbSlFmyLJUB/NNsNs8uSAM5FFAeRI5IE67eYO7o6Ons0jF2YLIgZWeSZDOFvW+ZGlWBVL3X7+b3abpMN1g5EeG38TQM2ZzUI37y2JJjDJ+nsiwdtW2MncTE5saCtLb3QZ3wpUIZgExc2+11+J9ocp1Z+7sXSfH3IKvkXr5lsTpw/oI7TyxarfYs1jtdXwa6Vec0MpV/C90sUSMi5mEOuedchezAFu28oqqsQoqU9wf7jYbD/WRd6MqCCQCbt5H5Zs03bUJyMvHH87U6YF4oPQaHyhjJO7yb2jZUlDINij4qySjy8oqyUS0Xm425XZY+YLiM01UXuri09JvE5lt7LTsng0EO5jXw/iHX2zYe/LAKkVKx2+8RY6tpOIf+fh0TCR3/xo4GzhcbJnQgtU4yuIX6VFxouALaF/t58h5Vlbnvm/nA0vQWrp8FDwgXRZN2t+qeGK8gn7taUfBhjad7HjBRXuTrUgyP1nJbBZeJr7zcvQaM7yrs7i2ZoqArJSayQ8RS+ejFWyDGJzI0lxfF+rz4OLlZ1u6QKHAc/d80sdqoDEbnyskqu5A97OoM0G+QXyEuL4hVc7KL+c+xECqPXA2o/xdirXV5KWAOMol8UX1MItHdhUzR3N2BpyIH+gPIyyVTQv+8msyy6k88g1Q+/KrBpY6qf0vVODMT9ardGTTg8bwUo92JHPlxspr7J5EfkI9pbpHt4CZG/xp+n9ocfmfB7UOOKIkPc33MtpE4+gzYT4zZBVm9ce7DwH+BDCqzf4UFMEM+psitGC/BOHENQsnk/cP+d5mizQdrLRAkFRyVP1afoYblDHGXkOG8Nfn6ImqcRIN7kEdd1eDwC/CZRpDUT4H8B1J8xwNmzoQCFoYPqThdryPBn+YfUAVn0iHH44YfRAaL7JQnyNcK+lYwGC/vGPJGVeUMrgb1WhulJT/xmuTc1+maH+DBOAEqh8NHHuOBKa4ZHh4+x+Ur8MP6CObyGTnd5YB0+TB8/xPGfxiqURutUIVt/gX5diT/LqcYbgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABVElEQVR4Xn1RPUvFMBRteQ8cxMGpQ9N70y+q4PJ+hJO/QAR5i6CTq5OzuPgXxNHB3T/gIjgLDrqJ4OKiOD1P0qZJk9ZL0+bec869J2kUDSIevtvNSPSAK2g34xoP8GTj4Q3x9576Pw/uOB/z6iE1FERtMbQ1ZWkiH5MNbGRC1Fkmdmwl1k/TNBuWi5AyPyOiK2Z+kVKedLyoLIuMib+JeU9XirIgEG6VCMA7gBvTSAmZaZWmqdAFIj5ExwVICwWAsDQzVRPUX3VijcQKuIDoF+JNAyD/YD3J8gzwTEz3poh8G2uFsxxZMqKqqrUOODdiHPpY1TBxC2sf2K7hzwF8wt+lStR1IX/C+oF4hu9dkiTrjg06wG28MdE19o8yl0uQvrAeYO/UdHVOGs9FJmqkM5VVtbaX95yW1/2FIPxiz3MBn+RFKHDTrqNrw8XcGJkzJbAz/wBU+DVFjTIOUwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABWElEQVR4XnVQO07EMBD1iuXTUSAoyGbGzgYhBQqkFJRbUFOwNPR0W3IASiQaQNAgUSFRISEQB6DgClwALsAFKJbnJGsmEzOS7Tdv3ht7bIzpGR/1XoOAA98wUhRA1NBK/hPp3lGRAh5H3B2nTHWt3uMGWYgrJBX2wEizNshQNsUIt7Vuh5j28jxf9GS+ka86a3dbN5RlOW+dfbTW3jPzNRN/ENEpM90BP4O7CI0hOsc68mlRFAsQTZG/ryfJCoQ/WG9NX2PQ5XLmBN5CcQrDsSeI+cQ5uxnEMiCa4GqI2ela56cgeoLhS0rEh/T6+IEziMbAczi/sR5mIuBDmMdVjkFG1RuJJ3jvvsd+YF/LsmzZ/wZgvxIPs+Ea/vIVgisIbzDQAc5P3HbLRC9pmm7XLxAhB4J5aTBIE4/lPH+hBpU1MZ9gQ9olG0YVQqir4qF6t0BooHYpatJfyWMyjf2I8jgAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAADUklEQVR4Xr1VTWgUMRjdbdWKvyjWle3uJNmdZaGiHpZqUURFEbF69qgivQiCICKlqAjiQRER8SxFEEQUL4IHr4JC0Ztg/Tl4rYLiSQ/Wl81MJvmS+dml+iA7mfe9L9/Ll3RaKkUox5OMuRdEoF7LDm9Ch0xZEX3/0LYIs1jwdsGF3rD8TRMRuLKMYq5YQRcmZO78fyCjXtKtdEYhrSlpeg+80tzMcgGNCVdPvGqYfixvvdbLnafB9arIDETRAc75yVqtNmJHc9CrPxPeLmYsQkNBwCYZY48wvjEWLNTr9TGqyUAPUgM9ZXnFigyC4AyMT7CA3cSza54IY3jPlCC/hTja9WErHPJ23cYgJWLQXM7FVI55jUEIBUyEtXqthWd3oAvrqLCLqIC8j9A8DRj7g/wfmD9PKyY4H8eVOFqoXyVpnhPzbpY0fU4WxjEtSLE5UOweTTABzRMYnhRCVKrV6gp06wS4d9jMXbxvSJRlWeeVzVHY5hLztTGP767xF1zwOQiPYxyDkdfgfsPEHjx3NhqNtTQpBjbWkAUoH4atIaw5hQ3MI/4QYwbrfsF60zLuXkLXmWSca2OeGAIXMb52u6HvGT8kE9DJbU4CqRp1epPBWuJms7kRa53CF2Ma5jumzAtdS/3GnYeX7db2RlBYdZhfMXl067BMSIqZy7k7kb/QHkDOA6x1G88JHSQYHh5exYXYGr+TXhAmMY/nDhrYax1JBNz7q+B/jY6OLjP5BHYBGL+ADb+RTeCCn8V8Fvkv8Ye8RYuiPeMEbkC/32eUvkqkmpcdkn+g7XZ7dcx1Op2lKPAJBW6ZWgXlwCzcCsM1WOc+pgOGUG7oCPj3KDqD5uyWDWLqij4zdXkQiflxa2/KKJtHoV1xAO/XMWYhXp4oPS2JIISAnO2zWaUPw3AIsfNo0FusN4c617CJqq3NBnIuS/OoczBhIz9YFJ+14DMK3IHoMeanQS9JhAbS95DA1GTo3ZA+0u5c/q+An48Y35n8hDP2E94+gL+U5JbVFwPk5kqlsrJgbYIiWS5fhFFI47NgdGJxoLqrVtMTF966OqsfZKT1F+ob/2LNHBQs6e06QUbIQvZaxY/S1biMAuHLDhMhNWAgV0Cg13QTXSYbpp7O/wIFq5UgNH8ZMgAAAABJRU5ErkJggg==>
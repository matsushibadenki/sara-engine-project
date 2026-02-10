# **SARA Engine (Liquid Harmony)**

**SARA (Spiking Advanced Recursive Architecture)** は、生物学的脳の「省電力・イベント駆動・自己組織化」を模倣した次世代AIエンジン（SNNベース）です。

現代の深層学習（ANN）が依存する「誤差逆伝播法（BP）」や「行列演算」を完全に排除し、**スパースなスパイク通信のみ**で高度な認識・学習能力を実現しました。

GPUを一切使用せず、CPUのみで動作します。

Current Version: **v35.1 (Code Name: Liquid Harmony)**

## **特徴**

* **No Backpropagation**: 誤差逆伝播法を使用せず、局所的な学習則（Momentum Delta）とリザーバ計算を用いて学習します。  
* **CPU Only & Lightweight**: 高価なGPUリソースを必要としません。標準的なCPU環境で高速に動作します。  
* **Multi-Scale True Liquid Reservoir**: 異なる時間特性（Decay）を持つ3つのリザーバ層を並列配置し、さらに層内再帰結合（Recurrent Connections）を実装。情報の「反響（Echo）」を利用して短期記憶を実現しています。  
* **Sleep Phase**: 学習エポック間に「睡眠フェーズ」を設け、不要なシナプスを物理的に切断（Pruning）することで過学習を抑制します。

## **インストール**

pip install sara-engine

## **クイックスタート**

from sara\_engine import SaraEngine

\# 1\. エンジンの初期化 (入力:784, 出力:10クラス)  
engine \= SaraEngine(input\_size=784, output\_size=10)

\# 2\. データの準備 (ポアソンエンコーディングされたスパイク列)  
\# spike\_train \= \[\[neuron\_idx, ...\], \[\], \[neuron\_idx\], ...\]   
\# ... (データ準備の詳細は examples/train\_mnist.py を参照)

\# 3\. 学習 (GPU不要、CPUで動作)  
\# target\_label: 正解クラスのインデックス  
engine.train\_step(spike\_train, target\_label=1)

\# 4\. 推論  
prediction \= engine.predict(spike\_train)

## **アーキテクチャ (v35.1)**

SARAは脳の皮質構造を模倣し、以下の3つのReservoir層を持っています。

| 層タイプ | ニューロン数 | 減衰率 (Decay) | 役割 | 再帰結合強度 |
| :---- | :---- | :---- | :---- | :---- |
| **Fast** | 1,500 | 0.3 (速い) | エッジ検出・ノイズ処理 | 1.2 (中) |
| **Medium** | 2,000 | 0.7 (中) | 形状・ストロークの統合 | 1.5 (強) |
| **Slow** | 1,500 | 0.95 (遅い) | 文脈・大域的パターンの保持 | 2.0 (最強) |

### **処理フロー**

graph TD  
    Image\[Image / Sensor\] \--\>|Poisson Encoding| Spikes  
    Spikes \--\> Fast\[Fast Reservoir\]  
    Spikes \--\> Med\[Medium Reservoir\]  
    Spikes \--\> Slow\[Slow Reservoir\]  
      
    Fast \<--\> Fast  
    Med \<--\> Med  
    Slow \<--\> Slow  
      
    Fast \--\> Readout  
    Med \--\> Readout  
    Slow \--\> Readout  
      
    Readout \--\>|Momentum Delta| Class

## **推奨パラメータ (Best Practice)**

MNISTタスクにおける黄金比率です。

* **Samples**: 20,000 (最低ライン)  
* **Reservoir Size**: 5,000 neurons (Fast:1500, Med:2000, Slow:1500)  
* **Input Scale**: Fast層には強く(1.0)、Slow層には弱く(0.4)入力する。  
* **Sleep Pruning**: 5% (毎エポック実行を推奨)

## **ライセンス**

MIT License
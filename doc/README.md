# **SARA Engine 🧠⚡**

**Beyond Matrix Multiplication: The Pure SNN AI Engine for Edge Devices.**

SARA (Spiking Artificial Reasoning Architecture) は、既存のLLMが依存している重い行列演算（GPU）や誤差逆伝播法を一切使用せず、「スパイク（0と1の信号）」と「ニューロンの結合」のみを用いて自然言語を推論する次世代の超軽量エッジAIエンジンです。

## **特徴**

* **爆速ローカル推論**: MシリーズMacなどのCPU単体で **1,500〜2,000 tokens/sec** の生成速度を誇ります。  
* **純粋なO(1)検索**: 行列演算の代わりに、疎な分散表現（SDR）とMessagePackを用いた完全一致の辞書アクセス（海馬エンジン）で回答を導き出します。  
* **BPE耐性**: LLM特有のトークン分断や、青空文庫等のフォーマットの違いを文字列レベルで自動補正する機能を搭載。  
* **カスタマイズ容易**: 専用のJSONLデータを使って、一瞬でAIの「性格」や「対話ルール」を上書き学習させることができます。

## **インストール**

ソースコードからインストールし、ターミナルでコマンドを有効化します。

pip install \-e .

## **CLIツールの使い方**

### **1\. チャットの開始**

インストール後、ターミナルから以下のコマンドを打つだけでSARAと対話ができます。

sara-chat

*(※ プロジェクトのルートディレクトリで実行し、models/distilled\_sara\_llm.msgpack が存在することを確認してください)*

### **2\. 対話ルール（性格）の学習**

Q\&A形式の対話データ（JSONL形式）を用意し、SARAに「アシスタントとしての振る舞い」を学習させます。

sara-train data/chat\_data.jsonl

## **Pythonライブラリとしての組み込み**

SARAをWebアプリケーション（FastAPIなど）やDiscord Botに組み込むのは非常に簡単です。

from sara\_engine.inference import SaraInference

\# エンジンの初期化  
sara \= SaraInference(model\_path="models/distilled\_sara\_llm.msgpack")

\# 推論の実行 (確実性を高める場合は temperature=0.0 に設定)  
response \= sara.generate(  
    "You: こんにちは\\nSARA:",   
    max\_length=100,   
    top\_k=1,   
    temperature=0.0,  
    stop\_conditions=\["\\n"\]  
)

print(response)  

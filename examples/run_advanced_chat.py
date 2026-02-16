_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_advanced_chat.py",
    "//": "タイトル: アドバンスド SARA チャット (新機能統合デモ)",
    "//": "目的: STDP事前学習、SpikeAttention、Temporal Coding、および確率的デコーディングを組み合わせた、行列演算・BPなしの高度なチャットシステムの実現。英語のボキャブラリを追加。"
}

import os
import sys

# プロジェクトルート付近のsrcディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.memory.sdr import SDREncoder
from sara_engine.models.gpt import SaraGPT
from sara_engine.learning.stdp import STDPPretrainer
from sara_engine.core.attention import SpikeAttention
from sara_engine.core.temporal import TemporalEncoder

def run_advanced_demo():
    print("=== SARA Engine Advanced Chat Demo ===")
    print("初期化中... (STDP, SpikeAttention, Temporal Coding, Stochastic Decoding)\n")
    
    # 1. エンコーダとトークナイザーの準備
    encoder = SDREncoder(input_size=1024, density=0.05, use_tokenizer=True, apply_vsa=False)
    
    # 事前学習用のコーパス（英語の挨拶などを追加）
    corpus = [
        "こんにちは",
        "おはよう ございます",
        "私 の 名前 は sara です",
        "あなた の 名前 は 何 です か",
        "人工知能 は 人間 の 脳 を 模倣 し て 作ら れ まし た",
        "スパイキング ニューラル ネットワーク は 行列 演算 を 使わ ず に 省電力 で 動作 し ます",
        "猫 は とても かわいい 動物 です",
        "犬 は 走る の が 速い です",
        "私 は 学習 する こと が 好き です",
        "sara は 賢い ＡＩ です",
        "今日 の 天気 は 晴れ です",
        "誤差 逆 伝播 法 を 使わ ない 学習 は 生物 学 的 に 妥当 です",
        "海馬 は 短期 記憶 を 長期 記憶 に 変換 し ます",
        "時間 的 な 文脈 は 会話 において 重要 です",
        "hello",
        "good morning",
        "hello sara",
        "sara is a smart ai",
        "i love learning",
        "how are you"
    ]
    
    print("1/3: トークナイザーの学習...")
    encoder.tokenizer.train(corpus)
    
    # 2. 各モジュールの初期化
    gpt = SaraGPT(encoder)
    attention = SpikeAttention(input_size=1024, hidden_size=2048, num_heads=4)
    temporal = TemporalEncoder(input_size=1024, target_density=0.05)
    
    # 3. STDPによる事前学習
    print("2/3: STDP (スパイクタイミング依存可塑性) による自己回帰の事前学習を開始...")
    pretrainer = STDPPretrainer(window_size=4, a_plus=1.5, a_minus=0.5)
    pretrainer.pretrain(gpt, corpus)
    print(f"  -> シナプス結合が自己組織化されました。 (総アクティブニューロン: {len(gpt.synapses)})")
    
    print("3/3: 準備完了。\n")
    print("Commands: 'exit' または 'quit' で終了します。")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
        
        # 入力文のトークン化
        token_ids = encoder.tokenizer.encode(user_input)
        
        context_sdr = []
        # 入力シーケンスを順番に処理し、長期的な文脈を捉える
        for position, tid in enumerate(token_ids):
            base_sdr = encoder._get_token_sdr(tid)
            
            # 【Temporal Coding】単語のSDRに「文章内の位置」の情報を埋め込む
            temporal_sdr = temporal.encode(base_sdr, position)
            
            # 【SpikeAttention】位置情報を含んだSDRでアテンションを計算し、関連する過去の文脈を引き出す
            current_context = attention.compute(temporal_sdr)
            
            if current_context:
                # 抽出された文脈スパイクを蓄積（スパイクの重ね合わせ）
                context_sdr = sorted(list(set(context_sdr) | set(current_context)))
        
        print("SARA: ", end="", flush=True)
        
        # 【Stochastic Decoding】STDPで学習した重み、Attentionの文脈バイアス、そして確率的デコーディングによる生成
        response = gpt.generate(
            prompt=user_input, 
            context_sdr=context_sdr, 
            max_tokens=20, 
            temperature=0.8,       # 膜電位へのノイズ注入量（0.8で自然なゆらぎを持たせる）
            top_k=40,              # 発火候補の絞り込み
            top_p=0.9,             # 累積確率による足切り
            repetition_penalty=1.2 # 反復の抑制
        )
        
        if not response:
            print("...")
        else:
            print(response)

if __name__ == "__main__":
    run_advanced_demo()
_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_learning.py",
    "//": "タイトル: スパイキングニューラルネットワーク学習デモ",
    "//": "目的: リファクタリングされたCorticalColumnとSpikeAttentionを用いた、SDRベースのSNNコア機能の動作確認。Attentionの時系列処理バグを修正。"
}

import random
from sara_engine.core.cortex import CorticalColumn
from sara_engine.core.attention import SpikeAttention

def generate_random_sdr(size: int, density: float = 0.05) -> list[int]:
    """スパースな発火表現(SDR)を生成する"""
    target_len = int(size * density)
    return sorted(random.sample(range(size), target_len))

def main():
    print("=== SNN Core 学習デモンストレーション ===")
    
    input_size = 1024
    hidden_size = 2048
    seq_length = 5
    compartment = "general"
    
    print("SNNコンポーネントを初期化中...")
    try:
        cortex_column = CorticalColumn(
            input_size=input_size, 
            hidden_size_per_comp=hidden_size, 
            compartment_names=[compartment]
        )
        attention = SpikeAttention(input_size=hidden_size, hidden_size=500, memory_size=60)
    except ImportError as e:
        print(f"モジュールの読み込みに失敗しました: {e}")
        return
    
    print("\n[順伝播(潜在連鎖)とアテンションの計算]")
    prev_hidden = []
    
    for t in range(seq_length):
        # SDRとしてランダムな入力スパイク列を生成
        current_input = generate_random_sdr(input_size)
        
        # CorticalColumnのforward_latent_chainを実行 (内部でSTDP学習が走る)
        active_hidden = cortex_column.forward_latent_chain(
            active_inputs=current_input, 
            prev_active_hidden=prev_hidden, 
            current_context=compartment, 
            learning=True, 
            reward_signal=1.0
        )
        
        # 修正箇所: ループ内で毎ステップAttentionを計算し、シーケンス履歴を構築する
        # (内部でupdate_memory相当の処理が行われ、過去の文脈とのOverlapが計算される)
        attn_signal = attention.compute(active_hidden)
        
        prev_hidden = active_hidden
        
        print(f"Step {t+1}:")
        print(f"  - 皮質発火ニューロン数 = {len(active_hidden)}")
        print(f"  - Attentionシグナル生成数 = {len(attn_signal)}")
    
    print("\n[皮質カラムの膜電位・閾値状態 (スナップショット)]")
    print("※発火直後のニューロンは電位がリセットされるため、アクティブ(v>0)が少なくなるのは正常な挙動です。")
    states = cortex_column.get_compartment_states()
    for comp, state in states.items():
        print(f"コンパートメント '{comp}': 待機中ニューロン(v>0)={state['active_neurons']}, 平均閾値={state['avg_threshold']:.4f}")
        
    print("\nSNNデモが完了しました。")

if __name__ == "__main__":
    main()
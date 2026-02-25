_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_sara_board.py",
    "//": "ファイルの日本語タイトル: Sara-BoardとAttentionのデモ",
    "//": "ファイルの目的や内容: SNNModuleに移植されたAttention層と、Sara-Boardによるラスタープロット(可視化)の連携テスト。"
}

import os
from sara_engine import nn
from sara_engine.utils.board import SaraBoard

def main():
    print("--- Testing SNNModule Attention & Sara-Board ---")
    
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
    board = SaraBoard(log_dir=workspace_dir)
    
    # ネットワークの構築 (Linear -> Attention -> Linear)
    model = nn.Sequential(
        nn.LinearSpike(in_features=64, out_features=128, density=0.2),
        nn.SpikeSelfAttention(embed_dim=128, density=0.1, context_size=10),
        nn.LinearSpike(in_features=128, out_features=64, density=0.2)
    )
    
    # 20ステップのストリームデータをシミュレート
    for step in range(20):
        input_spikes = [(step * 3 + i) % 64 for i in range(8)]
        
        x = input_spikes
        # 各レイヤーの出力を個別にトラッキングしてSara-Boardに記録する
        for name, layer in model._modules.items():
            x = layer(x, learning=True)
            layer_type = layer.__class__.__name__
            
            # 発火イベントの記録
            board.log_spikes(f"{name}_{layer_type}", x)
            
        print(f"Step {step+1}: Final Output Spikes = {x}")
        
    # ラスタープロットの描画と保存
    print("\nGenerating Raster Plots...")
    board.plot_raster("0_LinearSpike", save_name="layer0_raster.png")
    board.plot_raster("1_SpikeSelfAttention", save_name="layer1_attn_raster.png")
    board.plot_raster("2_LinearSpike", save_name="layer2_raster.png")
    
    print(f"Sara-Board plots have been saved to {workspace_dir}")

if __name__ == "__main__":
    main()
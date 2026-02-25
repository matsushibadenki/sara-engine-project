_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_predictive_coding.py",
    "//": "ファイルの日本語タイトル: 予測符号化(Predictive Coding)デモ",
    "//": "ファイルの目的や内容: 予測符号化層を用いることで、同じ入力パターンに対するスパイク発火量(消費エネルギー)が学習に伴って減少していく「ハビチュエーション(慣れ)」を検証する。"
}

from sara_engine import nn

def main():
    print("--- Testing Predictive Coding vs Standard SNN ---")
    
    # 標準的なSNN
    standard_model = nn.Sequential(
        nn.LinearSpike(in_features=64, out_features=32, density=0.3),
        nn.LinearSpike(in_features=32, out_features=16, density=0.3)
    )
    
    # 予測符号化(省エネ)SNN
    predictive_model = nn.Sequential(
        nn.PredictiveSpikeLayer(in_features=64, out_features=32, density=0.3),
        nn.PredictiveSpikeLayer(in_features=32, out_features=16, density=0.3)
    )
    
    # 固定の繰り返しパターン (A -> B -> C の反復)
    # 予測符号化モデルは学習が進むにつれ「次はこのスパイクが来る」と予測できるようになり、発火を自ら抑制する。
    pattern = [
        [2, 5, 12, 44],  # 入力 A
        [3, 8, 15, 50],  # 入力 B
        [7, 10, 22, 60]  # 入力 C
    ]
    
    print("\n[Standard SNN Model]")
    standard_model.reset_state()
    for epoch in range(5):
        total_spikes = 0
        for step, inp in enumerate(pattern):
            out = standard_model(inp, learning=True)
            total_spikes += len(out)
        print(f"Epoch {epoch+1}: Total Output Spikes = {total_spikes}")
        
    print("\n[Predictive Coding SNN Model]")
    predictive_model.reset_state()
    for epoch in range(5):
        total_spikes = 0
        for step, inp in enumerate(pattern):
            out = predictive_model(inp, learning=True)
            total_spikes += len(out)
        print(f"Epoch {epoch+1}: Total Output Spikes = {total_spikes}")
        
        if epoch == 4:
            print(" -> Notice how the number of spikes decreases as the model learns to predict the pattern (Habituation)!")

if __name__ == "__main__":
    main()
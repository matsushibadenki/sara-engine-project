_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_multimodal_memory.py",
    "//": "タイトル: マルチモーダル記憶・睡眠定着デモ",
    "//": "目的: 視覚などの複数のモダリティからの入力をエンコードし、長期記憶に保存後、睡眠プロセスを通じて記憶を定着させる"
}

import numpy as np

def create_dummy_image_features():
    # 最新のImageSpikeEncoderは1次元のList[float]を受け取る仕様に適合
    return np.random.rand(1024).tolist()

def main():
    print("=== マルチモーダル記憶と睡眠による定着デモ ===")
    
    print("モジュールを初期化中...")
    try:
        from sara_engine.agent.sara_agent import SaraAgent
        # デモ用に軽量な次元数で初期化
        agent = SaraAgent(input_size=1024, hidden_size=2048)
    except ImportError as e:
        print(f"モジュールの読み込みに失敗しました: {e}")
        return
        
    # 1. マルチモーダルデータのエンコードと記憶
    print("\n[フェーズ1: 視覚データのエンコードと海馬での短期記憶]")
    img_features = create_dummy_image_features()
    label = "リンゴ"
    
    # SaraAgentのperceive_imageを使用してSDR化と海馬への保存を実行
    result = agent.perceive_image(img_features, label)
    print(result)
    
    # 2. 連想テスト
    print("\n[フェーズ2: テキストからの記憶連想]")
    print(f"入力: '{label}' を検索します...")
    response = agent.chat(label, teaching_mode=False)
    print(f"検索結果:\n{response}")
    
    # 3. 睡眠による記憶の定着
    print("\n[フェーズ3: 睡眠フェーズ（記憶の定着）]")
    print("睡眠プロセスを開始し、海馬から皮質へ記憶を統合します...")
    
    sleep_result = agent.sleep(consolidation_epochs=3)
    print(sleep_result)
    
    print("\nデモが完了しました。")

if __name__ == "__main__":
    main()
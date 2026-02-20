"""
{
    "//": "ディレクトリパス: examples/demo_crossmodal_recall.py",
    "//": "タイトル: クロスモーダル連想記憶（視覚→テキスト）デモ",
    "//": "目的: GPUや行列演算に依存せず、スパース分散表現（SDR）の重なりを用いて画像特徴量からエピソード記憶を想起するプロセスをテストする。"
}
"""

import sys
import os
import random

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.agent.sara_agent import SaraAgent

def generate_dummy_image(seed_value: float, size: int = 2048) -> list[float]:
    """行列演算に依存しない、ダミーの視覚特徴量生成器"""
    rng = random.Random(seed_value)
    return [rng.uniform(0.0, 1.0) for _ in range(size)]

def add_noise(image: list[float], noise_level: float = 0.1, seed_value: float = 0) -> list[float]:
    """ノイズ耐性テストのために特徴量にランダムな揺らぎを加える"""
    rng = random.Random(seed_value)
    return [max(0.0, min(1.0, v + rng.uniform(-noise_level, noise_level))) for v in image]

def main():
    print("=== SaraAgent クロスモーダル記憶想起デモ ===")
    print("※GPUや行列演算を使わず、SDRのスパースな重なりだけで視覚とテキストエピソードをリンクさせます。\n")
    
    agent = SaraAgent()
    
    # ダミー画像の生成
    dog_image = generate_dummy_image(42.0)
    cat_image = generate_dummy_image(99.0)
    
    print("--- 1. 視覚と概念のバインディング（学習モード） ---")
    print("ユーザーが画像を見せながら、それが何であるかを教えます。")
    print("User: [犬の画像を見せる]")
    res = agent.perceive_image(dog_image, "愛犬のポチ")
    print(f"Sara: {res}")
    
    print("\nUser: [猫の画像を見せる]")
    res = agent.perceive_image(cat_image, "近所の白猫")
    print(f"Sara: {res}")
    
    print("\n--- 2. テキストエピソードの追加学習 ---")
    agent.chat("私 の 愛犬 の ポチ は フリスビー が 大好き です", teaching_mode=True)
    agent.chat("近所 の 白猫 は いつも 日向ぼっこ を し て い ます", teaching_mode=True)
    
    print("\n--- 3. クロスモーダル連想想起（推論モード） ---")
    print("ノイズの乗った画像を提示し、PFCと海馬が視覚から記憶を引っ張り出せるかテストします。")
    
    # 15%のノイズを乗せたテスト画像
    noisy_dog = add_noise(dog_image, 0.15, 100)
    noisy_cat = add_noise(cat_image, 0.15, 200)
    unknown_img = generate_dummy_image(777.0)
    
    print("\nUser: [ノイズが乗った犬の画像を提示] これは何の画像か分かりますか？")
    res1 = agent.recognize_image(noisy_dog, "これ は 何 です か ？")
    print(f"Sara:\n{res1}")
    
    print("\nUser: [ノイズが乗った猫の画像を提示] では、こちらは？")
    res2 = agent.recognize_image(noisy_cat, "こちら は 何 の 画像 です か ？")
    print(f"Sara:\n{res2}")
    
    print("\nUser: [全く未知の画像を提示]")
    res3 = agent.recognize_image(unknown_img, "これ は 見 た こと が あり ます か ？")
    print(f"Sara:\n{res3}")

if __name__ == "__main__":
    main()
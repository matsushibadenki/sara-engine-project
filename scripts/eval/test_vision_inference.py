# ディレクトリパス: scripts/eval/test_vision_inference.py
# ファイルの日本語タイトル: 視覚連想推論テストスクリプト
# ファイルの目的や内容: 学習済みのSNNモデルを使用して、入力画像から関連するテキスト（キャプション）を連想・復元できるか確認する。

import os
import msgpack
import numpy as np
from PIL import Image
from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.encoders.vision import ImageSpikeEncoder
from transformers import AutoTokenizer

def run_vision_inference(image_path, model_path):
    """
    画像を入力し、SNNの直接記憶マップから連想されるテキストを出力する。
    """
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return

    print(f"--- 視覚連想テスト実行中: {image_path} ---")
    
    # 1. モデルとエンコーダーの初期化（学習時と同じ設定）
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    vision_encoder = ImageSpikeEncoder(output_size=8192)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    # 2. モデルデータの読み込み
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    raw_map = state.get("direct_map", {})
    # キーを復元
    student._direct_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
    print(f"✅ {len(student._direct_map)} 件の記憶パターンをロードしました。")

    # 3. 入力画像のSDR化
    try:
        # 学習時と同じ前処理 (64x64, グレースケール, 0.0-1.0)
        image = Image.open(image_path).convert('L').resize((64, 64))
        pixel_values = list(np.array(image).flatten() / 255.0)
        
        vision_sdr = vision_encoder.encode(pixel_values)
        vision_key = student._sdr_key(vision_sdr)

        # 4. 連想記憶の引き出し
        print("\nSARAの連想結果:")
        if vision_key in student._direct_map:
            # 重みの高い順にトークンを取得
            token_weights = student._direct_map[vision_key]
            # 上位のトークンをデコード
            sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)
            
            # トークンIDのリストを作成
            token_ids = [t[0] for t in sorted_tokens]
            
            # デコードして表示
            decoded_text = tokenizer.decode(token_ids)
            print(f"✨ 認識内容: {decoded_text}")
            
            # 詳細（重み付き）
            print("\n[内部発火強度]")
            for tid, weight in sorted_tokens[:5]:
                print(f" - '{tokenizer.decode([tid])}': {weight:.2f}")
        else:
            print("🤔 その画像パターンに対する記憶（連想結合）が見つかりませんでした。")
            print("（学習した画像と特徴が大きく異なるか、特徴抽出の閾値を超えていない可能性があります）")
            
    except Exception as e:
        print(f"❌ 推論中にエラーが発生しました: {e}")

if __name__ == "__main__":
    # 直接実行用
    run_vision_inference("data/raw/visual/images/apple.jpg", "models/distilled_sara_llm.msgpack")
# ディレクトリパス: scripts/train/train_vision.py
# ファイルの日本語タイトル: 視覚・言語マルチモーダル連想学習スクリプト
# ファイルの目的や内容: ImageSpikeEncoderを使用して画像とテキストを連想学習させる。保存時の変数名エラー(NameError)を修正。

import os
import csv
import torch
import tqdm
import numpy as np
from PIL import Image
from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.encoders.vision import ImageSpikeEncoder
from transformers import AutoTokenizer

def train_vision_association(csv_path, image_dir, model_path):
    print("Initializing Multi-modal Training Environment...")
    
    # SNNモデルとエンコーダーの準備 (8192ニューロン)
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    vision_encoder = ImageSpikeEncoder(output_size=8192)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    # 既存モデルの読み込み
    if os.path.exists(model_path):
        print(f"Loading existing memory: {model_path}")
        student.load_memory(model_path)
    else:
        student._direct_map = {}

    if not os.path.exists(csv_path):
        print(f"❌ キャプションファイルが見つかりません: {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data_pairs = list(reader)

    print(f"🚀 {len(data_pairs)} 件のペアを学習します...")

    for item in tqdm.tqdm(data_pairs, desc="Vision-Text Pairing"):
        img_name = item['image_file']
        caption = item['caption']
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            continue

        try:
            # 1. 画像を読み込み、グレースケールの数値リスト(0.0-1.0)に変換
            image = Image.open(img_path).convert('L').resize((64, 64))
            pixel_values = list(np.array(image).flatten() / 255.0)
            
            # 2. 特徴量からSDRを生成
            vision_sdr = vision_encoder.encode(pixel_values)
            vision_key = student._sdr_key(vision_sdr)

            # 3. テキストと紐付け (同時発火の原理)
            tokens = tokenizer.encode(caption, add_special_tokens=False)
            if vision_key not in student._direct_map:
                student._direct_map[vision_key] = {}
            
            target_map = student._direct_map[vision_key]
            for token_id in tokens:
                # 画像SDRからテキストトークンへの結合荷重を強化
                target_map[token_id] = min(target_map.get(token_id, 0.0) + 800.0, 2000.0)
                
        except Exception as e:
            print(f"❌ エラー ({img_name}): {e}")

    print("Saving updated memory...")
    student.save_memory(model_path)
    print("✨ 視覚連想学習が完了しました！")

if __name__ == "__main__":
    train_vision_association(
        "data/raw/visual/text/captions.csv",
        "data/raw/visual/images",
        "models/distilled_sara_llm.msgpack"
    )

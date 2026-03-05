# ディレクトリパス: scripts/eval/chat_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 推論・対話スクリプト
# ファイルの目的や内容: 文字単位で学習されたモデルに合わせて、入出力をUnicode文字単位で処理するよう修正。推論を安定化。

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel

def decode_tokens(tokens: list[int]) -> str:
    """Unicodeコードポイントのリストを文字列に変換する"""
    chars = []
    for t in tokens:
        # 有効なUnicodeコードポイントの範囲内かチェック
        if 0 <= t <= 0x10FFFF:
            chars.append(chr(t))
    return "".join(chars)

def chat_loop(model_dir: str):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Inference (Character-Level)")
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print("Error: Pre-trained model not found.")
        return

    print("Waking up SARA... Loading synaptic weights...")
    model = SpikingTransformerModel.from_pretrained(model_dir)
    print("SARA is ready! (Type 'quit' or 'exit' to stop)\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            # 入力文字列を文字のID（Unicodeコードポイント）に変換
            input_tokens = [ord(c) for c in user_input]
            
            print("SARA is thinking...", end="\r")
            
            # 修正箇所: temperatureを指定してサンプリングのランダム性を抑え、精度を向上
            # fire_thresholdを調整して不十分な電位での生成を抑制
            generated_tokens = model.generate(
                input_ids=input_tokens, 
                max_length=100,
                temperature=0.15,
                fire_threshold=10.0
            )
            
            response_text = decode_tokens(generated_tokens)
            print(f"SARA: {user_input}{response_text}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    chat_loop(model_dir=SAVE_DIRECTORY)
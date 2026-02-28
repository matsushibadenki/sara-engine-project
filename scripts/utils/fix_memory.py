# ディレクトリパス: scripts/fix_memory.py
# ファイルの日本語タイトル: SNN記憶修正（ピンポイント忘却）スクリプト
# ファイルの目的や内容: 特定の間違った文脈と回答を指定し、MessagePack内の重みを削除または減衰させることで記憶を修正する。

import msgpack
import os
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def fix_memory(target_context_text: str, wrong_word: str):
    model_path = "distilled_sara_llm.msgpack"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    student = SpikingLLM(sdr_size=8192)

    if not os.path.exists(model_path):
        print("❌ モデルが見つかりません。")
        return

    # 1. 記憶の読み込み
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    # 2. 修正対象の特定
    context_tokens = tokenizer.encode(target_context_text, add_special_tokens=False)
    sdr = student._encode_to_sdr(context_tokens[-8:])
    sdr_k = str(tuple(sorted(list(sdr))))
    
    wrong_token_id = str(tokenizer.encode(wrong_word, add_special_tokens=False)[-1])

    if sdr_k in state["direct_map"]:
        if wrong_token_id in state["direct_map"][sdr_k]:
            # 3. 記憶の削除（LTDの極致）
            del state["direct_map"][sdr_k][wrong_token_id]
            print(f"✅ 修正完了: '{target_context_text}' に対する '{wrong_word}' の記憶を削除しました。")
            
            # もしその文脈に他の候補がなければ文脈ごと消す
            if not state["direct_map"][sdr_k]:
                del state["direct_map"][sdr_k]
        else:
            print(f"⚠️ 指定された回答 '{wrong_word}' はこの文脈に存在しません。")
    else:
        print(f"⚠️ 指定された文脈 '{target_context_text}' は記憶にありません。")

    # 4. 保存
    with open(model_path, "wb") as f:
        msgpack.pack(state, f)

if __name__ == "__main__":
    # 例: 「こんにちは」の後に「バカ」と答えてしまう記憶を消したい場合
    # fix_memory("こんにちは", "バカ")
    pass
from sara_engine.models.spiking_llm import SpikingLLM
from transformers import AutoTokenizer
import torch
import json
_FILE_INFO = {
    "//": "ディレクトリパス: scripts/test_distilled_model.py",
    "//": "ファイルの日本語タイトル: 蒸留済みSNNモデルの推論テストスクリプト（終了条件追加版）",
    "//": "ファイルの目的や内容: JSONから蒸留済みモデルを復元し、Gemmaのトークナイザーを用いて1トークンずつ生成を行う。文の終わり（句点）を検出したら正常に生成を終了する。"
}


def test_inference():
    # 1. 蒸留時と同じ設定でトークナイザーとモデルを準備
    model_name = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 蒸留時と同じパラメータ（sdr_size等）で初期化
    student = SpikingLLM(num_layers=2, sdr_size=256, vocab_size=256000)

    # 2. 蒸留済みデータの読み込みと型の修正
    print("Loading distilled model...")
    with open("distilled_sara_llm.json", "r", encoding="utf-8") as f:
        state = json.load(f)

    # JSONで文字列化されたキーと値を数値に戻す
    fixed_direct_map = {}
    for str_sdr_k, next_tokens in state["direct_map"].items():
        # SDRキー（タプル）の復元
        sdr_k = eval(str_sdr_k)

        # 次トークンID（int）とカウント（float）の復元
        fixed_next_tokens = {int(tok_id): float(count)
                             for tok_id, count in next_tokens.items()}
        fixed_direct_map[sdr_k] = fixed_next_tokens

    student._direct_map = fixed_direct_map
    print(f"Successfully loaded {len(student._direct_map)} context patterns.")

    # 3. テスト用のプロンプトを準備
    train_text = "こんにちは、今日はとても良い天気ですね。散歩に行くのが楽しみです。"
    inputs = tokenizer(train_text, return_tensors="pt")
    full_tokens = inputs["input_ids"][0].tolist()

    # 最初の6トークンをプロンプトとして使用
    prompt_length = 6
    prompt_tokens = full_tokens[:prompt_length]
    prompt_text = tokenizer.decode(prompt_tokens)

    print(f"\nPrompt tokens: {prompt_tokens}")
    print(f"Prompt text: {prompt_text}")

    # 4. ステップバイステップの生成
    print("\n--- Generation Step-by-Step ---")
    current_tokens = prompt_tokens.copy()
    context_window = 8

    for step in range(15):  # 最大15トークン生成
        # 現在のコンテキスト（直近8トークン）を取得
        context_tokens = current_tokens[-context_window:]
        sdr = student._encode_to_sdr(context_tokens)
        sdr_k = student._sdr_key(sdr)

        # HIT/MISS の判定
        if sdr_k in student._direct_map:
            print(
                f"Step {step+1}: ✅ HIT  | Context: {tokenizer.decode(context_tokens)}")
        else:
            print(
                f"Step {step+1}: ❌ MISS | Context: {tokenizer.decode(context_tokens)}")

        # 1トークンだけ生成 (temperatureを極限まで下げて決定論的に)
        next_id = student.generate(
            prompt_tokens=current_tokens,
            max_new_tokens=1,
            temperature=0.01,
            top_k=1
        )[0]  # type: ignore[index]

        current_tokens.append(next_id)
        generated_word = tokenizer.decode([next_id])
        print(f"  -> Generated: {generated_word} (ID: {next_id})")

        # --- 終了条件の判定 ---
        # 句点「。」や特殊な終了記号が出たらループを抜ける
        if generated_word.strip() == "。":
            print("\n🎉 文の終端（。）を検出したため、生成を正常終了します。")
            break

    print(f"\nFinal text: {tokenizer.decode(current_tokens)}")


if __name__ == "__main__":
    test_inference()

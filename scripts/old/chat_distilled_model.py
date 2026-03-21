from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.direct_map import restore_direct_map
from transformers import AutoTokenizer
import json
_FILE_INFO = {
    "//": "ディレクトリパス: scripts/chat_distilled_model.py",
    "//": "ファイルの日本語タイトル: 蒸留済みSNNモデルとの対話スクリプト",
    "//": "ファイルの目的や内容: ユーザーがターミナルから入力したテキストをプロンプトとし、SNNモデルがリアルタイムで続きを生成するインタラクティブなチャット機能を提供する。"
}


def run_chat():
    print("Loading tokenizer and model...")
    model_name = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 蒸留時と同じ設定で初期化
    student = SpikingLLM(num_layers=2, sdr_size=256, vocab_size=256000)

    print("Loading distilled knowledge...")
    try:
        with open("distilled_sara_llm.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        student._direct_map = restore_direct_map(state["direct_map"])
        print(
            f"✅ Successfully loaded {len(student._direct_map)} context patterns.")
    except FileNotFoundError:
        print("❌ Error: 'distilled_sara_llm.json' が見つかりません。先に distill_llm.py を実行してください。")
        return

    print("\n" + "="*50)
    print("🧠 SARA SNN Chat Session Started")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            print("チャットを終了します。お疲れ様でした！")
            break

        if not user_input.strip():
            continue

        # 入力テキストをトークン化
        inputs = tokenizer(user_input, return_tensors="pt")
        prompt_tokens = inputs["input_ids"][0].tolist()

        print(f"SARA: ", end="", flush=True)

        current_tokens = prompt_tokens.copy()

        # 1文字ずつ生成してストリーミング表示する
        for step in range(20):  # 安全のため最大20トークンで区切る
            # SNNによる1トークンの予測
            next_id = student.generate(
                prompt_tokens=current_tokens,
                max_new_tokens=1,
                temperature=0.01,
                top_k=1
            )[0]  # type: ignore[index]

            current_tokens.append(next_id)
            generated_word = tokenizer.decode([next_id])

            # 画面に出力
            print(generated_word, end="", flush=True)

            # 終了条件の判定
            if generated_word.strip() in ["。", "！", "？", "!", "?", "\n"]:
                break

        print()  # 1回の応答が終わったら改行


if __name__ == "__main__":
    run_chat()

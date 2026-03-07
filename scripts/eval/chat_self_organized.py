# {
#     "//": "ディレクトリパス: scripts/eval/chat_self_organized.py",
#     "//": "ファイルの日本語タイトル: 自己組織化学習モデルの対話テストスクリプト",
#     "//": "ファイルの目的や内容: SNNの記憶モデルを読み込んで対話する。パス指定を柔軟に行えるように改善。"
# }

import os
import sys

# プロジェクトルートのsrcをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.project_paths import model_path

def main():
    print("="*50)
    print("🧠 Loading Self-Organized SNN Memory")
    print("="*50)
    
    # 1. コマンドライン引数があればそれを使う、なければ相対パス
    if len(sys.argv) > 1:
        model_dir = os.path.abspath(sys.argv[1])
    else:
        model_dir = model_path("self_organized_llm")
    
    # 2. ファイル名まで指定された場合はディレクトリパスに直す
    if os.path.isfile(model_dir) and model_dir.endswith(".json"):
        model_dir = os.path.dirname(model_dir)

    model_path = os.path.join(model_dir, "spiking_llm_weights.json")
    print(f"[INFO] ターゲットディレクトリ: {model_dir}")

    # 3. 存在チェック
    if not os.path.exists(model_path):
        print(f"[ERROR] ファイルが見つかりません: {model_path}")
        print("\n💡 解決策1: 必要なモデルファイルが存在しません。先にモデルの学習を行ってください。")
        print("実行例: python scripts/train/train_self_organized.py")
        print("\n💡 解決策2: すでに学習済みで保存場所が異なる場合は、ターミナルでディレクトリを直接指定して実行してください。")
        print("実行例: python scripts/eval/chat_self_organized.py models/self_organized_llm")
        return

    # 記憶の読み込み
    try:
        llm = SpikingLLM.from_pretrained(model_dir)
        print("[INFO] Successfully loaded self-organized memory.")
        print("\n⚠️  [WARNING] 以前の文字ベースモデルの重みが残っている場合、推論品質が低下する場合があります。")
        print("   SaraTokenizerを導入した単語ベースでの学習を推奨します。")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        return

    print("\n" + "="*50)
    print("SARA Engine Inference Mode (No Backprop Model)")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)

    # インタラクティブループ
    while True:
        try:
            prompt = input("\nUser > ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
            
            # 推論実行
            print("SARA > ", end="", flush=True)
            output = llm.generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.3
            )
            print(output)
            
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break

if __name__ == "__main__":
    main()

_FILE_INFO = {
    "//": "ディレクトリパス: examples/sara_cli.py",
    "//": "タイトル: SARA インタラクティブ・チャット (CLI)",
    "//": "目的: 未知語のハッシュ衝突を防ぐため、スペース区切り(分かち書き)を促すようヘルプを修正。"
}

import os
import sys

# プロジェクトルート付近のsrcディレクトリを最優先でパスに通す
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.agent.sara_agent import SaraAgent

def print_help():
    print("\n--- SARA CLI コマンド一覧 ---")
    print(" 通常の入力 : 質問や会話を行います (例: 葉緑体 は 何 を します か ？)")
    print(" /teach     : 新しい知識を教えます (例: /teach biology: 葉緑体 は 光合成 を 行い ます)")
    print("              ※重要: 日本語は必ず「スペース区切り(分かち書き)」で入力してください！")
    print(" /vision    : ダミーの視覚入力とラベルを結合させます (例: /vision みかん)")
    print(" /sleep     : 海馬から皮質への記憶の統合(睡眠フェーズ)を実行します")
    print(" /reset     : 記憶と語彙を初期化してエージェントをまっさらな状態から再起動します")
    print(" /help      : このヘルプを表示します")
    print(" /exit      : チャットを終了します (または /quit)")
    print("-----------------------------\n")

def reset_agent() -> SaraAgent:
    files_to_remove = ["sara_multimodal_ltm.pkl", "sara_vocab.json"]
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"システムファイル {f} を削除しました。")
            except OSError as e:
                print(f"削除エラー {f}: {e}")
    print("SaraAgentを再初期化しています...")
    return SaraAgent()

def main():
    print("=== SARA リアルタイム・インタラクティブ対話環境 ===")
    print("エージェントを起動中... (過去の記憶ファイルがある場合は読み込みます)")
    
    agent = SaraAgent()
    print("起動完了！")
    print_help()

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します...")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in ["/exit", "/quit"]:
            print("SARAを終了します。お疲れ様でした！")
            break

        elif command == "/help":
            print_help()

        elif command == "/sleep":
            print("SARA: 睡眠モードに入ります...")
            res = agent.sleep(consolidation_epochs=3)
            print(f"SARA: {res}")

        elif command == "/reset":
            confirm = input("本当に全ての記憶を消去してリセットしますか？ (y/n): ")
            if confirm.lower() == 'y':
                agent = reset_agent()
                print("リセットが完了しました。")
            else:
                print("リセットをキャンセルしました。")

        elif user_input.startswith("/vision"):
            parts = user_input.split(" ", 1)
            if len(parts) > 1:
                label = parts[1].strip()
                # ランダムなダミー特徴量
                dummy_features = [0.8 if i % 7 == 0 else 0.1 for i in range(1000)]
                res = agent.perceive_image(dummy_features, label)
                print(f"SARA: {res}")
            else:
                print("ラベルを指定してください。例: /vision みかん")

        elif user_input.startswith("/teach"):
            text = user_input[len("/teach"):].strip()
            if text:
                res = agent.chat(text, teaching_mode=True)
                print(f"SARA: {res}")
            else:
                print("教える内容を入力してください。")

        else:
            # 通常のチャット(推論・生成モード)
            res = agent.chat(user_input, teaching_mode=False)
            print(f"SARA:\n{res}")

if __name__ == "__main__":
    main()
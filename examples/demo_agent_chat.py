_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_agent_chat.py",
    "//": "タイトル: 統合チャットエージェントデモ",
    "//": "目的: SARAエージェントを使用した対話、自律的な思考などの機能を統合したデモ。最新のchatメソッドに適合させる。"
}

import os
import time

def main():
    print("=== SARA Agentic Chat デモンストレーション ===")
    print("注意: 現在の環境でAPIキーや必要なモジュールがロードされているか確認します...")
    
    try:
        from sara_engine.agent.sara_agent import SaraAgent
    except ImportError as e:
        print(f"SARAエージェントモジュールの読み込みに失敗しました: {e}")
        print("PYTHONPATHの設定などを確認してください。")
        return

    agent = SaraAgent()
    print("エージェントの初期化が完了しました。")
    print("('quit' または 'exit' で終了します)\n")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue

            print("SARA is thinking...")
            start_time = time.time()
            
            # generate_response から 最新の chat に変更
            response = agent.chat(user_input, teaching_mode=False)
            
            end_time = time.time()
            print(f"\nSARA: {response}")
            print(f"(応答時間: {end_time - start_time:.2f}秒)")

        except KeyboardInterrupt:
            print("\n終了します...")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
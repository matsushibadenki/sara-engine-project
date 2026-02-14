_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_cli_agent.py",
    "//": "タイトル: インタラクティブ・エージェント CLI",
    "//": "目的: ユーザーが直接SaraAgentと対話し、教え、寝かせ、成長させるためのコマンドラインインターフェース。"
}

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.agent.sara_agent import SaraAgent

def print_help():
    print("\n--- コマンド一覧 ---")
    print(" 通常のテキスト : AIに質問します（想起モード）")
    print(" /teach [文章]  : AIに新しい知識を教えます（記銘モード）")
    print("                  例) /teach rust_expert: Rustは高速です")
    print(" /sleep         : AIを寝かせて、記憶の定着と整理を行います")
    print(" /stats         : 現在の海馬(LTM)の記憶数を表示します")
    print(" /help          : このヘルプを表示します")
    print(" /quit          : 終了します")
    print("--------------------\n")

def run_cli():
    print("=========================================================")
    print(" 🧠 Sara Agent Interactive CLI (生物由来・省エネAI)")
    print("=========================================================")
    
    agent = SaraAgent()
    
    # 起動時に大脳皮質(STDP)のシナプス状態をロードする
    load_msg = agent.load_brain()
    print(f"📁 システム: {load_msg}")
    
    print_help()
    
    while True:
        try:
            user_input = input("\n👤 あなた: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/quit', '/exit']:
                # 終了時に大脳皮質をオートセーブ
                save_msg = agent.save_brain()
                print(f"📁 システム: {save_msg}")
                print("👋 エージェントをシャットダウンします。おやすみなさい！")
                break
                
            elif user_input.lower() == '/help':
                print_help()
                
            elif user_input.lower() == '/stats':
                mem_count = len(agent.brain.ltm.memories)
                print(f"📊 [システム状態] 現在の海馬(LTM)のエピソード数: {mem_count}")
                
            elif user_input.lower() == '/sleep':
                print("💤 エージェント: (大脳皮質へリプレイ中...)")
                sleep_result = agent.sleep(consolidation_epochs=20)
                print(f"✨ エージェント: {sleep_result}")
                
                # 睡眠による成長を直後にオートセーブ
                save_msg = agent.save_brain()
                print(f"📁 システム: {save_msg}")
                
            elif user_input.startswith('/teach '):
                teach_text = user_input[7:].strip()
                if teach_text:
                    response = agent.chat(teach_text, teaching_mode=True)
                    print(f"🤖 エージェント: {response}")
                else:
                    print("⚠️ エラー: /teach の後に教える内容を入力してください。")
                    
            else:
                response = agent.chat(user_input, teaching_mode=False)
                print(f"🤖 エージェント: {response}")
                
        except KeyboardInterrupt:
            # 強制終了時にも可能な限りセーブを試みる
            print("\n👋 強制終了が検出されました。大脳皮質を保存して終了します...")
            agent.save_brain()
            break
        except Exception as e:
            print(f"\n⚠️ エラーが発生しました: {e}")

if __name__ == "__main__":
    run_cli()
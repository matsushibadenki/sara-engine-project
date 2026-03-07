# ディレクトリパス: scripts/eval/chat_agent.py
# ファイルの日本語タイトル: 統合エージェント・チャット評価スクリプト
# ファイルの目的や内容: 強化された会話メモリ、再ランキング、文完成機構を持つ SaraAgent を用いた対話インターフェース。対話を通じた動的学習（teaching_mode）のテスト機能を追加。

import os
import sys
from datetime import datetime

# srcディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.pipelines.agent_chat import AgentChatPipeline

def get_current_time(prompt: str) -> str:
    """エージェントに提供する外部ツール（現在時刻の取得）"""
    now = datetime.now()
    return f"現在の時刻は {now.strftime('%Y年%m月%d日 %H時%M分')} です。"

def get_calculator(prompt: str) -> str:
    """エージェントに提供する外部ツール（簡易計算機）"""
    try:
        import re
        match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)\s*計算', prompt)
        if match:
            expr = match.group(1).strip()
            if re.fullmatch(r'[\d\s\+\-\*\/\(\)\.]+', expr):
                result = eval(expr)
                return str(result)
    except Exception:
        pass
    return "計算できませんでした"

def main():
    print("=" * 60)
    print("🧠 SARA-Engine: Integrated Agent Chat Inference")
    print("=" * 60)
    
    print("[INFO] SaraAgentを初期化しています...")
    agent = SaraAgent(
        input_size=2048,
        hidden_size=4096,
        compartments=["general", "python_expert", "biology", "vision", "audio"]
    )
    
    print("[INFO] 学習済みの記憶モデルをロードしています...")
    agent.load_agent("workspace/models/sara_agent")
    
    pipeline = AgentChatPipeline(agent)
    
    print("[INFO] 外部ツール（アクションスパイク）を登録しています...")
    pipeline.register_tool("何時", get_current_time)
    pipeline.register_tool("計算", get_calculator)
    
    print("\n✅ 準備が完了しました。SARAと対話を開始します。")
    print("・「終了」「quit」「exit」で終了します。")
    print("・「何時」や「[数式] 計算」と入力すると、自律的にツールを実行します。")
    print("・「学習: [教えたい内容]」と入力すると、即座に海馬とSNNに記憶させることができます。")
    print("・直前の文脈を踏まえた回答や、知識が足りない場合のフォールバックを評価できます。\n")

    while True:
        try:
            user_input = input("👤 あなた: ")
            
            if user_input.strip().lower() in ["終了", "quit", "exit"]:
                print("👋 対話を終了します。")
                break
                
            if not user_input.strip():
                continue

            # 💡 オンライン学習コマンドの処理
            if user_input.startswith("学習:"):
                learn_text = user_input[3:].strip()
                if not learn_text:
                    print("🤖 SARA: 学習する内容を入力してください。\n")
                    continue
                    
                print("🤖 SARA: 新しい知識を学習中...", end="\r")
                # teaching_mode=True で推論パイプラインに通す
                response = pipeline(text=learn_text, teaching_mode=True)
                # 学習した内容を永続化
                agent.save_agent("workspace/models/sara_agent")
                
                print(" " * 30, end="\r")
                print(f"🤖 SARA:\n{response}\n")
                continue

            print("🤖 SARA: 考え中...", end="\r")
            
            response = pipeline(text=user_input, teaching_mode=False)
            
            print(" " * 30, end="\r")
            print(f"🤖 SARA:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 対話を終了します。")
            break
        except Exception as e:
            import traceback
            print(f"\n❌ エラーが発生しました: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
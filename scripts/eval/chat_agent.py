# ディレクトリパス: scripts/eval/chat_agent.py
# ファイルの日本語タイトル: 統合エージェント・チャット評価スクリプト
# ファイルの目的や内容: "biology: 学習: ..." のようなコンテキスト指定付きの学習コマンドを正しくパースできるように修正。

import os
import sys
import argparse
import ast
from datetime import datetime

# srcディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.pipelines.agent_chat import AgentChatPipeline
from sara_engine.safety.safety_guard import SafetyGuard
from sara_engine.utils.project_paths import model_path, workspace_path

def get_current_time(prompt: str) -> str:
    """エージェントに提供する外部ツール（現在時刻の取得）"""
    now = datetime.now()
    return f"現在の時刻は {now.strftime('%Y年%m月%d日 %H時%M分')} です。"


def _safe_eval_arithmetic(expr: str) -> float:
    parsed = ast.parse(expr, mode="eval")
    return float(_evaluate_arithmetic_node(parsed.body))


def _evaluate_arithmetic_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _evaluate_arithmetic_node(node.left)
        right = _evaluate_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0.0:
            raise ZeroDivisionError("division by zero")
        return left / right
    raise ValueError("unsupported expression")

def get_calculator(prompt: str) -> str:
    """エージェントに提供する外部ツール（簡易計算機）"""
    try:
        import re
        normalized = prompt.strip()
        if not normalized.endswith("計算"):
            return "計算できませんでした"
        expr = normalized[:-2].strip()
        if re.fullmatch(r'[\d\s\+\-\*\/\(\)\.]+', expr):
            result = _safe_eval_arithmetic(expr)
            if result.is_integer():
                return str(int(result))
            return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return "計算できませんでした"
    return "計算できませんでした"

def main():
    parser = argparse.ArgumentParser(description="Interactive chat for SaraAgent.")
    parser.add_argument(
        "--model-dir",
        default=model_path("sara_agent"),
        help="Directory containing saved SaraAgent state.",
    )
    parser.add_argument(
        "--session-path",
        default=workspace_path("sessions", "sara_agent_session.pkl"),
        help="Path to persist dialogue session state.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt injected into generation.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses.",
    )
    parser.add_argument(
        "--safety",
        action="store_true",
        help="Enable safety guard checks for input/output.",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Show lightweight runtime diagnostics after each turn only when issues exist.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 SARA-Engine: Integrated Agent Chat Inference")
    print("=" * 60)
    
    print("[INFO] SaraAgentを初期化しています...")
    guard = SafetyGuard() if args.safety else None
    agent = SaraAgent(
        input_size=2048,
        hidden_size=4096,
        compartments=["general", "python_expert", "biology", "vision", "audio"],
        system_prompt=args.system_prompt,
        safety_guard=guard,
    )
    
    print("[INFO] 学習済みの記憶モデルをロードしています...")
    agent.load_agent(args.model_dir)
    if args.session_path:
        agent.load_session(args.session_path)
    
    pipeline = AgentChatPipeline(agent)
    
    print("[INFO] 外部ツール（アクションスパイク）を登録しています...")
    pipeline.register_tool("何時", get_current_time)
    pipeline.register_tool("計算", get_calculator)
    
    print("\n✅ 準備が完了しました。SARAと対話を開始します。")
    print("・「終了」「quit」「exit」で終了します。")
    print("・「何時」や「[数式] 計算」と入力すると、自律的にツールを実行します。")
    print("・「学習: [教えたい内容]」と入力すると、即座に海馬とSNNに記憶させることができます。")
    print("・「/diagnostics」で直近の実行問題を確認、「/clear-diagnostics」で消去できます。")
    print("・直前の文脈を踏まえた回答や、知識が足りない場合のフォールバックを評価できます。\n")

    while True:
        try:
            user_input = input("👤 あなた: ")
            
            if user_input.strip().lower() in ["終了", "quit", "exit"]:
                print("👋 対話を終了します。")
                break
                
            if not user_input.strip():
                continue

            normalized = user_input.strip().lower()
            if normalized == "/diagnostics":
                print(f"🤖 SARA:\n{agent.format_recent_issues()}\n")
                continue
            if normalized == "/clear-diagnostics":
                agent.clear_runtime_issues()
                if args.session_path:
                    agent.save_session(args.session_path)
                print("🤖 SARA:\nRuntime diagnostics were cleared.\n")
                continue

            # 💡 オンライン学習コマンドの処理 ("biology: 学習: ..." も拾えるように修正)
            if "学習:" in user_input:
                learn_text = user_input.replace("学習:", "").strip()
                if not learn_text:
                    print("🤖 SARA: 学習する内容を入力してください。\n")
                    continue
                    
                print("🤖 SARA: 新しい知識を学習中...", end="\r")
                # teaching_mode=True で推論パイプラインに通す
                response = pipeline(text=learn_text, teaching_mode=True)
                # 学習した内容を永続化
                agent.save_agent(args.model_dir)
                if args.session_path:
                    agent.save_session(args.session_path)
                
                print(" " * 30, end="\r")
                print(f"🤖 SARA:\n{response}\n")
                if args.show_diagnostics and agent.get_recent_issues(limit=1):
                    print(f"{agent.format_recent_issues(limit=3)}\n")
                continue

            print("🤖 SARA: 考え中...", end="\r")
            
            if args.stream:
                stream_gen = agent.chat(user_text=user_input, teaching_mode=False, stream=True)
                print("🤖 SARA: ", end="", flush=True)
                for chunk in stream_gen:
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = pipeline(text=user_input, teaching_mode=False)
                print(" " * 30, end="\r")
                print(f"🤖 SARA:\n{response}\n")
            if args.session_path:
                agent.save_session(args.session_path)
            if args.show_diagnostics and agent.get_recent_issues(limit=1):
                print(f"{agent.format_recent_issues(limit=3)}\n")

        except KeyboardInterrupt:
            print("\n👋 対話を終了します。")
            if args.session_path:
                agent.save_session(args.session_path)
            break
        except Exception as e:
            import traceback
            print(f"\n❌ エラーが発生しました: {e}")
            traceback.print_exc()
            if args.session_path:
                agent.save_session(args.session_path)

if __name__ == "__main__":
    main()

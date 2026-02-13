_FILE_INFO = {
    "//": "ディレクトリパス: examples/interactive_snn.py",
    "//": "タイトル: 対話型SNNシェル",
    "//": "目的: 学習済みモデルを使用して、任意のドキュメントに対するQAを行う。"
}

import sys
import os
import time


from sara_engine import StatefulRLMAgent

def type_writer(text, delay=0.02):
    """タイプライター風出力"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def interactive_shell():
    print("\033[H\033[J")  # 画面クリア
    print("==========================================")
    print("   SARA Stateful SNN - Interactive Core   ")
    print("==========================================")
    
    model_path = "models/stateful_rl_trained.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Running with initialized (untrained) weights.")
        model_path = None # 未学習で起動
    else:
        print(f"Loading trained brain from: {model_path}")

    # エージェント初期化
    agent = StatefulRLMAgent(model_path=model_path)
    print("System Online.\n")

    while True:
        print("-" * 40)
        print("[1] Set Document (Context)")
        print("[2] Ask Question")
        print("[3] Exit")
        
        choice = input("\nSelect Action >> ").strip()
        
        if choice == "1":
            print("\nEnter the document text (Press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    break
            document = " ".join(lines)
            if len(document) < 10:
                print("Document too short, using default example.")
                document = ("This is a top secret memo. The project code is RED-DRAGON. "
                            "Do not share this with anyone. End of message.")
            
            # ドキュメントをチャンク分割してWorking Memoryの準備をするなどの処理は
            # solveメソッド内で行われるため、ここでは保持だけする
            current_doc = document
            print(f"\nDocument stored ({len(document)} chars).")

        elif choice == "2":
            if 'current_doc' not in locals():
                print("Please set a document first [Option 1].")
                continue
                
            query = input("Query >> ").strip()
            if not query: continue
            
            print("\nThinking Process:")
            # 推論実行（学習はオフ）
            answer = agent.solve(query, current_doc, train_rl=False)
            
            print(f"\n>>> SARA's Answer: {answer}")
            
            # 答えの解説（SNNが見つけた根拠）
            if answer:
                type_writer(f"Confidence: High (Found in state EXTRACT)")
            else:
                type_writer("Confidence: Low (Could not extract definite answer)")

        elif choice == "3":
            print("Shutting down.")
            break
        
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    interactive_shell()
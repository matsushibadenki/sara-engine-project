_FILE_INFO = {
    "//": "ディレクトリパス: examples/interactive_snn.py",
    "//": "タイトル: 対話型SNNシェル",
    "//": "目的: インタラクティブにSARAのQA機能をテストする。"
}

import sys
import os
import time
from sara_engine import StatefulRLMAgent

def type_writer(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def interactive_shell():
    print("\033[H\033[J")
    print("==========================================")
    print("   SARA Stateful SNN - Interactive Shell  ")
    print("==========================================")
    
    model_path = "models/stateful_rl_trained.pkl"
    agent = StatefulRLMAgent(model_path=model_path if os.path.exists(model_path) else None)
    print("System Online.\n")

    current_doc = ""

    while True:
        print("-" * 40)
        print("[1] Set Document (Context)")
        print("[2] Ask Question")
        print("[3] Exit")
        
        choice = input("\nSelect Action >> ").strip()
        
        if choice == "1":
            print("\nEnter text (Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if not line: break
                lines.append(line)
            current_doc = " ".join(lines)
            print(f"\nDocument stored ({len(current_doc)} chars).")

        elif choice == "2":
            if not current_doc:
                print("Warning: No document set. Setting default context.")
                current_doc = "The system version is SARA-0.1.3. The key is BLUE-EYES."
                
            query = input("Query >> ").strip()
            if not query: continue
            
            print("\nThinking...")
            answer = agent.solve(query, current_doc, train_rl=False)
            print(f"\n>>> SARA: {answer}")

        elif choice == "3":
            print("Shutdown.")
            break
        
        else:
            print("Invalid input.")

if __name__ == "__main__":
    interactive_shell()
# file_path: examples/run_rlm_demo.py
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.sara_gpt_core import SaraGPT
from sara_engine.rlm import SaraRecursiveAgent

def get_dummy_long_context():
    text = (
        "CONFIDENTIAL SPECIFICATIONS v9.2\n"
        "--------------------------------\n"
        "Section 1: Overview\n"
        "SARA (Spiking Advanced Recursive Architecture) runs on CPU.\n"
        "\n"
    )
    for i in range(10):
        text += f"Log {i}: [REDACTED]\n"
    
    text += (
        "\n"
        "Section 25: Security\n"
        "The master override code is 'BLUE-OCEAN-42'.\n"
        "\n"
    )
    text += (
        "Section 36: Easter Egg\n"
        "Dream Mode activates after 8 hours of idle time.\n"
        "End.\n"
    )
    return text

def quick_bootstrap(brain: SaraGPT):
    print("Bootstrapping SNN...")
    patterns = [
        ["QUERY:", "code", "ACTION:", "SEARCH", "code"],
        ["RESULT:", "Found", "in", "25", "ACTION:", "READ", "25"],
        ["RESULT:", "BLUE-OCEAN", "ACTION:", "FINAL", "BLUE-OCEAN"],
    ]
    for _ in range(5):
        for p in patterns:
            brain.train_sequence(p)

def main():
    print("==================================================")
    print(" SARA ENGINE - RLM DEMO (Assisted Mode)")
    print("==================================================")
    
    brain = SaraGPT(sdr_size=1024)
    quick_bootstrap(brain)
    
    agent = SaraRecursiveAgent(brain)
    long_text = get_dummy_long_context()
    
    print(f"\n[Environment] Doc Length: {len(long_text)} chars")
    
    while True:
        try:
            query = input("\nQuery (e.g., 'What is the code?'): ").strip()
            if query.lower() in ['exit', 'quit']: break
            if not query: continue
            
            # プライミング
            target_word = query.split()[-1].replace("?", "")
            print(f"Priming Brain: '{target_word}' -> SEARCH")
            for _ in range(3):
                brain.train_sequence(["QUERY:", target_word, "ACTION:", "SEARCH", target_word])
            
            start_time = time.time()
            answer = agent.solve(long_text, query)
            elapsed = time.time() - start_time
            
            print(f"\n>>> Final Answer: {answer}")
            print(f">>> Time: {elapsed:.2f} sec")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
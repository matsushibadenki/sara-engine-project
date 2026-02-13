# file_path: examples/run_rlm_demo.py (v3 - SIMPLIFIED)
import os
import sys
import time


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

def minimal_bootstrap(brain: SaraGPT):
    """
    最小限のブートストラップ（v3はSNNに依存しない）
    """
    print("=== Minimal Bootstrapping (v3) ===")
    
    # トークナイザーの基本語彙のみ
    basic_corpus = [
        "SEARCH", "READ", "FINAL", "CHUNK",
        "code", "master", "override", "password",
        "dream", "mode", "activate", "security"
    ]
    
    if hasattr(brain.encoder, 'tokenizer'):
        print("Training tokenizer with basic vocabulary...")
        brain.encoder.tokenizer.train(basic_corpus)
    
    print("Bootstrapping complete.\n")

def main():
    print("==================================================")
    print(" SARA ENGINE - RLM DEMO (v3 - State Machine)")
    print("==================================================")
    
    brain = SaraGPT(sdr_size=1024)
    minimal_bootstrap(brain)
    
    agent = SaraRecursiveAgent(brain)
    long_text = get_dummy_long_context()
    
    print(f"\n[Environment] Doc Length: {len(long_text)} chars")
    
    # テストクエリセット
    test_queries = [
        "What is the code?",
        "What is the master override code?",
        "When does dream mode activate?",
    ]
    
    print("\n=== Running Test Queries ===")
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        start_time = time.time()
        answer = agent.solve(long_text, query)
        elapsed = time.time() - start_time
        
        print(f"\n>>> FINAL ANSWER: {answer}")
        print(f">>> Time: {elapsed:.2f} sec")
        print()
        
        # 短い休憩
        time.sleep(0.5)
    
    print("\n=== Interactive Mode ===")
    while True:
        try:
            query = input("\nQuery (or 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit', 'q']: 
                break
            if not query: 
                continue
            
            start_time = time.time()
            answer = agent.solve(long_text, query)
            elapsed = time.time() - start_time
            
            print(f"\n>>> FINAL ANSWER: {answer}")
            print(f">>> Time: {elapsed:.2f} sec")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
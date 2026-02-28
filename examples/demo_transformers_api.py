_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_transformers_api.py",
    "//": "ファイルの日本語タイトル: TransformersライクなAPIデモ",
    "//": "ファイルの目的や内容: SARAのオンライン学習機能を利用して、その場で知識を与えてから推論を行うデモ。結果はworkspaceに保存されます。"
}

import os
import json
from sara_engine.inference import SaraInference
from sara_engine.pipelines import pipeline

def main():
    print("[INFO] Initializing SARA Engine in Transformers-compatible mode...")
    
    # Create workspace directory for logs and outputs to avoid cluttering the project root
    workspace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    log_file_path = os.path.join(workspace_dir, "transformers_api_demo_log.json")
    dummy_model_path = os.path.join(workspace_dir, "dummy_model.msgpack")
    
    try:
        model = SaraInference(model_path=dummy_model_path)
        tokenizer = model.tokenizer
        
        # Initialize pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        # --- 追加: 空の脳にその場で知識を教え込む（オンライン学習） ---
        print("\n[INFO] Teaching SARA some basic facts online (Hebbian Learning)...")
        generator.learn("Hello, how are you? I am doing great, thank you!")
        generator.learn("The future of artificial intelligence is highly efficient Edge SNNs.")
        generator.learn("こんにちは、今日の天気は晴れです。")
        
        # 学習結果を次回以降のために保存
        generator.save_pretrained(dummy_model_path)
        print("[INFO] Learning complete.\n")
        # -------------------------------------------------------------

        prompts = [
            "Hello, how are you?",
            "The future of artificial intelligence is",
            "こんにちは、今日の天気は"
        ]
        
        results = []
        for prompt in prompts:
            print(f"[Prompt]: {prompt}")
            
            # Using biological parameters (refractory_period) instead of matrix-based penalties
            output = generator(
                prompt, 
                max_new_tokens=20, 
                temperature=0.7, 
                top_k=5,
                refractory_period=15, # Biological memory of recent spikes
                repetition_penalty=1.5 # Aliased to refractory_penalty internally
            )
            print(f"[Generated]: {output}\n")
            results.append({"prompt": prompt, "output": output})
            
        # Save results to workspace
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"[SUCCESS] Demo completed. Logs saved to {log_file_path}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
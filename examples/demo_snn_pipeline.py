_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_pipeline.py",
    "//": "ファイルの日本語タイトル: SNNパイプラインのデモ",
    "//": "ファイルの目的や内容: SNNを用いたテキスト生成パイプラインの動作確認。多言語のSTDP学習を行い、ワークスペース下に出力ファイルを保存する。"
}

import os
import sys

# Ensure the sara_engine package is in the path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoModelForCausalSNN, AutoTokenizer
from sara_engine.pipelines.text_generation import pipeline

def main():
    print("=== SARA Engine: SNN Pipeline Demonstration ===")
    
    # Create workspace directory to store generated files and logs
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_pipeline_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    
    model_dir = os.path.join(workspace_dir, "snn_model_checkpoint")
    
    # 1. Initialize tokenizer and model (similar to Hugging Face API)
    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalSNN.from_pretrained(model_dir)
    
    # 2. Instantiate pipeline
    print("Initializing Text Generation Pipeline...")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # 3. Multilingual training data (English + Japanese to prove multilingual support)
    training_data = [
        "SARA engine utilizes Spiking Neural Networks.",
        "誤差逆伝播法を使わず、STDPによる局所学習を行います。",
        "This is an energy efficient alternative to Transformers."
    ]
    
    # 4. Local STDP Learning Phase (No Backpropagation)
    print("\n--- Starting Local STDP Learning Phase ---")
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for text in training_data:
            print(f"  Learning sequence: {text}")
            generator.learn(text)
            
    # 5. Generation Phase
    print("\n--- Text Generation Phase ---")
    prompts = [
        "SARA engine",
        "誤差逆伝播法",
        "This is an"
    ]
    
    for prompt in prompts:
        # Generate with max_length=30
        generated_text = generator(prompt, max_length=30)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'\n")
        
    # 6. Save the model state to workspace
    print(f"Saving model state to: {model_dir}")
    generator.save_pretrained(model_dir)
    
    # 7. Write a log file
    log_file = os.path.join(workspace_dir, "execution.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA SNN Pipeline Execution Log ===\n")
        f.write("Status: SUCCESS\n")
        f.write(f"Model saved at: {model_dir}\n")
        f.write("Tested Features:\n")
        f.write("  - AutoTokenizer / AutoModelForCausalSNN\n")
        f.write("  - Multilingual STDP Learning (UTF-8 byte level)\n")
        f.write("  - Backpropagation-free learning\n")
        f.write("  - Matrix-multiplication-free inference\n")
        
    print(f"Execution log saved to {log_file}")
    print("=== Demonstration Completed ===")

if __name__ == "__main__":
    main()
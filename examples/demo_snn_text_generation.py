_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_text_generation.py",
    "//": "タイトル: SNNテキスト生成デモ",
    "//": "目的: SpikingCausalLMを用いてテキストの学習と生成を行う。SDRの衝突を防ぐためモデルの次元数を拡大。"
}

import os
import sys

# Ensure the core module is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.models.spiking_causal_lm import SpikingCausalLM

def main():
    print("Starting Spiking Causal LM Demo...")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../workspace'))
    logs_dir = os.path.join(workspace_dir, 'logs')
    models_dir = os.path.join(workspace_dir, 'models')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    tokenizer = SpikeTokenizer()
    
    training_data = [
        "SNN is a brain-inspired computing paradigm.",
        "We are building a Spiking Neural Network.",
        "Transformers are powerful, but SNNs are energy efficient.",
        "人工知能 は 新しい 時代 を 迎えて います。",
        "スパイク ニューラル ネットワーク は 省電力 です。",
        "これ は SNN ベース の 言語 モデル です。"
    ]
    
    print("Training Tokenizer...")
    tokenizer.train(training_data)
    tokenizer_path = os.path.join(models_dir, 'tokenizer_vocab.json')
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # 表現の衝突を防ぐために次元数を大幅に拡大（128 -> 1024）
    d_model = 1024
    model = SpikingCausalLM(vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=2, num_heads=2)
    
    print("\nTraining Spiking Causal LM...")
    epochs = 20
    for epoch in range(epochs):
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            
        # 最初の3エポックだけTransformer内部の重みを更新し、その後は固定してReadout層に集中させる
        update_backbone = epoch < 3 
        
        for text in training_data:
            input_ids = tokenizer.encode(text)
            input_ids = [2] + input_ids + [3]
            model.train_step(input_ids, update_backbone=update_backbone)
            
    print("\n--- Text Generation Test ---")
    prompts = [
        "SNN is a",
        "これ は SNN",
        "Transformers are"
    ]
    
    log_file_path = os.path.join(logs_dir, 'generation_results.txt')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            print(f"Input Prompt: {prompt}")
            input_ids = tokenizer.encode(prompt)
            input_ids = [2] + input_ids  # Prefix with BOS
            
            generated_ids = model.generate(input_ids, max_new_tokens=10, top_k=3)
            print(f"Raw Generated IDs: {generated_ids}")
            
            if generated_ids and generated_ids[0] == 2:
                generated_ids = generated_ids[1:]
                
            generated_text = tokenizer.decode(generated_ids)
            
            result_str = f"Prompt: {prompt}\nGenerated IDs: {generated_ids}\nGenerated: {generated_text}\n\n"
            print(f"Generated Text: {generated_text}\n")
            f.write(result_str)
            
    print(f"Generation results have been saved to {log_file_path}")

if __name__ == "__main__":
    main()
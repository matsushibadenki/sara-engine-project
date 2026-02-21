_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_pipeline.py",
    "//": "ファイルの日本語タイトル: SNNパイプラインのデモ",
    "//": "ファイルの目的や内容: STDP学習時に終了トークン[EOS]を付与し、モデルが自律的に生成を停止できるように修正。"
}

import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer
from sara_engine.pipelines import pipeline

def main():
    print("=== SNN Text Generation Pipeline Demo ===")
    print("Initiating bio-inspired, backprop-free, matrix-free model inference.\n")
    
    # 1. Setup workspace for artifacts
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'pipeline_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 2. Initialize tokenizer and train on multi-lingual data
    tokenizer = SpikeTokenizer()
    training_texts = [
        "Hello, I am a spiking neural network.",
        "This model is biologically inspired and energy efficient.",
        "こんにちは、私はスパイキングニューラルネットワークです。",
        "省エネルギーで動作し、行列演算を使用しません。",
        "Bonjour, je suis un réseau de neurones à impulsions."
    ]
    print("Training SpikeTokenizer on multilingual corpus...")
    tokenizer.train(training_texts)
    
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to: {tokenizer_path}\n")
    
    # 3. Initialize SNN model configuration
    config = SNNTransformerConfig(
        vocab_size=max(256, tokenizer.vocab_size + 50), 
        embed_dim=128, 
        num_layers=2, 
        ffn_dim=256
    )
    model = SpikingTransformerModel(config)
    
    # 4. Pre-train the model with STDP rule
    print("Training SNN model with STDP (No backpropagation, CPU-only)...")
    epochs = 20
    
    for epoch in range(epochs):
        for text in training_texts:
            # 重要: 文章の終わりに[EOS] (ID:3) を付与し、モデルに「ストップ信号」を学習させる
            token_ids = tokenizer.encode(text) + [3]
            model.reset_state()
            for i in range(len(token_ids) - 1):
                # Biologically plausible step-by-step learning via Hebbian STDP
                model.forward_step(token_ids[i], learning=True, target_id=token_ids[i+1])
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed.")
            
    model_save_path = os.path.join(workspace_dir, "saved_snn_model")
    model.save_pretrained(model_save_path)
    print(f"\nModel successfully saved to: {model_save_path}\n")

    # 5. Initialize Pipeline
    print("Initializing Text Generation Pipeline...")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # 6. Generate Text using the Pipeline API
    # 20エポックでEOSの出力まで完全に学習しているはずなので、max_new_tokensを少し長めに取ります
    prompts = [
        "Hello, I am",
        "こんにちは、私は"
    ]
    
    for prompt in prompts:
        print(f"\n[Prompt] -> '{prompt}'")
        output = generator(prompt, max_new_tokens=30)
        print(f"[Generated] -> '{output[0]['generated_text']}'")

if __name__ == "__main__":
    main()
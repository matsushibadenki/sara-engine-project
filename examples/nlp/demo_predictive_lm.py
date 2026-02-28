_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_predictive_lm.py",
    "//": "ファイルの日本語タイトル: 予測符号化因果言語モデルのデモ",
    "//": "ファイルの目的や内容: Predictive CodingとSpikeSelfAttentionを組み合わせた次世代SNN言語モデルの学習とテキスト生成のテスト。"
}

import os
from sara_engine.models.spiking_causal_lm import SpikingCausalLM, SpikingCausalLMConfig

def main():
    print("--- Testing Predictive Spiking Causal LM ---")
    
    # モデルの初期化
    config = SpikingCausalLMConfig(vocab_size=256, embed_dim=128, context_size=32)
    model = SpikingCausalLM(config)
    
    training_data = [
        "SARA Engine is an innovative AI framework.",
        "It uses Spiking Neural Networks.",
        "No backpropagation is required.",
        "Predictive coding saves energy."
    ]
    
    epochs = 8
    print(f"Training LM on {len(training_data)} sentences for {epochs} epochs using STDP...")
    
    for epoch in range(epochs):
        for text in training_data:
            model.learn_sequence(text)
            
    print("\nTraining complete. Testing Generation...")
    
    # 生成テスト
    prompts = [
        "SARA Engine is",
        "It uses Spiking",
        "Predictive"
    ]
    
    for prompt in prompts:
        # 学習データが少ないため文字化けする可能性がありますが、
        # バイナリレベルで予測が機能しているかを確かめます
        generated = model.generate(prompt, max_length=15)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'\n")

    # モデルの保存テスト
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace", "predictive_lm")
    model.save_pretrained(workspace_dir)
    print(f"Predictive LM saved to {workspace_dir}")

if __name__ == "__main__":
    main()
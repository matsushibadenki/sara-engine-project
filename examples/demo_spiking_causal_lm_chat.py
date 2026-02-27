_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_causal_lm_chat.py",
    "//": "ファイルの日本語タイトル: スパイキング因果LLMのチャット生成デモ（修正版）",
    "//": "ファイルの目的や内容: 実際に学習したコーパスから抽出した質問を用いてテストを行い、SNNが正しく質問ごとの回答を記憶・分離できているかを検証する。"
}

import os
import time
import json
from sara_engine.models.spiking_causal_lm import SpikingCausalLM
from sara_engine.encoders.spike_tokenizer import SpikeTokenizer

def main():
    print("Starting Spiking Causal LM Chat Demo with BPE Tokenizer...\n")
    
    # 1. データの準備
    data_path = "data/chat_data.jsonl"
    corpus = []
    test_prompts = []  # テスト用に実際の質問を保存するリスト
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:  # デモのため50件に制限
                    break
                try:
                    data = json.loads(line)
                    user_text = data.get('user', '')
                    ast_text = data.get('assistant', '')
                    text = f"User: {user_text} Assistant: {ast_text}"
                    corpus.append(text)
                    test_prompts.append(user_text)
                except json.JSONDecodeError:
                    continue
    else:
        print("chat_data.jsonl not found. Using sample chat corpus.")
        corpus = [
            "User: Hello ! Assistant: Hi there ! How can I help you ?",
            "User: What is SARA Engine ? Assistant: SARA Engine is a fast SNN library .",
            "User: Is SNN efficient ? Assistant: Yes , biological brains are highly efficient ."
        ]
        test_prompts = ["Hello !", "What is SARA Engine ?", "Is SNN efficient ?"]

    # 2. BPEトークナイザーの初期化と学習
    tokenizer = SpikeTokenizer(vocab_size=1000)
    print("Training BPE Tokenizer...")
    tokenizer.train(corpus)
    print(f"Vocabulary Size: {tokenizer.vocab_size}\n")
    
    # 3. モデルの初期化
    model = SpikingCausalLM(vocab_size=tokenizer.vocab_size, embed_dim=1024, hidden_dim=2048, use_lif=True)
    
    print("--- Training SpikingCausalLM ---")
    start_time = time.time()
    
    # 3エポック学習
    epochs = 3
    for epoch in range(epochs):
        for text in corpus:
            token_ids = tokenizer.encode(text)
            if len(token_ids) > 1:
                model.train_step(token_ids, learning_rate=0.5)
        print(f"Epoch {epoch+1}/{epochs} completed.")
        
    print(f"Training completed in {time.time() - start_time:.4f} seconds.\n")
    
    # 4. テキスト生成（チャット応答）用のヘルパー関数
    def generate_reply(prompt_text: str, max_tokens: int = 15, temperature: float = 0.1):
        print(f"User: '{prompt_text}'")
        full_prompt = f"User: {prompt_text} Assistant:"
        prompt_ids = tokenizer.encode(full_prompt)
        
        if not prompt_ids:
            print("Unknown prompt words.")
            return
            
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
        
        # IDリストを一括でテキストにデコード
        result_text = tokenizer.decode(generated_ids)
        
        # 不要な「Assistant:」が含まれていたら除去
        if result_text.startswith("Assistant:"):
            result_text = result_text.replace("Assistant:", "", 1).strip()
            
        # ピリオド、句点、クエスチョンマーク等で切り捨てて表示を整える
        for p in [".", "!", "?", "。", "！", "？"]:
            if p in result_text:
                result_text = result_text.split(p)[0] + p
                break
                
        print(f"Assistant: {result_text.strip()}\n")

    print("--- Chat Inference ---")
    
    # 学習データに実際に存在する最初の3つの質問でテスト
    prompts_to_test = test_prompts[:3]
    for prompt in prompts_to_test:
        generate_reply(prompt, max_tokens=15, temperature=0.1)

if __name__ == "__main__":
    main()
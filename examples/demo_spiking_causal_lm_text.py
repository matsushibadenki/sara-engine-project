_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_causal_lm_text.py",
    "//": "ファイルの日本語タイトル: スパイキング因果LLMのテキスト生成デモ",
    "//": "ファイルの目的や内容: 完成した最新のSpikingCausalLMを用いて、実際の英文テキストを単語レベルで学習させ、文脈に沿った文章生成ができるかを実証する。文末（ピリオド等）で生成をストップする機能付き。"
}

import os
import time
from sara_engine.models.spiking_causal_lm import SpikingCausalLM

def main():
    print("Starting Spiking Causal LM Text Generation Demo...\n")
    
    # 小規模な学習コーパス（複数の文脈を持たせる）
    corpus = [
        "SARA Engine is a fast and lightweight SNN library .",
        "The AI model is very smart and fast .",
        "SARA Engine can replace standard transformers .",
        "Biological brains are highly efficient and lightweight ."
    ]
    
    # 簡易的な単語レベルのトークナイザーを作成
    words = set()
    for sentence in corpus:
        for word in sentence.split():
            words.add(word)
            
    vocab = sorted(list(words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Vocabulary: {vocab}\n")
    
    # 最新のモデルの初期化
    model = SpikingCausalLM(vocab_size=vocab_size, embed_dim=1024, hidden_dim=2048, use_lif=True)
    
    print("--- Training ---")
    start_time = time.time()
    
    # 3エポック学習
    for epoch in range(3):
        for sentence in corpus:
            sequence = [word_to_id[w] for w in sentence.split()]
            model.train_step(sequence, learning_rate=0.5)
        print(f"Epoch {epoch+1} completed.")
        
    print(f"Training completed in {time.time() - start_time:.4f} seconds.\n")
    
    # テキスト生成テスト用のヘルパー関数（EOSストッパー付き）
    def generate_text(prompt_text: str, max_tokens: int = 10, temperature: float = 0.1):
        print(f"Prompt: '{prompt_text}'")
        prompt_ids = [word_to_id[w] for w in prompt_text.split() if w in word_to_id]
        
        if not prompt_ids:
            print("Unknown prompt words.")
            return
            
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
        
        generated_words = []
        for idx in generated_ids:
            word = id_to_word[idx]
            generated_words.append(word)
            
            # ピリオドなどの文末記号が出たら思考（生成）を停止する
            if word in [".", "!", "?"]:
                break
                
        result = prompt_text + " " + " ".join(generated_words)
        print(f"Generated: {result}\n")

    print("--- Text Generation ---")
    # "is" や "and" などの共通語から正しく分岐し、ピリオドで止まるかテスト
    
    # テスト1: SARA Engine -> is a fast ...
    generate_text("SARA Engine", max_tokens=8, temperature=0.1)
    
    # テスト2: The AI -> model is very ...
    generate_text("The AI", max_tokens=6, temperature=0.1)
    
    # テスト3: Biological brains -> are highly efficient ...
    generate_text("Biological brains", max_tokens=6, temperature=0.1)

if __name__ == "__main__":
    main()
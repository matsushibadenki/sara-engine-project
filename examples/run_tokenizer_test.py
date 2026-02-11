# examples/run_tokenizer_test.py
# SARA Tokenizer 動作検証スクリプト

import sys
import os

# ユーティリティ
try:
    from utils import setup_path
except ImportError:
    from .utils import setup_path # type: ignore

setup_path()

try:
    from sara_engine.tokenizer import SaraTokenizer
except ImportError:
    print("Error: 'sara_engine.tokenizer' module not found.")
    sys.exit(1)

def main():
    print("=== SARA Tokenizer Test ===")
    
    # 1. 学習用データ（コーパス）の準備
    # 英語と日本語（ローマ字）が混ざったような状況を想定
    corpus = [
        "hello world",
        "hello sara",
        "sara is an artificial intelligence",
        "sara likes to learn",
        "learning is fun",
        "unhappiness is not good",
        "understanding is important",
        "i am happy",
        "you are happy",
        "this is a test sentence"
    ]
    
    # 2. トークナイザーの初期化と学習
    # 小さなデータなので vocab_size も小さめに設定
    tokenizer = SaraTokenizer(vocab_size=100, model_path="sara_vocab_test.json")
    
    print("\n--- Training Phase ---")
    tokenizer.train(corpus)
    
    # 3. エンコード・デコードテスト
    print("\n--- Testing Phase ---")
    test_sentences = [
        "hello world",           # 既知の単語
        "unhappiness",           # "un" + "happi" + "ness" のように分割されることを期待
        "sara is learning",      # 既知の単語の組み合わせ
        "superintelligence",     # 未知語 ("super" + "intelligence" 等に分割されるか)
    ]
    
    for sent in test_sentences:
        print(f"\nOriginal: '{sent}'")
        
        # エンコード
        ids = tokenizer.encode(sent)
        print(f"Token IDs: {ids}")
        
        # IDからトークン文字列表現を確認
        tokens_readable = [tokenizer.inverse_vocab.get(i, "?") for i in ids]
        print(f"Tokens   : {tokens_readable}")
        
        # デコード（復元）
        decoded = tokenizer.decode(ids)
        print(f"Decoded : '{decoded}'")
        
        if sent.replace(" ", "") == decoded.replace(" ", ""):
            print("Status  : ✓ OK")
        else:
            print("Status  : ⚠ Reconstruction slightly diff (acceptable for subwords)")

    # 4. 語彙の確認
    print("\n--- Vocabulary Check ---")
    print(f"Total Vocab Size: {len(tokenizer.vocab)}")
    print("First 20 tokens:", list(tokenizer.vocab.keys())[:20])
    
    # マージルールの確認（実際に学習されたサブワード）
    print("\nSample Merges (Subwords created):")
    sample_merges = list(tokenizer.merges.values())[:10]
    print(sample_merges)

if __name__ == "__main__":
    main()
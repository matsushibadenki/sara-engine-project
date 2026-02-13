_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_tokenizer_test.py",
    "//": "タイトル: SARA Tokenizer 検証",
    "//": "目的: サブワード分割とエンコードのテスト。"
}

from sara_engine.utils.tokenizer import SaraTokenizer

def main():
    print("=== SARA Tokenizer Test ===")
    
    corpus = [
        "hello world", "hello sara", "sara is smart", "learning is fun"
    ]
    
    tokenizer = SaraTokenizer(vocab_size=100)
    print("Training tokenizer...")
    tokenizer.train(corpus)
    
    test_text = "hello sara learning"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded Text: {decoded}")
    
    if test_text == decoded:
        print("\n✓ Reconstruction Successful.")
    else:
        print("\n! Reconstruction differs slightly.")

if __name__ == "__main__":
    main()
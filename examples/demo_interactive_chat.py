_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_interactive_chat.py",
    "//": "ファイルの日本語タイトル: インタラクティブSNNチャット",
    "//": "ファイルの目的や内容: 学習フェーズをスキップし、TransformersライクなAutoクラスとパイプラインを利用して事前学習済みSNNモデルと即座に対話するデモ。"
}

import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine import AutoSpikeTokenizer, pipeline

def main():
    print("=== SNN Interactive Chat (Zero-shot Inference) ===")
    print("Initializing backprop-free, matrix-free biological neural network...\n")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'pipeline_demo'))
    model_path = os.path.join(workspace_dir, "saved_snn_model")
    tokenizer_path = os.path.join(workspace_dir, "tokenizer.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Pre-trained model not found at {model_path}.")
        print("Please run 'python examples/demo_snn_pipeline.py' first to train the STDP synapses.")
        return

    # HF Transformers style initialization!
    try:
        # Load Tokenizer using Auto Class
        tokenizer = AutoSpikeTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize pipeline by just passing the model path string
        generator = pipeline("text-generation", model=model_path, tokenizer=tokenizer)
        print("\nPipeline successfully initialized!")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Ending conversation. Goodbye!")
                break
            if not user_input.strip():
                continue
                
            # Generate text using biological spikes
            output = generator(user_input, max_new_tokens=30)
            full_generated_text = output[0]['generated_text']
            
            # Extract only the newly generated part for chat UI
            new_text = full_generated_text[len(user_input):].strip()
            print(f"SNN Agent: {new_text}")
            
        except KeyboardInterrupt:
            print("\nEnding conversation. Goodbye!")
            break

if __name__ == "__main__":
    main()
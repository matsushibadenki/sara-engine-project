_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_multimodal_pipeline.py",
    "//": "ファイルの日本語タイトル: マルチモーダルパイプラインのテスト",
    "//": "ファイルの目的や内容: テキスト（文字）と画像の特徴スパイクを同時に提示し、連合学習によってテキストから画像を想起できるか検証する。"
}

import os
import random
from sara_engine.models.multimodal_pipeline import SpikingMultimodalPipeline, MultimodalSNNConfig

def run_multimodal_demo():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "multimodal_pipeline.log")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Multimodal SNN Pipeline Demo ===\n")
        
        config = MultimodalSNNConfig(
            vocab_size=1000, embed_dim=64, num_layers=1, ffn_dim=128,
            vision_dim=256, audio_dim=128
        )
        model = SpikingMultimodalPipeline(config)
        
        # ダミーデータ: 'A'という文字のトークンと、それに紐づく架空の「画像特徴スパイク」
        text_token_apple = ord('A')
        vision_spikes_apple_img = random.sample(range(256), 20)
        
        f.write(f"\n[1] Learning Phase: Associating Text 'A' with Vision Pattern...\n")
        f.write(f"Vision Input Spikes: {vision_spikes_apple_img}\n")
        
        # 5回同時に刺激を与えて連合(STDP)を強化
        for _ in range(5):
            model.process_multimodal(
                text_token=text_token_apple, 
                vision_spikes=vision_spikes_apple_img, 
                learning=True
            )
            
        f.write("Learning complete.\n")
        
        f.write("\n[2] Recall Phase: Triggering with Text 'A' only...\n")
        # 画像スパイクを与えず、文字のトークンだけで処理
        result = model.process_multimodal(
            text_token=text_token_apple, 
            vision_spikes=None, 
            learning=False
        )
        
        vision_recall = result["vision_recall_from_text"]
        f.write(f"Recalled Vision Spikes from Text: {vision_recall}\n")
        
        if len(vision_recall) > 0:
            f.write("SUCCESS: Cross-modal association achieved. Text successfully triggered visual memory.\n")
        else:
            f.write("FAILED: No vision spikes recalled.\n")

    print(f"Log generated at: {log_file_path}")

if __name__ == "__main__":
    run_multimodal_demo()
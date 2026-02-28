_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_sara_edge.py",
    "//": "ファイルの日本語タイトル: Sara-Edge ランタイム デモ",
    "//": "ファイルの目的や内容: SNNTransformerModelを学習させ、その重みをSara-Edge用にエクスポートし、超軽量なエッジランタイムで推論（テキスト生成）が完全に一致するかをテストする。"
}

import os
import time
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from sara_engine.edge.exporter import export_for_edge
from sara_engine.edge.runtime import SaraEdgeRuntime

def run_edge_demo():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    edge_model_path = os.path.join(workspace_dir, "edge_model.json")
    log_file_path = os.path.join(workspace_dir, "sara_edge_demo.log")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Sara-Edge Lightweight Runtime Demo ===\n\n")

        # 1. フルモデルでの学習
        f.write("[1] Training Full SpikingTransformerModel...\n")
        config = SNNTransformerConfig(embed_dim=64, num_layers=1)
        full_model = SpikingTransformerModel(config)
        
        training_text = "Sara-Edge brings biological AI to microcontrollers."
        full_model.learn_sequence(training_text)
        
        prompt = "Sara-Edge brings"
        full_model_output = full_model.generate(prompt, max_length=35)
        f.write(f"Full Model Output: {full_model_output}\n\n")

        # 2. エッジ用フォーマットへのエクスポート
        f.write(f"[2] Exporting to Edge format: {edge_model_path}\n")
        export_for_edge(full_model, edge_model_path)
        file_size = os.path.getsize(edge_model_path) / 1024
        f.write(f"Exported Model Size: {file_size:.2f} KB\n\n")

        # 3. エッジランタイムでの推論
        f.write("[3] Running Inference on SaraEdgeRuntime...\n")
        start_time = time.time()
        edge_runtime = SaraEdgeRuntime(edge_model_path)
        edge_output = edge_runtime.generate(prompt, max_length=35)
        inference_time = time.time() - start_time
        
        f.write(f"Edge Runtime Output: {edge_output}\n")
        f.write(f"Inference Time: {inference_time:.4f} seconds\n\n")

        # 4. 整合性チェック
        if full_model_output == edge_output:
            f.write("SUCCESS: Edge Runtime output matches Full Model perfectly!\n")
            print("SUCCESS: Edge Runtime matches Full Model.")
        else:
            f.write("WARNING: Outputs do not match.\n")
            print("WARNING: Mismatch in outputs.")

    print(f"Log generated at: {log_file_path}")

if __name__ == "__main__":
    run_edge_demo()
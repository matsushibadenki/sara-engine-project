# ディレクトリパス: examples/integrated_data_visualization_demo.py
# ファイルの日本語タイトル: 統合データローダー & 可視化（Sara-Board）デモ
# ファイルの目的や内容: SpikeStreamLoaderによるデータのストリーミング、SNNモデルでの処理、
#   およびSaraBoardを用いた多層的な発火状態の可視化とダッシュボード出力を統合して実証する。

import os
import sys
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig  # noqa: E402
from sara_engine.utils.data.dataloader import SpikeStreamLoader  # noqa: E402
from sara_engine.utils.sara_board import SaraBoard  # noqa: E402


def char_to_spikes(char: str) -> list[int]:
    """ASCIIコードをスパイクIDに変換するシンプルなエンコーダー"""
    return [ord(char)]


def main():
    print("=== SARA Engine: Integrated Data Loading & Visualization Demo ===\n")

    # ワークスペースとログディレクトリの準備
    workspace_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../workspace/integrated_viz'))
    os.makedirs(workspace_dir, exist_ok=True)

    # 1. 可視化ツール (Sara-Board) の初期化
    print("[1] Initializing Sara-Board for multi-layer tracking...")
    board = SaraBoard(log_dir=workspace_dir)

    # 2. モデルの初期化 (可視化用に軽量な設定)
    config = SNNTransformerConfig(embed_dim=128, num_layers=1)
    model = SpikingTransformerModel(config)
    print("[2] Spiking Transformer Model initialized.")

    # 3. データストリームの準備 (テキストデータをスパイク流に変換)
    raw_text = "SARA-Engine: Neuromorphic Computing without Backprop."
    print(f"[3] Preparing SpikeStreamLoader for text: '{raw_text}'")
    loader = SpikeStreamLoader(dataset=list(
        raw_text), encode_fn=char_to_spikes, time_step=1)

    # 4. ストリーム処理とリアルタイム・ロギング
    print("\n[4] Processing stream and logging dynamics...")
    start_time = time.time()

    for event in loader.stream():
        step = event["time"]
        input_spikes = event["spikes"]
        token_id = input_spikes[0]

        # モデルの実行 (STDP学習あり)
        # 内部でアテンションやリザーバーの発火が発生する
        model.forward_step(token_id, learning=True, target_id=token_id)

        # --- Sara-Boardへの多角的なロギング ---

        # (A) 入力スパイクの記録
        board.log_spikes(
            step, layer_name="Layer 0: Input Stream", spikes=input_spikes)

        # (B) リザーバー/コンテキスト層の発火状況を取得して記録
        res_spikes = model._get_reservoir_spikes(token_id)
        board.log_spikes(
            step, layer_name="Layer 1: Context Reservoir", spikes=res_spikes)

        # (C) 読み出し層 (Readout) の活動状況を記録
        # 重みが閾値を超えている「予測された」スパイクを抽出
        active_readouts = []
        if token_id < len(model.readout_synapses):
            active_readouts = [
                k for k, w in model.readout_synapses[token_id].items() if w > 0.6]

        if active_readouts:
            board.log_spikes(
                step, layer_name="Layer 2: Readout Activity", spikes=active_readouts)

        if step % 10 == 0:
            print(f"    Step {step:03d}: Logged activity across 3 layers.")

    # 5. 可視化アウトプットの生成
    print("\n[5] Generating Visualization Artifacts...")

    # 全体を統合したHTMLダッシュボードのエクスポート (ラスタープロット含む)
    html_path = board.export_html("integrated_dashboard.html")

    duration = time.time() - start_time
    print("-" * 50)
    print(f"Execution Complete: {duration:.2f} seconds")
    print(f"Artifacts saved in: {workspace_dir}")
    print(f"Dashboard URL: file://{os.path.abspath(html_path)}")
    print("-" * 50)


if __name__ == "__main__":
    main()

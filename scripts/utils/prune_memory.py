# ディレクトリパス: scripts/utils/prune_memory.py
# ファイルの日本語タイトル: 記憶の刈り込み（プルーニング）ツール
# ファイルの目的や内容: SNNモデルの記憶マップから、重みの低い不要なシナプス結合（ノイズ）を削除し、モデルを軽量・高速化する。

import os
import msgpack

def prune_model_memory(model_path, threshold=50.0):
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return

    print(f"--- 記憶の刈り込みを開始します: {model_path} ---")
    print(f"（閾値: {threshold} 未満の弱いシナプス結合を削除します）")
    
    file_size_before = os.path.getsize(model_path)
    
    # 1. モデルの読み込み
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    raw_map = state.get("direct_map", {})
    
    original_patterns = len(raw_map)
    original_synapses = sum(len(v) for v in raw_map.values())
    
    pruned_map = {}
    pruned_synapses = 0
    
    # 2. 刈り込み（プルーニング）処理
    for k_str, target_dict in raw_map.items():
        # 重みが閾値以上のシナプスだけを残す
        filtered_targets = {tk: tv for tk, tv in target_dict.items() if tv >= threshold}
        
        # 遷移先（連想先）が1つでも残っている場合のみ、その発火パターン（SDR）を記憶しておく
        if filtered_targets:
            pruned_map[k_str] = filtered_targets
            pruned_synapses += len(filtered_targets)
            
    # 3. 更新された記憶で上書き保存
    state["direct_map"] = pruned_map
    
    with open(model_path, "wb") as f:
        msgpack.pack(state, f)
        
    file_size_after = os.path.getsize(model_path)
    
    # 4. 効果のレポート
    print("\n✨ 刈り込みが完了しました！")
    print(f"  - 記憶パターン数: {original_patterns:,} -> {len(pruned_map):,} (削除: {original_patterns - len(pruned_map):,})")
    print(f"  - シナプス結合数: {original_synapses:,} -> {pruned_synapses:,} (削除: {original_synapses - pruned_synapses:,})")
    print(f"  - ファイルサイズ: {file_size_before / 1024 / 1024:.2f} MB -> {file_size_after / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    prune_model_memory("models/distilled_sara_llm.msgpack")
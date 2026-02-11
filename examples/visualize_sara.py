# examples/visualize_sara.py
# title: SARA Brain Visualizer (Fixed Path)
# description: SARAの脳内（L1, L2, L3のスパイク発火）をリアルタイムで可視化しながらチャットする

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- パス設定の修正 (ここを強化) ---
# 現在のファイル(visualize_sara.py)の場所から見て ../src を絶対パスで取得
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_dir)

try:
    from sara_engine.sara_gpt_core import SaraGPT
except ImportError as e:
    print(f"Error: {e}")
    print(f"Failed to import 'sara_engine' from {src_dir}")
    print("Please ensure 'src/sara_engine/__init__.py' exists.")
    sys.exit(1)
# ------------------------------------

def main():
    print("Initializing SARA Visualizer...")
    
    # モデル読み込み
    engine = SaraGPT(sdr_size=1024)
    model_path = "sara_brain.pkl"
    
    if os.path.exists(model_path):
        print(f"Loading brain from {model_path}...")
        engine.load_model(model_path)
    else:
        print("Error: sara_brain.pkl not found. Please run chat_sara.py first.")
        return

    # --- 可視化の準備 ---
    plt.ion() # インタラクティブモード
    fig, (ax_brain, ax_readout) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.manager.set_window_title('SARA Engine - Realtime Spike Monitor')
    
    # ニューロンの座標をランダムに配置 (可視化用)
    n_neurons = 2000
    l1_x = np.random.rand(n_neurons) * 1.0
    l1_y = np.random.rand(n_neurons) * 1.0
    
    l2_x = np.random.rand(n_neurons) * 1.0 + 1.2 # 横にずらす
    l2_y = np.random.rand(n_neurons) * 1.0
    
    l3_x = np.random.rand(n_neurons) * 1.0 + 2.4 # さらにずらす
    l3_y = np.random.rand(n_neurons) * 1.0

    print("\n--- Visual Chat Started ---")
    print("Type your message in the console. Watch the window for brain activity.")
    
    vocab_list = ["hello", "world", "sara", "is", "a", "good", "ai", "cat", "dog", "likes", "fish", "meat", "sleeps", "on", "bed", "floor", "i", "like", "nice", "are", "cats"]
    
    while True:
        try:
            user_input = input("\nYou: ").strip().lower()
            if user_input in ["exit", "quit"]: break
            if not user_input: continue

            for w in user_input.split():
                if w not in vocab_list: vocab_list.append(w)

            # 1. Listen & Visualize
            words = user_input.split()
            print("SARA is listening...")
            for w in words:
                sdr = engine.encoder.encode(w)
                _, all_spikes = engine.forward_step(sdr, training=False)
                
                update_plot(ax_brain, ax_readout, all_spikes, engine)
                plt.pause(0.1)

            # 2. Think & Visualize
            print(f"SARA: ", end="")
            generated = []
            empty_sdr = []
            trigger_text = user_input
            
            for _ in range(20):
                predicted_sdr, all_spikes = engine.forward_step(empty_sdr, training=False, force_output=True)
                
                update_plot(ax_brain, ax_readout, all_spikes, engine)
                plt.pause(0.05)
                
                search_vocab = vocab_list + ["<eos>"] if "<eos>" not in vocab_list else vocab_list
                next_word = engine.encoder.decode(predicted_sdr, search_vocab)
                
                # Repetition Penalty (Simple)
                if len(generated) > 0 and generated[-1] == next_word:
                     candidates = [w for w in vocab_list if w != next_word]
                     if candidates: next_word = np.random.choice(candidates)

                if next_word == "<eos>":
                    if len(generated) == 0: 
                         triggers = trigger_text.split()
                         if triggers:
                             empty_sdr = engine.encoder.encode(triggers[-1])
                             continue
                    break
                
                print(f"{next_word} ", end="", flush=True)
                generated.append(next_word)
                empty_sdr = engine.encoder.encode(next_word)
            
            print()
            engine.relax(50)
            
        except KeyboardInterrupt:
            break

    plt.ioff()
    plt.show()

def update_plot(ax_main, ax_sub, spikes, engine):
    ax_main.clear()
    ax_sub.clear()
    
    ax_main.set_title(f"Liquid State Machine Activity (L1->Fast, L2->Medium, L3->Slow)")
    ax_main.set_xlim(-0.1, 3.5)
    ax_main.set_ylim(-0.1, 1.1)
    ax_main.axis('off')
    
    ax_main.text(0.5, 1.05, "L1 (Fast)", ha='center')
    ax_main.text(1.7, 1.05, "L2 (Medium)", ha='center')
    ax_main.text(2.9, 1.05, "L3 (Slow)", ha='center')

    l1_spikes = [i for i in spikes if i < 2000]
    l2_spikes = [i-2000 for i in spikes if 2000 <= i < 4000]
    l3_spikes = [i-4000 for i in spikes if 4000 <= i < 6000]
    
    if l1_spikes:
        ax_main.scatter(np.random.rand(len(l1_spikes)), np.random.rand(len(l1_spikes)), c='blue', s=10, alpha=0.6)
    if l2_spikes:
        ax_main.scatter(np.random.rand(len(l2_spikes)) + 1.2, np.random.rand(len(l2_spikes)), c='green', s=10, alpha=0.6)
    if l3_spikes:
        ax_main.scatter(np.random.rand(len(l3_spikes)) + 2.4, np.random.rand(len(l3_spikes)), c='red', s=10, alpha=0.6)

    potentials = engine.readout_v
    ax_sub.set_title("Readout Neuron Potentials (Accumulated Evidence)")
    ax_sub.set_ylim(0, 1.0)
    ax_sub.bar(range(len(potentials)), potentials, color='purple')
    
    plt.draw()

if __name__ == "__main__":
    main()
# examples/run_chat.py
# SARAチャットボット & 可視化ツール (v2.0: Core v48対応)

import sys
import os
import time
import argparse
import numpy as np

# ユーティリティ
try:
    from utils import setup_path
except ImportError:
    from .utils import setup_path # type: ignore

setup_path()

try:
    # 注意: SaraGPTの実装も SaraEngine v48 の変更(sleep_phase)に対応している必要があります
    from sara_engine import SaraGPT
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

# 可視化用ライブラリ（オプション）
try:
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class ChatRunner:
    def __init__(self, use_visualizer=False, model_path="sara_brain.pkl"):
        self.engine = SaraGPT(sdr_size=1024)
        self.model_path = model_path
        self.use_visualizer = use_visualizer
        self.vocab_list = []
        
        # 可視化用の初期設定
        self.fig = None
        self.ax_brain = None
        self.ax_readout = None
        
        if self.use_visualizer and not VISUALIZATION_AVAILABLE:
            print("Warning: matplotlib not found. Visualization disabled.")
            self.use_visualizer = False

    def load_or_train(self, force_train=False):
        # 基本コーパス
        corpus = [
            "hello sara", "hello world", "hi there",
            "good morning", "good night",
            "who are you?", "i am sara", "i am an ai",
            "what is your name?", "my name is sara",
            "sara is smart", "sara likes to learn",
            "the cat likes fish", "the dog likes meat",
            "birds fly in sky", "fish swim in water",
            "i like cats", "i like dogs", "i love learning",
            "what is love?", "love is good",
            "thinking is fun"
        ]
        
        # 語彙リスト生成
        vocab = set()
        for sent in corpus:
            for w in sent.split(): vocab.add(w)
        self.vocab_list = sorted(list(vocab))
        
        if not force_train and os.path.exists(self.model_path):
            print(f"Loading brain from {self.model_path}...")
            self.engine.load_model(self.model_path)
        else:
            print(f"Training on {len(corpus)} sentences...")
            epochs = 50
            for epoch in range(epochs):
                shuffled = np.random.permutation(corpus)
                for sent in shuffled:
                    self.engine.train_sequence(sent.split())
                
                # 修正: Core v48の適応型プルーニングを適用
                # チャットボットの場合はデータ数が少ないため、サンプルサイズを考慮して渡す
                if hasattr(self.engine, 'sleep_phase'):
                    self.engine.sleep_phase(epoch=epoch, sample_size=len(corpus))
                
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} done.")
            self.engine.save_model(self.model_path)

    def init_plot(self):
        if not self.use_visualizer: return
        
        plt.ion()
        self.fig, (self.ax_brain, self.ax_readout) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}
        )
        self.fig.canvas.manager.set_window_title('SARA Engine - Realtime Spike Monitor')
        print("Visualization window opened.")

    def update_plot(self, spikes):
        if not self.use_visualizer: return
        
        self.ax_brain.clear()
        self.ax_readout.clear()
        
        self.ax_brain.set_title("Liquid State Machine Activity (L1->Fast, L2->Medium, L3->Slow)")
        self.ax_brain.set_xlim(-0.1, 3.5)
        self.ax_brain.set_ylim(-0.1, 1.1)
        self.ax_brain.axis('off')
        
        self.ax_brain.text(0.5, 1.05, "L1 (Fast)", ha='center')
        self.ax_brain.text(1.7, 1.05, "L2 (Medium)", ha='center')
        self.ax_brain.text(2.9, 1.05, "L3 (Slow)", ha='center')

        # スパイクの描画
        l1 = [i for i in spikes if i < 2000]
        l2 = [i-2000 for i in spikes if 2000 <= i < 4000]
        l3 = [i-4000 for i in spikes if 4000 <= i < 6000]
        
        if l1: self.ax_brain.scatter(np.random.rand(len(l1)), np.random.rand(len(l1)), c='blue', s=10, alpha=0.6)
        if l2: self.ax_brain.scatter(np.random.rand(len(l2)) + 1.2, np.random.rand(len(l2)), c='green', s=10, alpha=0.6)
        if l3: self.ax_brain.scatter(np.random.rand(len(l3)) + 2.4, np.random.rand(len(l3)), c='red', s=10, alpha=0.6)

        # 電位の描画
        # SaraGPTの構造によっては readout_v がない場合があるのでチェック
        if hasattr(self.engine, 'readout_v'):
            potentials = self.engine.readout_v
            self.ax_readout.set_title("Readout Neuron Potentials")
            self.ax_readout.set_ylim(0, 1.0)
            self.ax_readout.bar(range(len(potentials)), potentials, color='purple')
        
        plt.draw()
        plt.pause(0.05)

    def run(self):
        print("\n--- SARA Chat Started ---")
        print("Commands: 'exit' to quit, 'save' to save model.")
        
        if self.use_visualizer:
            self.init_plot()
            print("Note: Check the visualization window for brain activity.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip().lower()
                if user_input in ["exit", "quit", "bye"]:
                    print("Dreaming and saving...")
                    # 修正: dream機能がない場合のエラー回避
                    if hasattr(self.engine, 'dream'):
                        self.engine.dream(cycles=5)
                    self.engine.save_model(self.model_path)
                    break
                
                if not user_input: continue
                
                # 新しい単語の登録
                for w in user_input.split():
                    if w not in self.vocab_list:
                        self.vocab_list.append(w)
                
                # 1. Listen & Visualize
                words = user_input.split()
                if self.use_visualizer:
                    print("SARA is listening...")
                    for w in words:
                        sdr = self.engine.encoder.encode(w)
                        # forward_stepの戻り値を調整 (SNNの仕様に合わせて)
                        result = self.engine.forward_step(sdr, training=False)
                        all_spikes = result[1] if isinstance(result, tuple) else []
                        self.update_plot(all_spikes)
                    self.engine.listen(user_input, online_learning=True)
                else:
                    self.engine.listen(user_input, online_learning=True)

                # 2. Think
                print("SARA: ", end="")
                generated = []
                empty_sdr = []
                trigger_text = user_input
                
                # 生成ループ
                for _ in range(20):
                    predicted_sdr, all_spikes = self.engine.forward_step(
                        empty_sdr, training=False, force_output=True
                    )
                    
                    if self.use_visualizer:
                        self.update_plot(all_spikes)
                    
                    # デコード
                    search_vocab = self.vocab_list + ["<eos>"]
                    next_word = self.engine.encoder.decode(predicted_sdr, search_vocab)
                    
                    # リピート回避（簡易版）
                    if generated and generated[-1] == next_word:
                        candidates = [w for w in self.vocab_list if w != next_word]
                        if candidates: next_word = np.random.choice(candidates)

                    if next_word == "<eos>":
                        if not generated and trigger_text:
                            triggers = trigger_text.split()
                            if triggers:
                                empty_sdr = self.engine.encoder.encode(triggers[-1])
                                continue
                        break
                    
                    print(f"{next_word} ", end="", flush=True)
                    generated.append(next_word)
                    empty_sdr = self.engine.encoder.encode(next_word)
                
                print() # Newline
                if hasattr(self.engine, 'relax'):
                    self.engine.relax(50)
                
            except KeyboardInterrupt:
                break
        
        if self.use_visualizer:
            plt.ioff()
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="SARA Chatbot Demo")
    parser.add_argument("--visualize", "-v", action="store_true", help="Enable realtime visualization")
    parser.add_argument("--train", action="store_true", help="Force retraining")
    parser.add_argument("--model", type=str, default="sara_brain.pkl", help="Model file path")
    
    args = parser.parse_args()
    
    runner = ChatRunner(use_visualizer=args.visualize, model_path=args.model)
    runner.load_or_train(force_train=args.train)
    runner.run()

if __name__ == "__main__":
    main()
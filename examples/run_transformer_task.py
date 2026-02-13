_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_transformer_task.py",
    "//": "タイトル: SARA Transformer Test (v7: Deep Context)",
    "//": "目的: L3層の文脈維持能力を診断しつつ学習。"
}

import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from sara_engine import SaraGPT
from sara_engine.utils.visualizer import SaraVisualizer

def run_transformer_test():
    print("=== SARA Transformer Evolution Test (v7: Deep Context) ===")
    
    save_dir = "workspace/transformer_logs"
    os.makedirs(save_dir, exist_ok=True)
    
    brain = SaraGPT(sdr_size=1024)
    
    if hasattr(brain.l1, 'use_rust') and brain.l1.use_rust:
        print("✓ Rust Acceleration: ACTIVE")
    else:
        print("! Running in Pure Python Mode (Slow)")

    # --- 診断: 文脈の分離度 ---
    print("\n[Diagnosis] Checking Context Divergence...")
    # 1. Sky Context
    brain.reset_state()
    brain.encoder.encode("sky") # Prime encoder
    for word in ["sky", "is"]:
        sdr = brain.encoder.encode(word)
        for _ in range(8): brain.forward_step(sdr, training=False)
    state_sky = set(brain.prev_spikes[2]) # L3 state

    # 2. Fire Context
    brain.reset_state()
    for word in ["fire", "is"]:
        sdr = brain.encoder.encode(word)
        for _ in range(8): brain.forward_step(sdr, training=False)
    state_fire = set(brain.prev_spikes[2]) # L3 state
    
    # Overlap Check
    overlap = len(state_sky.intersection(state_fire))
    total = (len(state_sky) + len(state_fire)) / 2
    similarity = (overlap / total * 100) if total > 0 else 0
    
    print(f"L3 State Overlap between 'sky...is' and 'fire...is': {overlap} neurons ({similarity:.1f}%)")
    if similarity > 80:
        print("⚠ WARNING: High overlap! The brain is confusing the contexts.")
    else:
        print("✓ GOOD: Contexts are distinct.")

    # --- 学習 ---
    print("\nTask: Sequence Context Learning")
    sequences = [
        "sky is blue",
        "grass is green",
        "fire is red",
        "snow is white"
    ]
    
    print(f"Training sequences: {sequences}")
    print("Starting training (Competitive Learning)...")
    
    start_time = time.time()
    epochs = 40
    
    activity_log = []
    
    for epoch in range(epochs):
        epoch_spikes = 0
        train_seqs = sequences.copy()
        random.shuffle(train_seqs) 
        
        for seq in train_seqs:
            brain.reset_state()
            words = seq.split()
            
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i+1]
                
                input_sdr = brain.encoder.encode(current_word)
                target_sdr = brain.encoder.encode(next_word)
                
                # 8ステップ反復
                for _ in range(8):
                    _, spikes = brain.forward_step(input_sdr, training=True, target_sdr=target_sdr)
                    epoch_spikes += len(spikes)
        
        activity_log.append(epoch_spikes)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Activity: {epoch_spikes}", end='\r')
        
    duration = time.time() - start_time
    print(f"\nTraining finished in {duration:.2f} sec.")

    # 評価
    print("\n=== Evaluation: Context Completion ===")
    brain.attention_active = True
    
    test_prompts = ["sky is", "fire is"]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        brain.reset_state()
        
        for word in prompt.split():
            sdr = brain.encoder.encode(word)
            for _ in range(5):
                brain.forward_step(sdr, training=False)
        
        print("SARA predicts: ", end="")
        input_sdr = [] 
        
        generated_word = ""
        for _ in range(20): 
            pred_sdr, _ = brain.forward_step(input_sdr, training=False, force_output=True)
            
            if pred_sdr:
                candidates = ["blue", "green", "red", "white", "sky", "grass", "fire", "snow"]
                word = brain.encoder.decode(pred_sdr, candidates)
                if word != "<unk>":
                    print(f"[{word}]", end=" ")
                    generated_word = word
                    break
        
        if not generated_word:
            print("(No confident prediction)")
            
    print("\nDone.")

if __name__ == "__main__":
    run_transformer_test()
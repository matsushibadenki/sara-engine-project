_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_audio_demo.py",
    "//": "タイトル: 聴覚SNNデモ",
    "//": "目的: 音声をスパイクに変換し、SARAが音を聞き分ける様子を可視化する。"
}

import numpy as np
import os
import matplotlib.pyplot as plt
from sara_engine import SaraGPT, SaraVisualizer, AudioSpikeEncoder

def run_audio_demo():
    print("=== SARA Engine: Audio Hearing Demo ===")
    print("Simulating cochlear processing (Tonotopy)...")
    
    # 1. 音声データの生成 (ド・ミ・ソ)
    encoder = AudioSpikeEncoder(num_neurons=1024, sample_rate=44100)
    
    print("Generating synthetic audio (Do-Mi-So)...")
    # C4 (261.6Hz), E4 (329.6Hz), G4 (392.0Hz)
    tone_c = encoder.generate_tone(261.6, 0.2) # Do
    tone_e = encoder.generate_tone(329.6, 0.2) # Mi
    tone_g = encoder.generate_tone(392.0, 0.2) # So
    silence = np.zeros(int(44100 * 0.1), dtype=np.float32) # 無音
    
    # 連結: Do -> 無音 -> Mi -> 無音 -> So
    full_audio = np.concatenate([tone_c, silence, tone_e, silence, tone_g])
    
    # 2. スパイクへの変換 (エンコード)
    print("Encoding audio to spikes...")
    input_spike_train = encoder.encode_signal(full_audio)
    print(f"Total Time Steps: {len(input_spike_train)}")
    
    # 3. SARA Engineのセットアップ
    # 音声専用の脳として初期化
    brain = SaraGPT(sdr_size=1024)
    
    # 可視化のためにPythonモードで初期化
    for layer in brain.layers:
        layer.use_rust = False
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=0.5, # 音声は減衰を少し速めに（反響を抑える）
            density=layer.density,
            input_scale=1.5,
            rec_scale=layer.rec_scale,
            feedback_scale=layer.feedback_scale,
            use_rust=False
        )

    # 4. シミュレーション実行
    print("Listening...")
    spike_history_l1 = [] # 入力層に近いレイヤー (聴覚野)
    spike_history_l2 = [] # 処理層
    
    for t, input_spikes in enumerate(input_spike_train):
        if t % 10 == 0:
            print(f"Step {t}/{len(input_spike_train)}", end='\r')
            
        # 順伝播
        # 音声は既にスパイク化されているので、encoder.encode()は不要
        # input_spikes を直接 forward_step に渡す
        _, _ = brain.forward_step(input_spikes, training=False)
        
        # 履歴記録
        spike_history_l1.append(brain.prev_spikes[0])
        spike_history_l2.append(brain.prev_spikes[1])
        
    print("\nProcessing complete.")
    
    # 5. 可視化
    save_dir = "workspace/audio_logs"
    os.makedirs(save_dir, exist_ok=True)
    viz = SaraVisualizer(save_dir=save_dir)
    
    print("Generating spectrogram-like spike raster...")
    
    # 音声入力のスパイク（擬似スペクトログラム）
    # 入力スパイク列自体も可視化して、正しく周波数分解できているか確認
    viz.plot_raster(input_spike_train, 
                   title="Input Audio Spikes (Simulated Cochlea)", 
                   filename="audio_input_raster.png")
    
    # 脳の反応
    viz.plot_raster(spike_history_l1, 
                   title="Auditory Cortex Response (Layer 1)", 
                   filename="audio_cortex_response.png")
    
    print(f"\nSaved visualization to '{save_dir}/'.")
    print("Check 'audio_input_raster.png'. If successful, you should see three distinct bands corresponding to Do, Mi, and So.")

if __name__ == "__main__":
    run_audio_demo()
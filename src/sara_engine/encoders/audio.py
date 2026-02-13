_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/audio.py",
    "//": "タイトル: 聴覚エンコーダ (AudioSpikeEncoder)",
    "//": "目的: 音声波形を周波数成分ごとのスパイク列に変換する（トノトピーの実装）。"
}

import numpy as np
import wave
import struct
from typing import List, Tuple, Union

class AudioSpikeEncoder:
    """
    音声データをSNN用のスパイク列に変換するエンコーダ。
    生物の蝸牛（かぎゅう）を模倣し、周波数成分をニューロンの位置（インデックス）にマッピングします。
    
    手法:
    1. 短時間フーリエ変換 (STFT) で時間を区切って周波数分解
    2. パワースペクトルを計算
    3. 各周波数帯の強度をポアソン発火確率（Rate Coding）に変換
    """
    def __init__(self, num_neurons: int = 1024, sample_rate: int = 44100, 
                 window_ms: int = 20, step_ms: int = 10, min_freq: int = 20, max_freq: int = 8000):
        self.num_neurons = num_neurons
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_ms / 1000)
        self.step_size = int(sample_rate * step_ms / 1000)
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # ハミング窓の生成（スペクトル漏れ対策）
        self.window_func = np.hamming(self.window_size)

    def encode_file(self, filepath: str) -> List[List[int]]:
        """WAVファイルを読み込んでエンコード"""
        try:
            with wave.open(filepath, 'rb') as wf:
                # パラメータ取得
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # データ読み込み
                raw_data = wf.readframes(n_frames)
                
                # numpy配列への変換 (16bit PCMを想定)
                if sampwidth == 2:
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                else:
                    raise ValueError("Currently only supports 16-bit WAV files.")
                
                # ステレオの場合はモノラル化（平均）
                if n_channels == 2:
                    data = data.reshape(-1, 2).mean(axis=1)
                
                # 正規化 (-1.0 ~ 1.0)
                data /= 32768.0
                
                # サンプリングレートが異なる場合は簡易リサンプリングが必要だが、
                # 今回はデモ用としてエンコーダ設定をファイルに合わせることを推奨
                if framerate != self.sample_rate:
                    print(f"Warning: File sample rate ({framerate}Hz) differs from encoder setting ({self.sample_rate}Hz).")
                    
                return self.encode_signal(data)
                
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return []

    def encode_signal(self, signal: np.ndarray) -> List[List[int]]:
        """
        1次元の音声信号配列をスパイク列に変換
        Returns: [ [neuron_idx, ...], [neuron_idx, ...], ... ] (Time steps)
        """
        spike_train = []
        num_samples = len(signal)
        
        # FFTの周波数ビン数を計算
        fft_size = self.window_size
        freq_bins = np.fft.rfftfreq(fft_size, d=1.0/self.sample_rate)
        
        # 使用する周波数帯域のインデックス範囲
        valid_indices = np.where((freq_bins >= self.min_freq) & (freq_bins <= self.max_freq))[0]
        
        if len(valid_indices) == 0:
            return []

        # 周波数ビンをニューロン数に合わせてマッピングするためのインデックス配列
        # (線形補間のような簡易マッピング)
        freq_to_neuron = np.linspace(valid_indices[0], valid_indices[-1], self.num_neurons).astype(int)
        
        # スライディングウィンドウ処理
        for start in range(0, num_samples - self.window_size, self.step_size):
            end = start + self.window_size
            segment = signal[start:end]
            
            # 窓関数適用とFFT
            windowed = segment * self.window_func
            spectrum = np.abs(np.fft.rfft(windowed))
            
            # パワースペクトルをニューロンにマッピング
            # 指定された周波数帯域の強度を取り出す
            neuron_activities = spectrum[freq_to_neuron]
            
            # 強度の正規化 (0.0 ~ 1.0)
            max_val = np.max(neuron_activities)
            if max_val > 1e-6:
                neuron_activities /= max_val
            
            # スパイク生成 (Rate Coding)
            # 強度が強いほど高確率で発火
            # 感度係数 (sensitivity) で発火しやすさを調整
            sensitivity = 0.8
            probs = neuron_activities * sensitivity
            rand_vals = np.random.rand(self.num_neurons)
            
            fired_indices = np.where(rand_vals < probs)[0].tolist()
            spike_train.append(fired_indices)
            
        return spike_train

    def generate_tone(self, freq: float, duration: float, volume: float = 0.5) -> np.ndarray:
        """テスト用の純音生成ユーティリティ"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        tone = volume * np.sin(2 * np.pi * freq * t)
        return tone.astype(np.float32)
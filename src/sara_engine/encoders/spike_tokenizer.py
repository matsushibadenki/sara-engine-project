# filepath: src/sara_engine/encoders/spike_tokenizer.py
# title: スパイクトークナイザー
# description: テキストをスパイク列に変換し、時間的位相コーディングで位置情報を付与する多言語対応エンコーダー。Token-to-Spike EmbeddingsとTemporal Positional Encodingを担う。

import random

class SpikeTokenizer:
    def __init__(self, vocab_size=65536, sdr_size=1024, active_bits=32):
        self.vocab_size = vocab_size
        self.sdr_size = sdr_size
        self.active_bits = active_bits
        self.token_to_sdr = {} 

    def _generate_sdr(self, token_id):
        # Generate Sparse Distributed Representation (SDR) for a token
        if token_id not in self.token_to_sdr:
            random.seed(token_id)
            indices = list(range(self.sdr_size))
            random.shuffle(indices)
            self.token_to_sdr[token_id] = indices[:self.active_bits]
        return self.token_to_sdr[token_id]

    def encode(self, text, time_window=10):
        # Multi-language support by using unicode code points
        tokens = [ord(c) for c in text]
        spike_trains = [] 
        
        for pos, token in enumerate(tokens):
            sdr = self._generate_sdr(token)
            
            # Temporal Positional Encoding: Time-to-First-Spike
            # Spikes for tokens appearing later are delayed in time
            base_time = pos * time_window
            spike_trains.append((base_time, sdr))
            
        return spike_trains
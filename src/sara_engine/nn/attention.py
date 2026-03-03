_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/attention.py",
    "//": "ファイルの日本語タイトル: 高速化版スパイキング・アテンション",
    "//": "ファイルの目的や内容: sara_rust_core.SpikeEngine を統合し、大規模なスパイク伝播と学習を高速化したアテンション層。Fuzzy Recall (SDR Overlap) による連想記憶を統合。Transformers代替となるSDRFuzzyAttentionを追記。"
}

import random
from typing import List, Dict, Set, Optional, Tuple
from .module import SNNModule

# Import Rust core if available
try:
    from sara_engine import sara_rust_core # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class SpikeSelfAttention(SNNModule):
    def __init__(self, embed_dim: int, density: float = 0.1, context_size: int = 64, use_rust: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.use_rust = use_rust and RUST_AVAILABLE
        
        # Initialize weights
        self.q_weights: List[Dict[int, float]] = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.k_weights: List[Dict[int, float]] = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.v_weights: List[Dict[int, float]] = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.o_weights: List[Dict[int, float]] = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        # Setup Rust engine if used
        if self.use_rust:
            self.q_engine = sara_rust_core.SpikeEngine()
            self.k_engine = sara_rust_core.SpikeEngine()
            self.v_engine = sara_rust_core.SpikeEngine()
            self.o_engine = sara_rust_core.SpikeEngine()
            self._sync_to_rust()

        self.register_state("q_weights")
        self.register_state("k_weights")
        self.register_state("v_weights")
        self.register_state("o_weights")
        
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []

    def _sync_to_rust(self):
        """Sync Python weights to Rust core"""
        if self.use_rust:
            self.q_engine.set_weights(self.q_weights)
            self.k_engine.set_weights(self.k_weights)
            self.v_engine.set_weights(self.v_weights)
            self.o_engine.set_weights(self.o_weights)

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights: List[Dict[int, float]] = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num = max(1, int(out_dim * density))
            for t in random.sample(range(out_dim), num):
                weights[i][t] = random.uniform(0.1, 1.0)
        return weights

    def forward(self, x_spikes: List[int], learning: bool = False) -> List[int]:
        threshold = 1.0 if learning else 0.5
        max_out = max(1, int(self.embed_dim * 0.15))

        if self.use_rust:
            # Fast propagation via Rust engine
            q_list = self.q_engine.propagate(x_spikes, threshold, max_out)
            k_list = self.k_engine.propagate(x_spikes, threshold, max_out)
            v_list = self.v_engine.propagate(x_spikes, threshold, max_out)
        else:
            # Fallback Python implementation
            q_list = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold, max_out)
            k_list = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold, max_out)
            v_list = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold, max_out)

        q_spikes = set(q_list)
        self.key_buffer.append(set(k_list))
        self.value_buffer.append(set(v_list))
        
        if len(self.key_buffer) > self.context_size:
            self.key_buffer.pop(0)
            self.value_buffer.pop(0)

        # Dynamic routing (Coincidence Detection)
        routed_v = set()
        best_match_idx = -1
        max_coinc = 0
        dyn_thresh = max(1, int(len(q_spikes) * 0.2))
        
        for i, past_k in enumerate(self.key_buffer):
            coinc = len(q_spikes.intersection(past_k))
            if coinc >= dyn_thresh and coinc > max_coinc:
                max_coinc = coinc
                best_match_idx = i
        
        if best_match_idx != -1:
            routed_v = self.value_buffer[best_match_idx]

        if self.use_rust:
            y_spikes = self.o_engine.propagate(list(routed_v), threshold, max_out)
            if learning:
                self.q_engine.apply_stdp(x_spikes, list(q_spikes | set(x_spikes)), 0.05)
                self.k_engine.apply_stdp(x_spikes, k_list, 0.05)
                self.v_engine.apply_stdp(x_spikes, v_list, 0.05)
                self.o_engine.apply_stdp(list(routed_v), y_spikes, 0.05)
                # Sync back to Python for Save/Load
                self.q_weights = self.q_engine.get_weights()
        else:
            y_spikes = self._sparse_propagate(list(routed_v), self.o_weights, self.embed_dim, threshold, max_out)
            if learning:
                self._apply_stdp(x_spikes, q_list, self.q_weights)
                self._apply_stdp(x_spikes, k_list, self.k_weights)
                self._apply_stdp(x_spikes, v_list, self.v_weights)
                self._apply_stdp(list(routed_v), y_spikes, self.o_weights)

        return y_spikes

    def _sparse_propagate(self, active: List[int], weights: List[Dict[int, float]], out_dim: int, threshold: float, max_out: int) -> List[int]:
        potentials = [0.0] * out_dim
        for s in active:
            if s < len(weights):
                for t, w in weights[s].items(): potentials[t] += w
        active_sorted = sorted([(i, p) for i, p in enumerate(potentials) if p > threshold], key=lambda x: x[1], reverse=True)
        return [i for i, p in active_sorted[:max_out]]

    def _apply_stdp(self, pre_spikes: List[int], post_spikes: List[int], weights: List[Dict[int, float]]) -> None:
        # Dummy Python STDP fallback (not fully implemented to match Rust behavior exactly here)
        post_set = set(post_spikes)
        lr = 0.05
        for pre in pre_spikes:
            if pre < len(weights):
                targets = weights[pre]
                to_remove = []
                for target, w in targets.items():
                    if target in post_set:
                        targets[target] = min(3.0, w + lr)
                    else:
                        targets[target] = max(0.0, w - lr * 0.05)
                        if targets[target] < 0.01:
                            to_remove.append(target)
                for t in to_remove:
                    del targets[t]
                for post in post_set:
                    if post not in targets:
                        targets[post] = 0.2


class SpikeFuzzyAttention(SNNModule):
    """
    Biological Attention mechanism using Fuzzy Recall (SDR Overlap).
    Replaces matrix multiplication with Scalable SDR Memory search to handle ambiguity.
    """
    def __init__(self, embed_dim: int, threshold: float = 0.2, top_k: int = 3, use_rust: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.top_k = top_k
        self.use_rust = use_rust and RUST_AVAILABLE
        
        # Core associative memory
        if self.use_rust and hasattr(sara_rust_core, 'ScalableSDRMemory'):
            self.kv_memory = sara_rust_core.ScalableSDRMemory(threshold=self.threshold)
        else:
            self.kv_memory = None
            self.python_memory: List[Tuple[int, Set[int]]] = []
            
        self.values: Dict[int, List[int]] = {}
        self.register_state("values")
        
        self.current_mem_id = 0
        self.register_state("current_mem_id")

    def reset_state(self) -> None:
        """Reset internal short-term memory state."""
        super().reset_state()
        if self.kv_memory:
            self.kv_memory.clear()
        else:
            self.python_memory.clear()
        self.values.clear()
        self.current_mem_id = 0

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        """
        Recall associated values using fuzzy matching on the query spikes.
        """
        out_spikes: Set[int] = set()
        
        # 1. Search (Fuzzy Recall / Biological Association)
        if self.current_mem_id > 0 and x_spikes:
            if self.kv_memory:
                results = self.kv_memory.search(x_spikes, self.top_k)
                for mem_id, score in results:
                    if mem_id in self.values:
                        out_spikes.update(self.values[mem_id])
            else:
                # Python fallback for SDR overlap
                query_set = set(x_spikes)
                query_len = len(query_set)
                if query_len > 0:
                    results_py = []
                    for mem_id, mem_set in self.python_memory:
                        overlap = len(query_set.intersection(mem_set))
                        score = overlap / query_len
                        if score >= self.threshold:
                            results_py.append((mem_id, score))
                    results_py.sort(key=lambda x: x[1], reverse=True)
                    for mem_id, score in results_py[:self.top_k]:
                        if mem_id in self.values:
                            out_spikes.update(self.values[mem_id])
                            
        # 2. Store (Self-Attention Context Accumulation)
        if learning and x_spikes:
            if self.kv_memory:
                self.kv_memory.add_memory(self.current_mem_id, x_spikes)
            else:
                self.python_memory.append((self.current_mem_id, set(x_spikes)))
                
            self.values[self.current_mem_id] = list(x_spikes)
            self.current_mem_id += 1
            
        return list(out_spikes)


class SDRFuzzyAttention(SNNModule):
    """
    Bio-plausible Attention Mechanism alternative to Transformers.
    Uses SDR overlap (Fuzzy Recall) instead of dot-product matrix multiplication.
    Supports multi-lingual processing implicitly through language-agnostic SDRs.
    Transformersの Q, K, V の概念をSNNのスパイクオーバーラップ率で代替するクラスです。
    """
    def __init__(self, sdr_size: int, threshold: float = 0.3):
        super().__init__()
        self.sdr_size = sdr_size
        self.threshold = threshold
        # Memory states for Keys and Values (Episodic Buffer)
        self.keys: List[List[int]] = []
        self.values: List[List[int]] = []
        self.register_state("keys")
        self.register_state("values")

    def forward(self, query: List[int], key: Optional[List[int]] = None, value: Optional[List[int]] = None) -> List[int]:
        """
        Routes spikes based on SDR similarity.
        クエリとなるSDRスパイク列を受け取り、保持しているKeyとのFuzzy RecallによってValueを出力します。
        """
        if key is not None and value is not None:
            self.keys.append(key)
            self.values.append(value)
            
        if not self.keys:
            return query
            
        # Evaluate similarities using Rust core (Fuzzy Recall) if available
        scores = []
        for i, k_sdr in enumerate(self.keys):
            if RUST_AVAILABLE and hasattr(sara_rust_core, 'calculate_sdr_overlap'):
                score = sara_rust_core.calculate_sdr_overlap(query, k_sdr)
            else:
                set_a, set_b = set(query), set(k_sdr)
                if not set_a or not set_b:
                    score = 0.0
                else:
                    score = len(set_a.intersection(set_b)) / float(max(len(set_a), len(set_b)))
            scores.append((i, score))
            
        valid_scores = [item for item in scores if item[1] >= self.threshold]
        
        # Output is a union of values that passed the threshold, modulated by similarity score
        output_spikes = set()
        for i, score in valid_scores:
            # Stochastic routing: higher overlap -> higher probability of spike transmission
            # 確率的ルーティングによって、行列演算を使わずにAttentionの重み付けをシミュレート
            for spike in self.values[i]:
                if random.random() < score:
                    output_spikes.add(spike)
                    
        result = list(output_spikes)
        result.sort()
        return result

    def reset_state(self):
        super().reset_state()
        self.keys.clear()
        self.values.clear()
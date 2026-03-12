# パス: src/sara_engine/models/spiking_jepa.py
# 英語タイトル: Spiking JEPA
# 目的や内容: 誤差逆伝播の代替となる局所エネルギー（Surprise）計算を用いて、スパイクベースでの予測符号化と階層的な表現学習を行う Joint Embedding Predictive Architecture の実装。

import random
from typing import Dict, List, Tuple, Any, Optional

class EnergyMinimizer:
    """
    誤差逆伝播の代替となる局所エネルギー（Surprise）計算。
    """
    def __init__(self, size: int):
        self.size = size

    def _to_indices(self, input_data: Any) -> set:
        if isinstance(input_data, set):
            return input_data
        if isinstance(input_data, list):
            if not input_data:
                return set()
            # If it looks like a mask (contains only 0, 1 and has length > 1)
            if all(v in (0, 1) for v in input_data) and len(input_data) > 1:
                return {i for i, v in enumerate(input_data) if v == 1}
            return set(input_data)
        return set()

    def compute_surprise_signal(self, predictions: Any, targets: Any) -> Tuple[List[int], float]:
        """
        予測と目標を比較し、サプライズスパイクと学習シグナルを生成する。
        """
        pred_set = self._to_indices(predictions)
        targ_set = self._to_indices(targets)
        
        surprise = [0] * self.size
        # Over-prediction and Under-prediction both cause surprise
        for i in pred_set:
            if i < self.size and i not in targ_set:
                surprise[i] = 1
        for i in targ_set:
            if i < self.size and i not in pred_set:
                surprise[i] = 1
        
        if not targ_set:
            # If target is empty, signal remains 0.0 (no standard for comparison)
            return surprise, 0.0
            
        intersect = len(pred_set.intersection(targ_set))
        accuracy = intersect / len(targ_set)
        signal = (accuracy * 2.0) - 1.0 # -1.0 to 1.0
        
        return surprise, signal


class SpikingJEPALayer:
    """
    A single layer of the Spiking Joint Embedding Predictive Architecture.
    Handles encoding of bottom-up signals and prediction of target states.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        learning_rate: float = 0.05,
        w_max: float = 5.0,
        initial_density: float = 0.3
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.w_max = w_max
        self.prune_threshold = 0.01
        
        # Online Encoder: input_dim -> embed_dim
        self.encoder_synapses = self._init_sparse_synapses(input_dim, embed_dim, initial_density)
        # Predictor: embed_dim -> input_dim (Top-down prediction)
        self.predictor_synapses = self._init_sparse_synapses(embed_dim, input_dim, initial_density)
        
        self.potentials = [0.0] * embed_dim
        self._threshold = 1.0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    def _init_sparse_synapses(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        synapses: List[Dict[int, float]] = []
        for _ in range(out_dim):
            connections: Dict[int, float] = {}
            for i in range(in_dim):
                if random.random() < density:
                    connections[i] = random.uniform(0.1, 0.5)
            synapses.append(connections)
        return synapses

    def forward_encoder(self, input_spikes: List[int]) -> List[int]:
        """Bottom-up: Encode input spikes into embedding spikes."""
        out_spikes = [0] * self.embed_dim
        for j in range(self.embed_dim):
            self.potentials[j] *= 0.9  # Leak
            for i in input_spikes:
                if i in self.encoder_synapses[j]:
                    self.potentials[j] += self.encoder_synapses[j][i]
            
            if self.potentials[j] >= self.threshold:
                out_spikes[j] = 1
                self.potentials[j] = 0.0
        return [i for i, s in enumerate(out_spikes) if s == 1]

    def predict_top_down(self, upper_spikes: List[int]) -> List[int]:
        """Top-down: Predict lower layer state from embedding spikes."""
        predicted = [0] * self.input_dim
        # Use a temporary potential for prediction to not interfere with state
        temp_potentials = [0.0] * self.input_dim
        for j in range(self.input_dim):
            # Predict input from hidden/embedding
            for i in upper_spikes:
                if i in self.predictor_synapses[j]:
                    temp_potentials[j] += self.predictor_synapses[j][i]
            
            if temp_potentials[j] >= self.threshold:
                predicted[j] = 1
        return [i for i, s in enumerate(predicted) if s == 1]

    def update_stdp(self, pre_spikes: List[int], post_spikes: List[int], synapses: List[Dict[int, float]], signal: float):
        """Local STDP update based on global/local reinforcement signal."""
        pre_set = set(pre_spikes)
        for j in post_spikes:
            current_synapses = synapses[j]
            for i in list(current_synapses.keys()):
                if i in pre_set:
                    # LTP or LTD based on signal
                    delta = self.learning_rate * signal * (self.w_max - current_synapses[i])
                    current_synapses[i] += delta
                else:
                    # LTD for non-contributing synapses
                    current_synapses[i] -= (self.learning_rate * 0.05 * current_synapses[i])
                
                if current_synapses[i] < self.prune_threshold:
                    del current_synapses[i]
                elif current_synapses[i] > self.w_max:
                    current_synapses[i] = self.w_max

    def reset_state(self):
        self.potentials = [0.0] * self.embed_dim


class SpikingJEPA:
    """
    Hierarchical Joint Embedding Predictive Architecture (H-JEPA).
    Compatibility version for legacy single-layer tests.
    """

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
        for layer in self.layers:
            layer.learning_rate = value

    @property
    def threshold(self) -> float:
        return self.layers[0].threshold if self.layers else 1.0

    @threshold.setter
    def threshold(self, value: float):
        for layer in self.layers:
            layer.threshold = value

    def __init__(
        self,
        arg1: Any = None,
        arg2: Any = None,
        arg3: Any = None,
        layer_configs: Optional[List[Dict[str, int]]] = None,
        ema_decay: float = 0.99,
        learning_rate: float = 0.05,
        time_scales: Optional[Dict[str, int]] = None,
        # Explicit keyword arguments for legacy tests
        context_size: Optional[int] = None,
        target_size: Optional[int] = None,
        hidden_size: Optional[int] = None
    ):
        self.layers: List[SpikingJEPALayer] = []
        self._learning_rate = learning_rate
        
        # Determine if called with old positional args or keyword args
        c_size = context_size or (arg1 if isinstance(arg1, int) else None)
        t_size = target_size or (arg2 if isinstance(arg2, int) else None)
        h_size = hidden_size or (arg3 if isinstance(arg3, int) else None)

        if c_size is not None and t_size is not None and h_size is not None:
            # Legacy Single Layer mode: Context -> Hidden -> Target
            self.layers.append(SpikingJEPALayer(c_size, h_size, learning_rate))
            self.layers[0].predictor_synapses = self.layers[0]._init_sparse_synapses(h_size, t_size, 0.3)
            self.layers[0].input_dim = t_size
            
            self.context_size = c_size
            self.target_size = t_size
            self.hidden_size = h_size
        else:
            # Hierarchical mode
            if layer_configs is None:
                layer_configs = []
            for config in layer_configs:
                # Robustly find input and embed dimensions
                in_dim = config.get("input_dim") or config.get("embed_dim") or config.get("hidden_dim") or 0
                embed_dim = config.get("embed_dim") or config.get("hidden_dim") or 0
                self.layers.append(SpikingJEPALayer(in_dim, embed_dim, learning_rate))
            
            if layer_configs:
                config0 = layer_configs[0]
                confign = layer_configs[-1]
                self.context_size = config0.get("input_dim") or config0.get("embed_dim") or config0.get("hidden_dim") or 0
                self.target_size = confign.get("embed_dim") or confign.get("hidden_dim") or 0
                self.hidden_size = config0.get("embed_dim") or config0.get("hidden_dim") or 0
            else:
                self.context_size = 0
                self.target_size = 0
                self.hidden_size = 0

        self.ema_decay = ema_decay
        self.time_scales = time_scales or {"base": 1}
        self.history: List[List[int]] = []

        # Compatibility attributes for tests
        self.prune_threshold = 0.01
        self.context_projector = self.layers[0].encoder_synapses if self.layers else []
        self.predictor = self.layers[0].predictor_synapses if self.layers else [] # Mapping Hidden -> Target
        
        self.hidden_potentials = [0.0] * self.hidden_size
        self.target_potentials = [0.0] * self.target_size
        self.minimizer = EnergyMinimizer(self.target_size)

    def get_status_message(self, language: str = "en") -> str:
        messages = {
            "ja": "Spiking JEPA: 予測符号化によるエネルギー最小化実行中",
            "en": "Spiking JEPA: Energy minimization via predictive coding in progress",
            "fr": "Spiking JEPA: Minimisation de l'énergie sans rétropropagation"
        }
        return messages.get(language, messages["en"])

    def step(self, context_spikes: Any, target_spikes: Any, learning: bool = True) -> Tuple[List[int], float]:
        """Backward compatibility step function."""
        # Convert list input to spike indices if needed
        c_spiking_indices = self.minimizer._to_indices(context_spikes)
        t_spiking_indices = self.minimizer._to_indices(target_spikes)

        # Single layer logic for legacy tests
        hidden_spikes = self.layers[0].forward_encoder(list(c_spiking_indices))
        pred_target_spikes = self.layers[0].predict_top_down(hidden_spikes)
        
        surprise, signal = self.minimizer.compute_surprise_signal(pred_target_spikes, list(t_spiking_indices))
        
        if learning:
            # Update weights: context -> hidden (signal by surprise)
            self.layers[0].update_stdp(list(c_spiking_indices), hidden_spikes, self.layers[0].encoder_synapses, signal)
            # Update predictor: hidden -> target
            self.layers[0].update_stdp(hidden_spikes, list(t_spiking_indices), self.layers[0].predictor_synapses, signal)
            
        return surprise, signal

    def _update_weights_stdp(self, pre: Any, post: Any, synapses: List[Dict[int, float]], signal: float):
        """Internal helper for test_pruning_mechanism"""
        pre_idx = list(self.minimizer._to_indices(pre))
        post_idx = list(self.minimizer._to_indices(post))
        self.layers[0].update_stdp(pre_idx, post_idx, synapses, signal)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
        self.history = []

    def state_dict(self) -> Dict[str, Any]:
        """Returns serializable state for saving."""
        state = []
        for layer in self.layers:
            state.append({
                "encoder": layer.encoder_synapses,
                "predictor": layer.predictor_synapses
            })
        return {"layers": state}

    def load_state_dict(self, state: Dict[str, Any]):
        """Loads state from dictionary."""
        layers_data = state.get("layers", [])
        for i, data in enumerate(layers_data):
            if i < len(self.layers):
                self.layers[i].encoder_synapses = data["encoder"]
                self.layers[i].predictor_synapses = data["predictor"]

    def forward(
        self,
        x_spikes: List[int],
        y_spikes: Optional[List[int]] = None,
        learning: bool = False
    ) -> Tuple[List[int], float]:
        """
        Hierarchical forward pass for H-JEPA.
        """
        # (Rest of H-JEPA logic remains same but we use step for legacy)
        encodings: List[List[int]] = [x_spikes]
        for layer in self.layers:
            next_spikes = layer.forward_encoder(encodings[-1])
            encodings.append(next_spikes)
        
        predictions: List[List[int]] = [[] for _ in range(len(self.layers) + 1)]
        for i in reversed(range(len(self.layers))):
            predictions[i] = self.layers[i].predict_top_down(encodings[i+1])
            
        top_prediction = predictions[0]

        surprise_signal = 0.0
        if y_spikes is not None and learning:
            _, surprise_signal = self.minimizer.compute_surprise_signal(top_prediction, y_spikes)
            
            current_input = x_spikes
            for i, layer in enumerate(self.layers):
                layer.update_stdp(current_input, encodings[i+1], layer.encoder_synapses, surprise_signal)
                layer.update_stdp(encodings[i+1], current_input, layer.predictor_synapses, surprise_signal)
                current_input = encodings[i+1]

        return top_prediction, surprise_signal
### ROADMAP  

### Policy  
・Create as a PyPI library  
・Aim for energy efficiency  
・Do not use backpropagation  
・Do not use matrix operations  
・Do not use GPU  
・Prioritize biologically-inspired design  

### Work Policy  
・Prioritize creating SNN versions of features that exist in Transformers  
    
### 1. Neuromorphic Debugging & Visualization Tools

In SNNs, "why a particular decision was made" manifests as a chain of spikes, making visualization crucial to avoid black-box behavior.

* **Real-time Spike Raster Plot Generation**:
A feature to display the firing timing of neurons across all layers on a time axis, visualizing the "Echo of information" within the reservoir.
* **Attention Heatmap**:
A tool to dynamically visualize which past spike patterns the `SpikeAttention` strongly responds to (Overlap).
* **Membrane Potential Distribution Statistical Analysis**:
A feature to display histograms showing how the membrane potential of neurons in each layer is distributed relative to the threshold, diagnosing non-firing or excessive firing (Spike Storm).

### 2. Enhancement of Biological Learning Rules (Algorithm Deepening)

Currently, supervised learning in the output layer is primary, but strengthening "self-organization" in hidden layers will improve feature extraction capability from unlabeled data.

* **Formal Integration of STDP (Spike-Timing-Dependent Plasticity)**:
Implement STDP from the roadmap as a standard layer, enabling automatic learning of "frequent patterns" from input data alone.
* **Intrinsic Plasticity**:
A homeostasis feature that automatically optimizes network-wide activity efficiency by lowering the threshold of neurons with too-low firing rates and raising the threshold of neurons with too-high firing rates.
* **Automatic Assignment of Heterogeneous Time-constants**:
A feature to automatically distribute "individual differences in time constants," whose effectiveness has been confirmed in validation results, according to the role of each layer.

### 3. Edge & Multimodal Support

Features to leverage the strengths of GPU-free and low power consumption.

* **Event-driven Data Loader**:
Standard utilities for directly converting time-series data such as DVS (Dynamic Vision Sensor) or audio waveforms into spike trains (Rate/Temporal Coding).
* **Quantization & Integer Arithmetic Mode**:
An option to perform membrane potential calculations using fixed-point (int8/int16) instead of floating-point (f32). This facilitates porting to more affordable microcomputers (ESP32, ARM Cortex-M series).
* **Multimodal Reservoir**:
An architecture that mixes and integrates visual SDR and audio SDR within a single reservoir, enabling cross-modal associations (such as recalling images from hearing sounds).

### 4. Developer Experience (DX) Improvement

Interface improvements to attract users familiar with PyTorch and TensorFlow.

* **Scikit-learn / PyTorch-like API**:
Unification to standard method names such as `model.fit(X, y)` and `model.predict(X)`.
* **JSON/YAML Export of Model Weights and Structure**:
Export functionality in formats that are easy to load from other languages (standalone C or Rust), not just `pickle`.
* **Pre-trained Model Zoo**:
Provision of trained reservoir weights specialized for specific domains (Japanese text, waveform analysis, sensor anomaly detection).

### Recommended Priority for Implementation

First, implementation of **"1. Visualization Tools"** is recommended. By visualizing SNN behavior, users (developers) can easily understand "where information is being lost in which layer" and "where abnormal firing is occurring," which will increase the library's adoption rate.
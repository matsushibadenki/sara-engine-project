# SARA Engine (Liquid Harmony)

**SARA (Spiking Advanced Recursive Architecture)** is a next-generation AI engine (SNN-based) that mimics the biological brain's "power efficiency, event-driven processing, and self-organization."

It completely eliminates the "backpropagation (BP)" and "matrix operations" that modern deep learning (ANNs) rely on, achieving advanced recognition and learning capabilities using **only sparse spike communication**.

It operates on CPU only, without using any GPU.

Current Version: **v0.1.4**

## Features

* **No Backpropagation**: Learns without error backpropagation, using local learning rules (Momentum Delta) and reservoir computing.
* **CPU Only & Lightweight**: Does not require expensive GPU resources. Runs fast on standard CPU environments.
* **Multi-Scale True Liquid Reservoir**: Three parallel reservoir layers with different temporal characteristics (Decay), with recurrent connections within each layer. Achieves short-term memory using information "echo."
* **Rust Acceleration**: Core computation logic is written in Rust for high performance.

## Installation  
  
```bash
pip install sara-engine
```  
Quick Start  
  
```bash
from sara_engine import SaraGPT

# Initialize the brain
brain = SaraGPT(sdr_size=1024)

# Create an input pattern (SDR)
input_sdr = brain.encoder.encode("Hello SARA")

# Think (Forward pass)
output_sdr, spikes = brain.forward_step(input_sdr)

print(f"Output Active Neurons: {len(output_sdr)}")
```
  
License  
MIT License  
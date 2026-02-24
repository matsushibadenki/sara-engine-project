# SARA Engine (Liquid Harmony)

**SARA (Spiking Advanced Recursive Architecture)** is a next-generation AI engine that mimics the biological brain's power efficiency and event-driven processing. 

It completely eliminates the "backpropagation (BP)" and "matrix operations" that modern deep learning (ANNs) rely on, achieving advanced recognition, text generation, and retrieval capabilities using **only sparse spike communication**.

It operates completely on CPU, without using any GPU or NumPy.

Current Version: **v0.2.1**

## Features

* **Hugging Face-like Pipelines**: Incredibly easy-to-use API (`pipeline("text-generation")`, `pipeline("image-classification")`, etc.).
* **No Backpropagation**: Learns natively using biological rules like Spike-Timing-Dependent Plasticity (STDP) and Homeostatic Plasticity.
* **Zero Matrix Math**: Replaces dense tensor multiplications with purely discrete spike routing.
* **CPU Only & Eco-Friendly**: Does not require expensive GPU resources.

## Installation  
  
```bash
pip install sara-engine

```

## Quick Start

SARA v0.2.1 introduces an incredibly intuitive pipeline API, bringing the ease of modern NLP frameworks to Spiking Neural Networks.

### 1. Text Generation (Zero-Shot Inference)

You can easily load pre-trained STDP synapses and generate text autoregressively.

```python
from sara_engine import pipeline

# Load a biologically trained model and tokenizer
generator = pipeline("text-generation", model="path/to/saved_snn_model")

output = generator("Hello, I am a spiking", max_new_tokens=15)
print(output[0]['generated_text'])

```

### 2. Feature Extraction & RAG

Extract semantic embeddings using Liquid State Machines (LSM) without any mathematical attention layers.

```python
from sara_engine import pipeline

extractor = pipeline("feature-extraction", model="path/to/saved_extractor")

# Extracts a high-dimensional membrane potential vector
vector = extractor("Artificial intelligence is evolving.")
print(f"Spike Vector Length: {len(vector)}")

```

### 3. Image Classification

Process pixels via Retinal Rate Coding directly into SNN layers.

```python
from sara_engine import pipeline

vision_classifier = pipeline("image-classification", model="path/to/vision_model")

# Pass a 2D array of pixel intensities (0.0 to 1.0)
image = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0]
]
prediction = vision_classifier(image)
print(prediction[0]['label']) # e.g., "Cross (X)"

```

## License

MIT License

```
# **About Tools**

This document describes the tools and commands used for testing and verifying the SARA Engine project.

## **Test Commands**

The following commands are used to verify the functionality of the SARA Engine. Ensure you are in the project root directory before running these commands.

### **1\. Unit Tests (pytest)**

Run the full test suite or specific modules using pytest.

| Description | Command |
| :---- | :---- |
| **Run All Tests** | pytest |
| **NeuroFEM Tests** (Core & Integration) | pytest tests/test\_neurofem.py tests/test\_neurofem\_2d.py tests/test\_neurofem\_integration.py tests/test\_neurofem\_visualize.py |
| **SNN & STDP Tests** | pytest tests/test\_spatiotemporal\_stdp.py tests/test\_event\_driven\_snn.py |
| **Memory & Hippocampus Tests** | pytest tests/test\_hippocampal\_system.py tests/test\_crossmodal\_association.py |
| **New Features Tests** | pytest tests/test\_new\_features.py |
| **Million Token SNN** | pytest tests/test\_million\_token\_snn.py |

### **2\. Demos & Examples**

Run these Python scripts to see the engine in action. All scripts are located in the examples/ directory.

| Description | Command |
| :---- | :---- |
| **MNIST SNN Demo** | python examples/demo\_mnist\_snn.py |
| **Fashion-MNIST SNN** | python examples/demo\_advanced\_snn.py |
| **RL Training Demo** | python examples/demo\_rl\_training.py |
| **SNN Learning** | python examples/demo\_snn\_learning.py |
| **Rust SNN (No Numpy)** | python examples/demo\_rust\_snn\_no\_numpy.py |
| **Benchmark Rust** | python examples/benchmark\_rust.py |
| **SNN LLM** | python examples/demo\_snn\_llm.py |
| **SNN Text Generation** | python examples/demo\_snn\_text\_generation.py |
| **Spiking LLM** | python examples/demo\_spiking\_llm.py |
| **Spiking LLM Save/Load** | python examples/demo\_spiking\_llm\_save\_load.py |
| **Spiking LLM Text** | python examples/demo\_spiking\_llm\_text.py |
| **Spiking Transformer** | python examples/demo\_spiking\_transformer.py |
| **Bio Transformer** (STDP Self-Attention) | python examples/demo\_bio\_transformer.py |
| **Agent Chat** | python examples/demo\_agent\_chat.py |
| **Million Token Agent** | python examples/demo\_million\_token\_agent.py |
| **Crossmodal Recall** | python examples/demo\_crossmodal\_recall.py |
| **Multimodal Memory** | python examples/demo\_multimodal\_memory.py |
| **Knowledge Recall** | python examples/test\_knowledge\_recall.py |
| **Interactive Demo** | python examples/interactive\_demo.py |
| **Interactive SNN** | python examples/interactive\_snn.py |
| **Visualize STDP** | python examples/visualize\_stdp.py |
| **SNN transformer** | python examples/demo\_snn\_transformer.py |
| **Stream Learning** | python examples/demo\_stream\_learning.py |

  
*(Note: examples/utils.py is a utility module for the examples and is not meant to be run directly as a standalone demo.)*

### **3\. Utilities & Diagnostics**

| Description | Command |
| :---- | :---- |
| **System Health Check** | python scripts/health\_check.py |

## **Environment Setup**

Ensure your environment is set up correctly before running tests:

1. **Install Dependencies:** pip install \-e . (or use requirements if specified)  
2. **Rust Compilation (if using Rust core):** Ensure the sara\_rust\_core library is compiled and available in src/sara\_engine/.
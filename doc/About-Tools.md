# **About Tools**

This document describes the tools and commands used for testing and verifying the SARA Engine project.

## **Test Commands**

The following commands are used to verify the functionality of the SARA Engine. Ensure you are in the project root directory before running these commands.

### **0\. health_check**  
python scripts/health_check.py  
  
### **1\. Unit Tests (pytest)**

Run the full test suite or specific modules using pytest.

| Description | Command |
| :---- | :---- |
| **Run All Tests** | pytest |
| **NeuroFEM Tests** (Core & Integration) | pytest tests/test\_neurofem.py tests/test\_neurofem\_2d.py tests/test\_neurofem\_integration.py |
| **SNN & STDP Tests** | pytest tests/test\_spatiotemporal\_stdp.py tests/test\_event\_driven\_snn.py |
| **Memory & Hippocampus Tests** | pytest tests/test\_hippocampal\_system.py tests/test\_crossmodal\_association.py |
| **New Features Tests** | pytest tests/test\_new\_features.py |
| **Visualization Tests** | pytest tests/test\_neurofem\_visualize.py |

### **2\. Demos & Examples**

Run these Python scripts to see the engine in action.

| Description | Command |
| :---- | :---- |
| **MNIST SNN Demo** (Spiking Neural Network on MNIST) | python examples/demo\_mnist\_snn.py |
| **RL Training Demo** (Reinforcement Learning) | python examples/demo\_rl\_training.py |
| **Interactive SNN** (Real-time interaction) | python examples/interactive\_snn.py |
| **Agent Chat Demo** (SARA Agent conversation) | python examples/demo\_agent\_chat.py |
| **Multimodal Memory** (Vision & Audio association) | python examples/demo\_multimodal\_memory.py |
| **STDP Visualization** (Visualizing weight updates) | python examples/visualize\_stdp.py |
| **Rust Benchmark** (Performance comparison) | python examples/benchmark\_rust.py |

### **3\. Utilities & Diagnostics**

| Description | Command |
| :---- | :---- |
| **System Health Check** | python scripts/health\_check.py |

## **Environment Setup**

Ensure your environment is set up correctly before running tests:

1. **Install Dependencies:**  
   pip install \-r requirements.txt

2. **Build Rust Extension (if modified):**  
   maturin develop \--release

3. **Check Python Path:**  
   Ensure the project root is in your PYTHONPATH if you encounter import errors (though the package structure usually handles this).  
   export PYTHONPATH=$PYTHONPATH:.  

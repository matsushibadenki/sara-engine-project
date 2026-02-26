# **Details of Tools and Sample Scripts**

This document provides an overview of various sample scripts, benchmarks, and test tools included in the SARA Engine project.

## **1\. Examples (examples/)**

This directory contains demo scripts illustrating various engine features and benchmarks for performance measurement (total 59 files).

### **Demonstrations (Demos)**

Showcases the primary components and use cases of SARA.

* **Agent & Chat**  
  * demo\_agent\_chat.py: Conversational demo with autonomous agents.  
  * demo\_interactive\_chat.py: Interactive chat interface.  
  * demo\_million\_token\_agent.py: Demo of an agent handling massive contexts.  
* **Spiking Neural Networks (SNN)**  
  * demo\_advanced\_snn.py: Demo of advanced SNN configurations.  
  * demo\_mnist\_snn.py: MNIST digit recognition using SNN.  
  * demo\_snn\_classification.py: Demo for general classification tasks.  
  * demo\_snn\_learning.py: Demo of online learning capabilities.  
  * demo\_snn\_feature\_extraction.py: Feature extraction pipeline.  
* **Transformer & LLM**  
  * demo\_bio\_transformer.py: Demo of biologically plausible Transformers.  
  * demo\_snn\_transformer.py: SNN-based Transformer implementation.  
  * demo\_snn\_transformer\_multipath.py: Multi-path configuration Transformer.  
  * demo\_spiking\_llm.py: Spike-based language model demo.  
  * demo\_spiking\_llm\_text.py: Spiking LLM specialized for text generation.  
  * demo\_spiking\_llm\_save\_load.py: Example of saving and loading models.  
* **Multimodal & Pipelines**  
  * demo\_multimodal\_memory.py: Memory system across multiple modalities.  
  * demo\_multimodal\_pipeline.py: Building multimodal processing pipelines.  
  * demo\_snn\_pipelines.py: Collective demo of inference pipelines.  
* **Specific Tasks & Applications**  
  * demo\_snn\_audio\_classification.py: Audio data recognition.  
  * demo\_snn\_image\_classification.py: Image data recognition.  
  * demo\_snn\_text\_classification.py: Text classification.  
  * demo\_snn\_text\_generation.py: Spike-based text generation.  
  * demo\_snn\_token\_classification.py: Token-level classification.  
  * demo\_snn\_rag.py: Implementation example of RAG (Retrieval-Augmented Generation).  
  * demo\_snn\_rag\_persistent.py: RAG using persistent storage.  
  * demo\_rl\_training.py: Reinforcement learning training loop.  
  * demo\_predictive\_coding.py: Implementation of Predictive Coding.  
  * demo\_predictive\_lm.py: Predictive language model.  
  * demo\_semantic\_spike\_routing.py: Semantic spike routing.  
  * demo\_spike\_attention.py: Spike attention mechanism.  
  * demo\_spike\_dataloader.py: Specialized loader for spike data.  
  * demo\_spike\_stream\_processing.py: Stream data processing.  
  * demo\_stream\_learning.py: Real-time stream learning.  
* **Hardware, Edge & Rust**  
  * demo\_sara\_board.py: Demo of the SARA Board interface.  
  * demo\_sara\_edge.py: Deployment example for edge devices.  
  * demo\_saraboard\_and\_loader.py: Collaboration between board and data loader.  
  * demo\_rust\_snn\_no\_numpy.py: SNN using Rust core independent of NumPy.  
  * demo\_nn\_module.py: Demo of the new NN module configuration.

### **Benchmarks (Benchmarks)**

Used for measuring performance and accuracy.

* benchmark\_hal.py: Efficiency measurement of the Hardware Abstraction Layer (HAL).  
* benchmark\_long\_context.py: Evaluation of memory and speed during long-text processing.  
* benchmark\_memory\_retention.py: Evaluation of memory retention (forgetting resistance).  
* benchmark\_multicore.py: Scaling performance of multi-core parallel processing.  
* benchmark\_multimodal\_associative.py: Performance of multimodal associative memory.  
* benchmark\_rl\_stdp.py: Convergence of Reinforcement Learning (RL) using STDP.  
* benchmark\_rust.py: Speed comparison between Python and Rust core implementations.  
* benchmark\_rust\_acceleration.py: Detailed measurement of acceleration effects by Rust.  
* benchmark\_snn\_transformer.py: Computational cost evaluation of SNN Transformers.

### **Interactive & Utilities**

* interactive\_demo.py: Comprehensive interactive GUI demo.  
* interactive\_snn.py: Tool to check SNN behavior in real-time.  
* visualize\_stdp.py: Visualization of STDP (Spike-Timing-Dependent Plasticity).  
* utils.py: Common utility functions for sample scripts.  
* test\_knowledge\_recall.py: Knowledge recall accuracy evaluation (in-sample test).  
* test\_spike\_dataloader.py: Functional verification of the data loader (in-sample test).  
* test\_transformer\_components.py: Operational check of Transformer components.

## **2\. Tests (tests/)**

Unit and integration tests to ensure system reliability (total 10 files).

* test\_crossmodal\_association.py: Test for cross-modal association capabilities.  
* test\_event\_driven\_snn.py: Operational verification of the event-driven SNN engine.  
* test\_hippocampal\_system.py: Test for hippocampal-like short-term/long-term memory systems.  
* test\_million\_token\_snn.py: Stability test for massive context processing.  
* test\_neurofem.py: Basic calculation test for NeuroFEM (Neuro-Finite Element Method).  
* test\_neurofem\_2d.py: NeuroFEM simulation in 2D space.  
* test\_neurofem\_integration.py: Integrated operational test of NeuroFEM and SNN.  
* test\_neurofem\_visualize.py: Visualization test of NeuroFEM calculation results.  
* test\_new\_features.py: Comprehensive verification of newly added features.  
* test\_spatiotemporal\_stdp.py: Operational verification of spatiotemporal STDP rules.

## **3\. Scripts (scripts/)**

Scripts for maintenance and operations.

* health\_check.py: Diagnostic tool to check installation environment, dependencies, and Rust core connectivity.
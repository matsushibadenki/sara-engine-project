# **About SARA Engine Tools**

The SARA Engine project includes a diverse set of scripts designed to facilitate development, experimentation, and maintenance, ranging from model training and evaluation to data collection and database management.

All scripts are intended to be executed from the project root directory.

## **1\. Integrated CLI Tool**

* **scripts/sara\_cli.py**  
  The main entry point for unifying and invoking core SARA Engine functionalities (such as chatting, training, and evaluation) from the command line. It acts as a wrapper for individual scripts.

## **2\. Training Scripts (scripts/train/)**

Scripts for training SNN models (pre-training, fine-tuning, and self-organization). **Note: Following SARA's core policy, these scripts rely on local learning rules like STDP and do not use backpropagation.**

* **train\_snn\_lm.py**  
  Executes pre-training of the Spiking Language Model (SNN-LM). It learns word co-occurrence patterns and contexts as spike timings from the corpus.  
* **train\_chat.py**  
  Fine-tunes (or directly wires) the SNN model for conversational tasks using chat datasets (e.g., chat\_data.jsonl).  
* **train\_vision.py**  
  Trains visual cortex spiking modules using image datasets like MNIST or Fashion MNIST.  
* **train\_self\_organized.py**  
  Runs experiments on unsupervised/self-organized learning utilizing structural plasticity (dynamic creation/deletion of synapses) and homeostasis.  
* **distill\_llm.py**  
  Uses the Biological Distillation module to efficiently extract and transfer knowledge from large existing models into spiking representations.  
* **optimize\_hyperparams.py**  
  Automatically searches and optimizes SNN-specific hyperparameters, such as firing thresholds, leak rates, and STDP learning rates.

## **3\. Evaluation & Inference Scripts (scripts/eval/)**

Scripts for benchmarking pre-trained models and interacting with them via terminal interfaces.

* **chat\_agent.py**  
  Launches the interactive chat interface with the fully integrated sara\_agent, which possesses multimodal processing capabilities and memory access.  
* **chat\_snn\_lm.py**  
  Initiates an interactive text chat directly with the pre-trained pure SNN Language Model.  
* **chat\_self\_organized.py**  
  Tests conversational agents built purely on self-organizing mechanisms.  
* **test\_math\_chat.py**  
  Evaluates the SNN model's response accuracy and logical reasoning capabilities on mathematical reasoning tasks and prompts.  
* **test\_vision\_inference.py**  
  Tests and evaluates the inference accuracy of visual models (e.g., spiking image classifiers).  
* **health\_check.py**  
  Performs a system health check, verifying module dependencies, environmental variables, and the loading status of the Rust core (sara\_rust\_core).

## **4\. Data Collection & Preprocessing Scripts (scripts/data/)**

Scripts for gathering, parsing, and formatting corpora and datasets used for training.

* **collect\_all.py**  
  Collects and integrates text data from all configured data sources in a single batch.  
* **collect\_aozora.py**  
  Collects Japanese texts from open-source repositories like Aozora Bunko and converts them into an SNN-processable corpus format.  
* **collect\_docs.py**  
  Collects project documentation and external technical papers, formatting them as a knowledge base (e.g., for SNN-RAG).  
* **collect\_math.py**  
  Gathers and parses mathematical problems and formula datasets to enhance logical reasoning capabilities.
* **collect\_dvs.py**  
  Converts event-based DVS datasets (CSV/NPZ/AEDAT) into spike-train JSONL files with optional spatial and temporal downsampling.

## **5\. Utilities & Database Management (scripts/utils/)**

Scripts for maintaining Long-Term Memory (LTM) vector stores, SQLite databases, and optimizing network structures.

* **manage\_db.py**  
  Manages the initialization, updates, and migrations of the SARA Engine corpus database (data/sara\_corpus.db).  
* **fix\_memory.py**  
  Repairs corrupted entries or inconsistencies within the Long-Term Memory (LTM) and SNN vector stores.  
* **prune\_memory.py**  
  Prunes unnecessary synapses with extremely low firing rates and trims obsolete memory vectors to improve inference speed and memory efficiency.

# **SARA Engine: Data Pipeline & Training Manual**

This manual provides a comprehensive guide on how to collect and manage data materials, train the SARA Engine using different learning methods, and test the trained models. All operations are smoothly managed through the centralized sara\_cli.py command-line interface.

## **1\. Data Collection and Management**

The SARA Engine utilizes a centralized SQLite database to manage all training materials. This "Data Hub" architecture allows you to throw any text or conversational data into the database, which will then automatically format and export it for the specific training method you choose.

### **Step 1.1: Reset the Database (Optional)**

If you are starting fresh or want to clear old data from previous experiments, initialize the database.

python scripts/sara\_cli.py db-reset

### **Step 1.2: Prepare and Import Data Materials**

You can import plain text files (.txt) for general knowledge acquisition and JSONL files (.jsonl) for conversational fine-tuning. The system automatically ignores exact duplicates.

* **For Document Data (e.g., Wikipedia articles, novels, manuals):**  
  Create a .txt file where text is written naturally. (If you want to feed it a novel, just save the text as novel.txt).  
  python scripts/sara\_cli.py db-import path/to/your/document.txt

* **For Conversational Data:**  
  Create a .jsonl file containing prompt and response pairs.  
  *Example format inside the file:* {"prompt": "Hello", "response": "Hi there\!"}  
  python scripts/sara\_cli.py db-import path/to/your/chat\_data.jsonl

### **Step 1.3: Check Database Status**

To verify how many documents and chat interactions are currently stored in your Data Hub:

python scripts/sara\_cli.py db-status

### **Step 1.4: Export Data for Training**

Before starting the training process, you must export the database contents. The system will automatically generate the required formats for both training methods.

python scripts/sara\_cli.py db-export

*This command generates:*

* data/processed/corpus.txt (A continuous text file optimized for Self-Organized SNN Learning)  
* data/raw/chat\_data.jsonl (A structured file for Distillation Learning)

## **2\. Training the Model**

The SARA Engine supports two distinct training algorithms. You can easily switch between them depending on your research goals.

### **Method A: Self-Organized Learning (Recommended / Pure SNN)**

This is the core Spiking Neural Network (SNN) approach. It **does not use backpropagation**. It builds a long-term memory (Direct Wiring) and short-term episodic memory (STDP) directly from reading the text.

*(Note: Ensure you have the morphological analyzer installed for optimal Japanese word tokenization: pip install janome)*

python scripts/sara\_cli.py train-self-org

* **What happens:** The model reads corpus.txt, builds synaptic connections (N-gram delays) at high speed using the Rust core, and learns sequences via STDP.  
* **Output:** The memory is saved to models/self\_organized\_llm/spiking\_llm\_weights.json.

### **Method B: Distillation Learning (Legacy Chat Fine-Tuning)**

This legacy path fine-tunes the `SaraAgent` conversational memory using structured conversational data.

python scripts/sara\_cli.py train-distill

* **What happens:** The agent trains on the prompt/response pairs in `data/raw/chat_data.jsonl`.  
* **Output:** The trained agent state is saved under `models/sara_agent` by default. You can override the save directory with `--model`.

### **Method C: Subword-Level SNN Pre-training (サブワードレベルSNN事前学習)**

新しく追加されたSpiking Transformer Modelを用いた、より高度な言語獲得アプローチです。単語単位ではなく、BPE（Byte-Pair Encoding）を用いたサブワード単位で学習を行います。これにより、未知語への対応力や表現の柔軟性が向上します。

`python scripts/train/train_snn_lm.py --corpus data/corpus.txt`

オプションとして、`--chat-data data/raw/chat_data.jsonl` を指定することで、コーパスに加えて対話形式のデータも同時に学習させることができます。

* **What happens:** 指定されたコーパス（例えば `data/corpus.txt`）を読み込み、不要なノイズを除去したうえで `SaraTokenizer` を学習させます。その後、トランスフォーマーアーキテクチャを持つSNNモデル（`SpikingTransformerModel`）にテキストのシーケンスを学習させます。
* **Output:** 学習済みの重みと語彙データ（`sara_vocab.json`）は、デフォルトで `models/snn_lm_pretrained` に保存されます。保存先は `--save-dir` で変更可能です。


## **3\. Testing and Inference**

Once the model is trained, you can interact with it using the built-in chat interface. **Make sure to use the chat command that matches your training method.**

### **Testing the Self-Organized Model**

If you trained using train-self-org, start the interactive chat with:

python scripts/sara\_cli.py chat-self-org

* Type your prompt at the User \> prompt.  
* The SNN will generate a response based on Coincidence Detection, fuzzy recall, and neuron fatigue mechanisms.  
* The model features an "Awareness of Ignorance" safety mechanism: if you ask about a topic it hasn't learned, it will safely decline to answer rather than hallucinating.  
* Type quit or exit to stop the session.

### **Testing the Distilled / Legacy Model**

If you trained using `train-distill`, start the matching interactive agent chat with:

python scripts/sara\_cli.py chat-distill

By default this loads `models/sara_agent`. You can override the model directory with `--model`.

### **Testing the Subword-Level SNN Model (サブワードモデルのテスト)**

Method C (`train_snn_lm.py`) で学習したサブワードレベルのモデルをテストするための推論・対話スクリプトです。

`python scripts/eval/chat_snn_lm.py`

* デフォルトで `models/snn_lm_pretrained` から学習済みの重みと語彙を読み込みます。別のディレクトリを指定する場合は `--model-dir` を使用してください。
* `--debug` フラグを付けることで、各推論ステップごとのトークン候補や内部の発火ポテンシャルを可視化でき、学習結果の分析に役立ちます。（例: `python scripts/eval/chat_snn_lm.py --debug`）
* **機能:** 入力されたプロンプトに対して、SNNのニューロンの活動（スパイク）をシミュレートしながら次のトークンを予測・生成します。また、RAG（Retrieval-Augmented Generation）のような仕組みを持ち、学習データに基づいたローカルナレッジも活用して応答します。


## **4\. Utility Commands**

### **Pruning Memory (Synaptic Pruning)**

To reduce the model size and improve efficiency by removing weak, unused synaptic connections (forgetting):

python scripts/sara\_cli.py prune \--threshold 50.0

### **Cleaning the Environment**

To remove all interim data, processed files, and reset the workspace for a clean slate:

python scripts/sara\_cli.py clean

This command currently cleans `data/interim` and `data/processed`. There is no `--all` option in the current CLI implementation.

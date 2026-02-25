_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/text_generation.py",
    "//": "ファイルの日本語タイトル: テキスト生成パイプライン",
    "//": "ファイルの目的や内容: Transformersのpipeline('text-generation')をSNNで再現。トークナイザーを利用したエンコード・デコード処理とモデルとの連携を実装。"
}

from typing import Any

class TextGenerationPipeline:
    """Pipeline for text generation using a causal Spiking Neural Network."""
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str, max_length: int = 50, **kwargs) -> str:
        """
        Generates text based on the provided prompt string.
        No backpropagation or matrix multiplications are used during this phase.
        """
        # トークナイザーで入力テキストをIDのリストに変換
        input_ids = self.tokenizer.encode(text)
        
        # モデルでIDを生成
        output_ids = self.model.generate(input_ids, max_length=max_length)
        
        # IDのリストをテキストにデコードし、プロンプトと結合して返す
        generated_text = self.tokenizer.decode(output_ids)
        
        # 連続するスペースや不自然な空白の調整 (必要に応じて)
        return text + generated_text

    def learn(self, text: str) -> None:
        """
        Trains the SNN locally on the provided sequence using STDP.
        This allows continuous, energy-efficient learning on the edge.
        """
        # トークナイザーで入力テキストをIDのリストに変換して学習させる
        input_ids = self.tokenizer.encode(text)
        self.model.learn_sequence(input_ids)

    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN configuration and synaptic weights to disk."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save() # トークナイザーの語彙情報も一緒に保存することが望ましい

def pipeline(task: str, model: Any = None, tokenizer: Any = None) -> Any:
    """
    Utility factory method to build a pipeline for SNN tasks.
    Modeled after the Hugging Face Transformers pipeline() function.
    """
    if task == "text-generation":
        if model is None or tokenizer is None:
            raise ValueError("Both 'model' and 'tokenizer' must be provided for the pipeline.")
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)
    else:
        raise NotImplementedError(f"The task '{task}' is not yet supported in the SARA engine.")
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/base.py",
    "//": "ファイルの日本語タイトル: パイプライン基底クラス",
    "//": "ファイルの目的や内容: すべてのSNNパイプラインのベースとなるクラスを定義する。汎用的でTransformersに似た設計。"
}

class Pipeline:
    """
    Base class for all Spiking Neural Network pipelines.
    Provides a unified interface similar to Hugging Face's pipeline,
    but specialized for neuromorphic and bio-inspired SNN architectures.
    """
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Each pipeline must implement the __call__ method.")
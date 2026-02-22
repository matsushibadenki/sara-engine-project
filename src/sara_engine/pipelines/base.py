_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/base.py",
    "//": "ファイルの日本語タイトル: パイプライン基底クラス",
    "//": "ファイルの目的や内容: TransformersのPipelineベースクラスに相当する、SNNパイプラインの共通基盤。"
}

class Pipeline:
    """
    Base class for all SARA Engine pipelines.
    """
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = kwargs.get("device", "cpu") # Always CPU/Edge for SARA
        
    def save_pretrained(self, save_directory: str) -> None:
        """Saves the SNN model state."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)
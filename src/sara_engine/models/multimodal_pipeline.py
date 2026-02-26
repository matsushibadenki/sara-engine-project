_FILE_INFO = {
    "path": "src/sara_engine/models/multimodal_pipeline.py",
    "title": "マルチモーダル・スパイキング・パイプライン",
    "purpose": "テキスト、画像、音声のスパイクを統合し、CrossModalAssociatorを用いて相互の連想（グラウンディング）を行う。ROADMAP Phase 2のマルチモーダル処理の確立。",
}

from typing import List, Dict, Optional
from sara_engine import nn
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

class MultimodalSNNConfig(SNNTransformerConfig):
    def __init__(self, vision_dim: int = 256, audio_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "vision_dim": self.vision_dim,
            "audio_dim": self.audio_dim
        })
        return d
        
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class SpikingMultimodalPipeline(nn.SNNModule):
    """
    テキスト(LLMモジュール)、視覚、聴覚のスパイクを統合的に処理し、
    概念のグラウンディング（言葉と感覚の結びつけ）を行うモジュール。
    """
    def __init__(self, config: MultimodalSNNConfig):
        super().__init__()
        self.config = config
        
        # 1. テキスト処理のバックボーン (SNN Transformer)
        self.text_model = SpikingTransformerModel(config)
        
        # 2. 視覚と聴覚のエンコーダ (入力を共通の潜在空間次元にマッピング)
        self.vision_encoder = nn.Sequential(
            nn.LinearSpike(in_features=config.vision_dim, out_features=config.embed_dim, density=0.2),
            nn.SpikeLayerNorm(target_spikes=max(1, int(config.embed_dim * 0.25)))
        )
        
        self.audio_encoder = nn.Sequential(
            nn.LinearSpike(in_features=config.audio_dim, out_features=config.embed_dim, density=0.2),
            nn.SpikeLayerNorm(target_spikes=max(1, int(config.embed_dim * 0.25)))
        )
        
        # 3. 異種モダリティ間の連合学習層 (Cross-modal Association)
        self.text_vision_associator = nn.CrossModalAssociator(dim_a=config.embed_dim, dim_b=config.embed_dim)
        self.text_audio_associator = nn.CrossModalAssociator(dim_a=config.embed_dim, dim_b=config.embed_dim)

    def process_multimodal(
        self, 
        text_token: Optional[int] = None, 
        vision_spikes: Optional[List[int]] = None, 
        audio_spikes: Optional[List[int]] = None, 
        learning: bool = True
    ) -> Dict[str, List[int]]:
        """
        複数のモダリティの入力スパイクを同時に処理し、相互連想結果を返す。
        """
        # --- モダリティのエンコード ---
        text_embed_spikes = []
        if text_token is not None:
            # text_model内のリザバー層のスパイクを取得し、共通次元へ射影
            res_spikes = self.text_model._get_reservoir_spikes(text_token)
            text_embed_spikes = list(set([s % self.config.embed_dim for s in res_spikes]))
            # 内部でのテキスト予測学習も進める
            self.text_model.forward_step(text_token, learning=learning, target_id=None)

        encoded_vision = []
        if vision_spikes is not None:
            encoded_vision = self.vision_encoder(vision_spikes, learning=learning)
            
        encoded_audio = []
        if audio_spikes is not None:
            encoded_audio = self.audio_encoder(audio_spikes, learning=learning)

        # --- クロスモーダル連合 (STDPによる結びつけ、または想起) ---
        tv_recall = self.text_vision_associator(spikes_a=text_embed_spikes, spikes_b=encoded_vision, learning=learning)
        ta_recall = self.text_audio_associator(spikes_a=text_embed_spikes, spikes_b=encoded_audio, learning=learning)

        return {
            "vision_recall_from_text": tv_recall["recall_b"],
            "text_recall_from_vision": tv_recall["recall_a"],
            "audio_recall_from_text": ta_recall["recall_b"],
            "text_recall_from_audio": ta_recall["recall_a"]
        }
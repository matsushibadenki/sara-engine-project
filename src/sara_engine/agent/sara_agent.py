_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/agent/sara_agent.py",
    "//": "タイトル: 統合マルチモーダル・エージェント (Sara Agent)",
    "//": "目的: SNN-GPTのアテンションマスクとしてVSAシフトを含まない純粋なSDRを渡すよう修正。"
}

import os
from typing import List

from ..memory.sdr import SDREncoder
from ..core.cortex import CorticalColumn
from ..memory.hippocampus import CorticoHippocampalSystem
from ..core.prefrontal import PrefrontalCortex
from ..models.gpt import SaraGPT
from ..encoders.vision import ImageSpikeEncoder
from ..encoders.audio import AudioSpikeEncoder

class SaraAgent:
    def __init__(self, input_size: int = 2048, hidden_size: int = 4096, 
                 compartments: List[str] = ["general", "python_expert", "biology", "vision", "audio"]):
        
        self.encoder = SDREncoder(input_size=input_size, density=0.02, use_tokenizer=True, apply_vsa=True)
        self.prefrontal = PrefrontalCortex(encoder=self.encoder, compartments=compartments) # type: ignore
        
        self.cortex = CorticalColumn(
            input_size=input_size, 
            hidden_size_per_comp=hidden_size, 
            compartment_names=compartments
        )
        self.brain = CorticoHippocampalSystem(cortex=self.cortex, ltm_filepath="sara_multimodal_ltm.pkl", max_working_memory_size=15)
        
        self.gpt = SaraGPT(self.encoder)
        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)
        
        self._bootstrap()

    def _bootstrap(self):
        corpus = [
            "python_expert Python リスト 内包 表記 コード 簡潔",
            "biology ミトコンドリア 細胞 エネルギー 作り",
            "vision 画像 視覚 リンゴ 果物 赤く",
            "general 挨拶 日常 会話",
            "リスト 内包 表記 を 使う と コード は 簡潔 に 書け ます",
            "ミトコンドリア は 細胞 の エネルギー を 作り ます",
            "リンゴ は 赤く て 美味しい 果物 です"
        ]
        if not os.path.exists("sara_vocab.json"):
            self.encoder.tokenizer.train(corpus)
        
        self.encoder.train_semantic_network(corpus, window_size=3, epochs=2)
        
        for text in corpus[4:]:
            self.gpt.learn_sequence(text)
            
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        for comp in self.prefrontal.compartments:
            self.prefrontal.context_anchors[comp] = self.encoder.encode(comp)
        self.encoder.apply_vsa = original_vsa

    def perceive_image(self, image_features: List[float], label: str):
        vision_sdr = self.vision.encode(image_features)
        label_sdr = self.encoder.encode(label)
        
        bound_sdr = sorted(list(set(vision_sdr) | set(label_sdr)))
        
        self.brain.experience_and_memorize(bound_sdr, content=f"[視覚入力: {label}]", context="vision", learning=True)
        return f"[視覚野] 画像を解析し、『{label}』の概念と結合しました。"

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa
        
        context = self.prefrontal.determine_context(pfc_input_sdr)
        input_sdr = self.encoder.encode(user_text)
        
        if teaching_mode:
            if ":" in user_text:
                context = user_text.split(":")[0].strip()
            self.brain.experience_and_memorize(input_sdr, content=user_text, context=context, learning=True)
            self.gpt.learn_sequence(user_text)
            return f"[PFC: {context}] 海馬に記憶し、シーケンスを学習しました。"
            
        else:
            icl_results = self.brain.in_context_inference(current_sensory_sdr=input_sdr, context=context)
            valid_results = [res for res in icl_results if res.get('type') == context]
            
            if not valid_results:
                return f"[PFC: {context}] 関連する記憶が見つかりませんでした。"
                
            best_memory = valid_results[0]['content']
            
            # 【修正】GPTのアテンションマスクにはVSAシフトを含まない純粋なSDRを渡す
            original_vsa = self.encoder.apply_vsa
            self.encoder.apply_vsa = False
            gpt_attention_sdr = self.encoder.encode(best_memory)
            self.encoder.apply_vsa = original_vsa
            
            generated_text = self.gpt.generate(prompt=user_text, context_sdr=gpt_attention_sdr, max_tokens=10)
            
            response = f"[PFC: {context}]\n"
            response += f" >> 海馬記憶: {best_memory}\n"
            response += f" >> SNN-GPT生成: {generated_text}"
            return response
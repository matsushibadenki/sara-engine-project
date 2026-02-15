_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/agent/sara_agent.py",
    "//": "タイトル: 統合マルチモーダル・エージェント (Sara Agent)",
    "//": "目的: オンライン教示時のアンカー取り込み率の向上と、PFC判定の専門性バイアス強化。"
}

import os
import random
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
            "vision 画像 視覚 リンゴ 果物 赤く みかん",
            "audio 音声 音楽 聴覚 音",
            "general 挨拶 日常 会話 こんにちは おはよう 猫 犬 かわいい です ます",
            "リスト 内包 表記 を 使う と コード は 簡潔 に 書け ます",
            "ミトコンドリア は 細胞 の エネルギー を 作り ます",
            "リンゴ は 赤く て 美味しい 果物 です"
        ]
        if not os.path.exists("sara_vocab.json"):
            self.encoder.tokenizer.train(corpus)
        self.encoder.train_semantic_network(corpus, window_size=3, epochs=2)
        
        for text in corpus[5:]:
            self.gpt.learn_sequence(text, weight=1.0)
            
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        for comp in self.prefrontal.compartments:
            self.prefrontal.context_anchors[comp] = self.encoder.encode(comp)
        self.encoder.apply_vsa = original_vsa

    def _register_dynamic_vocab(self, text: str):
        if not (hasattr(self.encoder, 'tokenizer') and hasattr(self.encoder.tokenizer, 'vocab')):
            return
        target_text = text.split(":", 1)[1] if ":" in text else text
        for word in target_text.split():
            if word and word not in self.encoder.tokenizer.vocab:
                new_id = len(self.encoder.tokenizer.vocab)
                self.encoder.tokenizer.vocab[word] = new_id
                self.encoder.token_sdr_map[new_id] = self.encoder._get_base_token_sdr(new_id)

    def perceive_image(self, image_features: List[float], label: str):
        self._register_dynamic_vocab(label)
        vision_sdr = self.vision.encode(image_features)
        label_sdr = self.encoder.encode(label)
        bound_sdr = sorted(list(set(vision_sdr) | set(label_sdr)))
        self.brain.experience_and_memorize(bound_sdr, content=f"[視覚入力: {label}]", context="vision", learning=True)
        return f"[視覚野] 画像を解析し、『{label}』の概念と結合しました。"

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        self._register_dynamic_vocab(user_text)
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa
        
        # --- 【修正】専門性優先のPFCルーティング ---
        overlaps = {}
        for comp, anchor in self.prefrontal.context_anchors.items():
            overlap = len(set(pfc_input_sdr).intersection(set(anchor)))
            overlaps[comp] = overlap
        
        max_overlap = max(overlaps.values())
        
        # どのアンカーとも重なりが薄い場合はgeneral
        if max_overlap < 5:
            context = "general"
        else:
            # 「は」などの助詞によるgeneralへの引きずりを防ぐため、
            # スコアが近い場合はgeneralよりも専門コンパートメントを優先する
            best_comp = "general"
            best_score = -1
            for comp, score in overlaps.items():
                adjusted_score = score
                if comp != "general":
                    adjusted_score += 2 # 専門コンパートメントに僅かなボーナス
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_comp = comp
            context = best_comp
        # ----------------------------------------
        
        input_sdr = self.encoder.encode(user_text)
        
        if teaching_mode:
            if ":" in user_text:
                parts = user_text.split(":", 1)
                explicit_context = parts[0].strip()
                if explicit_context in self.prefrontal.compartments:
                    context = explicit_context
                    user_text = parts[1].strip()
            
            if context in self.prefrontal.context_anchors:
                anchor_set = set(self.prefrontal.context_anchors[context])
                # 【修正】取り込み率を40%にアップして、教えた専門用語をアンカー内で優位にする
                sample_size = max(1, int(len(pfc_input_sdr) * 0.40))
                sampled_bits = random.sample(pfc_input_sdr, min(sample_size, len(pfc_input_sdr)))
                anchor_set.update(sampled_bits)
                
                target_bits = int(self.encoder.input_size * self.encoder.density)
                if len(anchor_set) > target_bits:
                    self.prefrontal.context_anchors[context] = sorted(random.sample(list(anchor_set), target_bits))
                else:
                    self.prefrontal.context_anchors[context] = sorted(list(anchor_set))
            
            self.brain.experience_and_memorize(input_sdr, content=user_text, context=context, learning=True)
            self.gpt.learn_sequence(user_text, weight=5.0)
            return f"[PFC: {context}] 海馬に記憶し、シーケンスとアンカーを学習しました。"
            
        else:
            icl_results = self.brain.in_context_inference(current_sensory_sdr=input_sdr, context=context)
            valid_results = [res for res in icl_results if res.get('type') == context]
            
            if not valid_results:
                return f"[PFC: {context}] 関連する記憶が見つかりませんでした。"
                
            best_memory = valid_results[0]['content']
            
            original_vsa = self.encoder.apply_vsa
            self.encoder.apply_vsa = False
            gpt_attention_sdr = self.encoder.encode(best_memory)
            self.encoder.apply_vsa = original_vsa
            
            generated_text = self.gpt.generate(prompt=user_text, context_sdr=gpt_attention_sdr, max_tokens=10)
            
            response = f"[PFC: {context}]\n"
            response += f" >> 海馬記憶: {best_memory}\n"
            response += f" >> SNN-GPT生成: {generated_text}"
            return response

    def sleep(self, consolidation_epochs: int = 3) -> str:
        for comp in self.prefrontal.compartments:
            self.brain.consolidate_memories(context=comp, replay_count=consolidation_epochs)
        return "[睡眠完了] 海馬から皮質への記憶の統合(リプレイ)を実行し、シナプスを強化しました。"
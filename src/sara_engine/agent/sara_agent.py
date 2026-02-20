"""
{
    "//": "ディレクトリパス: src/sara_engine/agent/sara_agent.py",
    "//": "タイトル: 統合マルチモーダル・エージェント (Sara Agent) - PFC干渉抑制版",
    "//": "目的: 前頭前野（PFC）の認知制御を模倣し、確実なエピソード記憶が想起された場合は、ノイズとなる意味記憶（直感連想）の干渉を抑制してハルシネーションを防ぐ。"
}
"""

import os
import random
import hashlib
from typing import List, Dict, Any, Optional

from ..memory.sdr import SDREncoder
from ..core.cortex import CorticalColumn
from ..memory.hippocampus import CorticoHippocampalSystem
from ..core.prefrontal import PrefrontalCortex
from ..models.gpt import SaraGPT
from ..encoders.vision import ImageSpikeEncoder
from ..encoders.audio import AudioSpikeEncoder
from ..memory.million_token_snn import DynamicSNNMemory

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
        self.brain = CorticoHippocampalSystem(cortex=self.cortex, ltm_filepath="models/sara_multimodal_ltm.pkl", max_working_memory_size=15)
        
        self.gpt = SaraGPT(self.encoder)
        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)
        
        self.episodic_snn = DynamicSNNMemory(vocab_size=100000, sdr_size=3)
        
        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history_turns = 5

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
        if not os.path.exists("workspace/sara_vocab.json"):
            self.encoder.tokenizer.train(corpus)
        self.encoder.train_semantic_network(corpus, window_size=3, epochs=2)
        
        for text in corpus[5:]:
            self.gpt.learn_sequence(text, weight=1.0)
            
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        for comp in self.prefrontal.compartments:
            self.prefrontal.context_anchors[comp] = self.encoder.encode(comp)
        self.encoder.apply_vsa = original_vsa

    def _generate_stable_sdr(self, text: str) -> List[int]:
        hex_digest = hashlib.md5(text.encode('utf-8')).hexdigest()
        word_seed = int(hex_digest, 16)
        
        rng = random.Random(word_seed)
        
        n = self.encoder.input_size
        target_w = int(n * self.encoder.density)
        
        if 1 < len(text) <= 10:
            char_bits = set()
            for char in text:
                char_seed = int(hashlib.md5(char.encode('utf-8')).hexdigest(), 16)
                char_rng = random.Random(char_seed)
                char_bits.update(char_rng.sample(range(n), target_w))
            
            sorted_candidates = sorted(list(char_bits))
            
            if len(sorted_candidates) >= target_w:
                return sorted(rng.sample(sorted_candidates, target_w))
            else:
                needed = target_w - len(sorted_candidates)
                remaining = list(set(range(n)) - set(sorted_candidates))
                extras = rng.sample(remaining, needed)
                return sorted(sorted_candidates + extras)
        else:
            return sorted(rng.sample(range(n), target_w))

    def _register_dynamic_vocab(self, text: str):
        if not (hasattr(self.encoder, 'tokenizer') and hasattr(self.encoder.tokenizer, 'vocab')):
            return
            
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(self.encoder.tokenizer, 'split_text') else target_text.split()
        
        for word in words:
            if not word: continue
            
            is_unknown = word not in self.encoder.tokenizer.vocab
            
            if is_unknown:
                new_id = len(self.encoder.tokenizer.vocab)
                self.encoder.tokenizer.vocab[word] = new_id
                
                stable_sdr = self._generate_stable_sdr(word)
                self.encoder.token_sdr_map[new_id] = stable_sdr
            
            elif word in self.encoder.tokenizer.vocab:
                tid = self.encoder.tokenizer.vocab[word]
                if tid not in self.encoder.token_sdr_map:
                    self.encoder.token_sdr_map[tid] = self._generate_stable_sdr(word)

    def _text_to_ids(self, text: str) -> List[int]:
        if not hasattr(self.encoder, 'tokenizer'):
            return []
        
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(self.encoder.tokenizer, 'split_text') else target_text.split()
        
        ids = []
        for word in words:
            if word in self.encoder.tokenizer.vocab:
                ids.append(self.encoder.tokenizer.vocab[word])
        return ids

    def _ids_to_text(self, ids: List[int]) -> str:
        if not hasattr(self.encoder, 'tokenizer'):
            return ""
        
        id_to_word = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        words = [id_to_word[tid] for tid in ids if tid in id_to_word]
        return " ".join(words)

    def _update_history(self, role: str, text: str, sdr: List[int]):
        self.dialogue_history.append({
            "role": role,
            "text": text,
            "sdr": sdr
        })
        if len(self.dialogue_history) > self.max_history_turns * 2:
            self.dialogue_history.pop(0)

    def _get_history_context_sdr(self) -> List[int]:
        combined_sdr = set()
        decay_rate = 1.0
        min_decay = 0.2
        
        for item in reversed(self.dialogue_history):
            sdr = item["sdr"]
            if not sdr: continue
            
            sample_count = int(len(sdr) * decay_rate)
            if sample_count > 0:
                sampled_bits = random.sample(sdr, min(sample_count, len(sdr)))
                combined_sdr.update(sampled_bits)
            
            decay_rate *= 0.6
            if decay_rate < min_decay:
                break
                
        target_on_bits = int(self.encoder.input_size * 0.05)
        sdr_list = sorted(list(combined_sdr))
        if len(sdr_list) > target_on_bits:
            sdr_list = sorted(random.sample(sdr_list, target_on_bits))
            
        return sdr_list

    def perceive_image(self, image_features: List[float], label: str):
        self._register_dynamic_vocab(label)
        vision_sdr = self.vision.encode(image_features)
        label_sdr = self.encoder.encode(label)
        bound_sdr = sorted(list(set(vision_sdr) | set(label_sdr)))
        
        self._update_history("system", f"[視覚情報: {label}]", bound_sdr)
        
        self.brain.experience_and_memorize(bound_sdr, content=f"[視覚入力: {label}]", context="vision", learning=True)
        return f"[視覚野] 画像を解析し、『{label}』の概念と結合しました。"

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        self._register_dynamic_vocab(user_text)
        
        user_ids = self._text_to_ids(user_text)
        if user_ids and teaching_mode:
            self.episodic_snn.process_sequence(user_ids, is_training=True)
        
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa
        
        input_sdr = self.encoder.encode(user_text)
        self._update_history("user", user_text, input_sdr)
        history_context_sdr = self._get_history_context_sdr()

        routing_sdr = set(pfc_input_sdr)
        if history_context_sdr:
            routing_sample = random.sample(history_context_sdr, int(len(history_context_sdr) * 0.1))
            routing_sdr.update(routing_sample)
            
        overlaps = {}
        for comp, anchor in self.prefrontal.context_anchors.items():
            overlap = len(routing_sdr.intersection(set(anchor)))
            overlaps[comp] = overlap
        
        max_overlap = max(overlaps.values())
        
        if max_overlap < 5:
            context = "general"
        else:
            best_comp = "general"
            best_score = -1
            for comp, score in overlaps.items():
                adjusted_score = score
                if comp != "general":
                    adjusted_score += 2
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_comp = comp
            context = best_comp
        
        if teaching_mode:
            if ":" in user_text:
                parts = user_text.split(":", 1)
                explicit_context = parts[0].strip()
                if explicit_context in self.prefrontal.compartments:
                    context = explicit_context
                    user_text = parts[1].strip()
            
            if context in self.prefrontal.context_anchors:
                anchor_set = set(self.prefrontal.context_anchors[context])
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
            
            response_text = f"[PFC: {context}] 海馬に記憶し、シーケンスとアンカーを学習しました。"
            
            response_ids = self._text_to_ids(response_text)
            if response_ids:
                self.episodic_snn.process_sequence(response_ids, is_training=True)
                
            self._update_history("system", response_text, [])
            return response_text
            
        else:
            associated_text = ""
            associated_sdr = []
            if user_ids:
                associated_ids = self.episodic_snn.process_sequence(user_ids, is_training=False)
                associated_text = self._ids_to_text(associated_ids)
                if associated_text:
                    original_vsa = self.encoder.apply_vsa
                    self.encoder.apply_vsa = False
                    associated_sdr = self.encoder.encode(associated_text)
                    self.encoder.apply_vsa = original_vsa

            search_sdr = set(input_sdr)
            if associated_sdr:
                search_sdr.update(associated_sdr)
            
            all_results = []
            for comp in self.prefrontal.compartments:
                res = self.brain.in_context_inference(current_sensory_sdr=list(search_sdr), context=comp)
                all_results.extend(res)
            
            if all_results:
                all_results.sort(key=lambda x: x['score'], reverse=True)
                best_memory = all_results[0]['content']
            else:
                best_memory = "なし"
            
            memory_context = ""
            if best_memory != "なし":
                # 前頭前野(PFC)による意味記憶の干渉抑制
                # 海馬から確かなエピソード事実が検索できた場合、SNNの直感連想（ノイズ）を遮断し、事実に集中させる
                memory_context = best_memory
            elif associated_text:
                # 記憶が見つからない場合のみ、直感連想のキーワードを頼りにする
                memory_context = associated_text
                
            if memory_context.strip():
                original_vsa = self.encoder.apply_vsa
                self.encoder.apply_vsa = False
                memory_sdr = self.encoder.encode(memory_context.strip())
                self.encoder.apply_vsa = original_vsa
                
                generation_context_sdr = set(memory_sdr)
                
                if best_memory != "なし":
                    # ハブノード（「は」など）での脱線を防ぐため、シード軌道を先頭の3語に強化する
                    seed_words = best_memory.split()[:3]
                    seed_prompt = "".join(seed_words)
                    prompt_for_gpt = " ".join(seed_words)
                elif associated_text:
                    seed_prompt = associated_text.split()[0]
                    prompt_for_gpt = seed_prompt
                else:
                    seed_prompt = user_text[-1]
                    prompt_for_gpt = user_text
                
                generated_text = self.gpt.generate(
                    prompt=prompt_for_gpt, 
                    context_sdr=list(generation_context_sdr), 
                    max_tokens=15,
                    temperature=0.1
                )
                
                final_generated_text = seed_prompt + generated_text
            else:
                final_generated_text = "関連する明確なエピソード記憶が海馬に見つかりませんでした。"

            full_response = f"[PFC: {context}]\n"
            full_response += f" >> 海馬記憶: {best_memory}\n"
            if associated_text:
                full_response += f" >> SNN直感連想: {associated_text}\n"
            full_response += f" >> SNN-GPT生成: {final_generated_text}"
            
            response_sdr = self.encoder.encode(final_generated_text)
            self._update_history("system", final_generated_text, response_sdr)
            
            return full_response

    def sleep(self, consolidation_epochs: int = 3) -> str:
        for comp in self.prefrontal.compartments:
            self.brain.consolidate_memories(context=comp, replay_count=consolidation_epochs)
        
        self.dialogue_history = []
        
        return "[睡眠完了] 海馬から皮質への記憶の統合(リプレイ)を実行し、短期記憶バッファをクリアしました。"
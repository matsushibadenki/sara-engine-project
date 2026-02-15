_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/agent/sara_agent.py",
    "//": "タイトル: 統合マルチモーダル・エージェント (Sara Agent)",
    "//": "目的: 動的語彙学習を強化。文字列ハッシュによる決定論的SDR生成と、文字単位のビット合成（Morphological SDR）を導入し、未知語の意味的安定性を向上させる。"
}

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
        
        # --- 会話履歴（短期記憶）バッファ ---
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

    def _generate_stable_sdr(self, text: str) -> List[int]:
        """
        決定論的かつ形態素的な意味を持つSDRを生成する。
        1. 文字列のハッシュ値をシードに使い、再起動後も同じSDRを保証（安定化）。
        2. 文字単位のSDRを合成し、字面が似ている単語（自動車/自転車）に類似性を持たせる（意味強化）。
        """
        # 1. 単語全体のハッシュからシードを生成（決定論的ランダムの基礎）
        hex_digest = hashlib.md5(text.encode('utf-8')).hexdigest()
        word_seed = int(hex_digest, 16)
        
        # 影響範囲を限定したローカルな乱数生成器
        rng = random.Random(word_seed)
        
        n = self.encoder.input_size
        target_w = int(n * self.encoder.density)
        
        # 2. 文字単位SDRの合成 (Morphological SDR Mixing)
        # 1文字〜10文字程度の単語に対して、構成文字のSDRを混ぜ合わせる
        if 1 < len(text) <= 10:
            char_bits = set()
            for char in text:
                # 文字ごとのハッシュとSDR生成
                char_seed = int(hashlib.md5(char.encode('utf-8')).hexdigest(), 16)
                char_rng = random.Random(char_seed)
                # 文字は単語全体よりも少しスパースに生成して合成時の過密を防ぐ
                char_bits.update(char_rng.sample(range(n), target_w))
            
            sorted_candidates = sorted(list(char_bits))
            
            # 合成結果がターゲットより大きければ、単語シードを使って決定論的に間引く
            if len(sorted_candidates) >= target_w:
                return sorted(rng.sample(sorted_candidates, target_w))
            else:
                # ターゲットより小さければ（稀）、不足分を単語シード由来のランダムビットで埋める
                needed = target_w - len(sorted_candidates)
                remaining = list(set(range(n)) - set(sorted_candidates))
                extras = rng.sample(remaining, needed)
                return sorted(sorted_candidates + extras)
        else:
            # 1文字の単語や長すぎる文章は、単純なハッシュベースのSDRを使用
            return sorted(rng.sample(range(n), target_w))

    def _register_dynamic_vocab(self, text: str):
        """動的語彙登録の強化版：Janome対応 + Stable SDR生成"""
        if not (hasattr(self.encoder, 'tokenizer') and hasattr(self.encoder.tokenizer, 'vocab')):
            return
            
        # チャットなどから渡された生のテキストから登録対象を抽出
        target_text = text.split(":", 1)[1] if ":" in text else text
        
        # Tokenizerの split_text() を経由して形態素解析を行う
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(self.encoder.tokenizer, 'split_text') else target_text.split()
        
        for word in words:
            if not word: continue
            
            # 未知語、またはSDRマップに未登録の単語を処理
            is_unknown = word not in self.encoder.tokenizer.vocab
            
            if is_unknown:
                # 新しいIDを発行
                new_id = len(self.encoder.tokenizer.vocab)
                self.encoder.tokenizer.vocab[word] = new_id
                
                # 【重要】ランダムではなく、決定論的で意味的なSDRを生成して登録
                stable_sdr = self._generate_stable_sdr(word)
                self.encoder.token_sdr_map[new_id] = stable_sdr
            
            elif word in self.encoder.tokenizer.vocab:
                # 既知語でもSDRマップに存在しない場合（ロード直後の不整合など）の修復
                tid = self.encoder.tokenizer.vocab[word]
                if tid not in self.encoder.token_sdr_map:
                    self.encoder.token_sdr_map[tid] = self._generate_stable_sdr(word)

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
        # ラベルも安定SDRで登録
        self._register_dynamic_vocab(label)
        vision_sdr = self.vision.encode(image_features)
        label_sdr = self.encoder.encode(label)
        bound_sdr = sorted(list(set(vision_sdr) | set(label_sdr)))
        
        self._update_history("system", f"[視覚情報: {label}]", bound_sdr)
        
        self.brain.experience_and_memorize(bound_sdr, content=f"[視覚入力: {label}]", context="vision", learning=True)
        return f"[視覚野] 画像を解析し、『{label}』の概念と結合しました。"

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        # 入力文に含まれる未知語を全て安定SDRとして登録・学習
        self._register_dynamic_vocab(user_text)
        
        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa
        
        input_sdr = self.encoder.encode(user_text)
        self._update_history("user", user_text, input_sdr)
        history_context_sdr = self._get_history_context_sdr()

        # --- PFCルーティング ---
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
        # -----------------------
        
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
            self._update_history("system", response_text, [])
            return response_text
            
        else:
            search_sdr = set(input_sdr)
            if history_context_sdr:
                search_sample = random.sample(history_context_sdr, int(len(history_context_sdr) * 0.2))
                search_sdr.update(search_sample)
            
            icl_results = self.brain.in_context_inference(current_sensory_sdr=list(search_sdr), context=context)
            valid_results = [res for res in icl_results if res.get('type') == context]
            
            if not valid_results:
                return f"[PFC: {context}] 関連する記憶が見つかりませんでした。"
                
            best_memory = valid_results[0]['content']
            
            original_vsa = self.encoder.apply_vsa
            self.encoder.apply_vsa = False
            memory_sdr = self.encoder.encode(best_memory)
            self.encoder.apply_vsa = original_vsa
            
            generation_context_sdr = set(memory_sdr)
            if history_context_sdr:
                generation_context_sdr.update(history_context_sdr)
            
            generated_text = self.gpt.generate(
                prompt=user_text, 
                context_sdr=list(generation_context_sdr), 
                max_tokens=20
            )
            
            response_text = generated_text
            full_response = f"[PFC: {context}]\n"
            full_response += f" >> 海馬記憶: {best_memory}\n"
            full_response += f" >> SNN-GPT生成: {response_text}"
            
            response_sdr = self.encoder.encode(response_text)
            self._update_history("system", response_text, response_sdr)
            
            return full_response

    def sleep(self, consolidation_epochs: int = 3) -> str:
        for comp in self.prefrontal.compartments:
            self.brain.consolidate_memories(context=comp, replay_count=consolidation_epochs)
        
        self.dialogue_history = []
        
        return "[睡眠完了] 海馬から皮質への記憶の統合(リプレイ)を実行し、短期記憶バッファをクリアしました。"
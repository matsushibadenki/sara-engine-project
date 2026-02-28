_FILE_INFO = {
    "//": "配置するディレクトリのパス: src/sara_engine/agent/sara_agent.py",
    "//": "ファイルの日本語タイトル: 統合マルチモーダル・エージェント",
    "//": "ファイルの目的や内容: SNNのアクションスパイクで外部ツールと連携するエージェント機能。Mypy対応のため絶対インポートに変更。"
}

import os
import random
import hashlib
from typing import List, Dict, Any, Callable, Optional

# Mypy対応: 相対インポートから絶対インポートへ変更
from sara_engine.memory.sdr import SDREncoder
from sara_engine.core.cortex import CorticalColumn
from sara_engine.memory.hippocampus import CorticoHippocampalSystem
from sara_engine.core.prefrontal import PrefrontalCortex
from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.encoders.vision import ImageSpikeEncoder
from sara_engine.encoders.audio import AudioSpikeEncoder
from sara_engine.memory.million_token_snn import DynamicSNNMemory

class SaraAgent:
    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 4096,
        compartments: List[str] = [
            "general",
            "python_expert",
            "biology",
            "vision",
            "audio",
        ],
    ):
        self.encoder = SDREncoder(
            input_size=input_size, density=0.02, use_tokenizer=True, apply_vsa=True
        )
        self.prefrontal = PrefrontalCortex(
            encoder=self.encoder, compartments=compartments
        )

        self.cortex = CorticalColumn(
            input_size=input_size,
            hidden_size_per_comp=hidden_size,
            compartment_names=compartments,
        )
        
        self.brain = CorticoHippocampalSystem(
            cortex=self.cortex,
            ltm_filepath="models/sara_multimodal_ltm.pkl",
            max_working_memory_size=15,
        )

        self.llm = SpikingLLM(num_layers=2, sdr_size=128, vocab_size=100000, enable_learning=True)
        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)
        self.episodic_snn = DynamicSNNMemory(vocab_size=100000, sdr_size=3)

        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history_turns = 5

        self.tools: Dict[str, Callable[[str], str]] = {}
        self._bootstrap()

    def register_tool(self, trigger_spike: str, tool_func: Callable[[str], str]) -> None:
        self.tools[trigger_spike] = tool_func
        self._register_dynamic_vocab(f"tool_trigger {trigger_spike}")

    def _bootstrap(self) -> None:
        corpus = [
            "python_expert Python リスト 内包 表記 コード 簡潔",
            "biology ミトコンドリア 細胞 エネルギー 作り",
            "vision 画像 視覚 リンゴ 果物 赤く みかん",
            "audio 音声 音楽 聴覚 音",
            "general 挨拶 日常 会話 こんにちは おはよう 猫 犬 かわいい です ます",
            "リスト 内包 表記 を 使う と コード は 簡潔 に 書け ます",
            "ミトコンドリア は 細胞 の エネルギー を 作り ます",
        ]
        if not os.path.exists("workspace/sara_vocab.json"):
            self.encoder.tokenizer.train(corpus)
        self.encoder.train_semantic_network(corpus, window_size=3, epochs=2)

        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        for comp in self.prefrontal.compartments:
            self.prefrontal.context_anchors[comp] = set(self.encoder.encode(comp))
        self.encoder.apply_vsa = original_vsa

    def _generate_stable_sdr(self, text: str) -> List[int]:
        hex_digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        word_seed = int(hex_digest, 16)
        rng = random.Random(word_seed)
        n = self.encoder.input_size
        target_w = int(n * self.encoder.density)
        return sorted(rng.sample(range(n), target_w))

    def _register_dynamic_vocab(self, text: str) -> None:
        if not hasattr(self.encoder, "tokenizer"): return
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(self.encoder.tokenizer, "split_text") else target_text.split()
        
        for word in words:
            if not word: continue
            if word not in self.encoder.tokenizer.vocab:
                new_id = len(self.encoder.tokenizer.vocab)
                self.encoder.tokenizer.vocab[word] = new_id
                self.encoder.token_sdr_map[new_id] = self._generate_stable_sdr(word)

    def _text_to_ids(self, text: str) -> List[int]:
        if not hasattr(self.encoder, "tokenizer"): return []
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(self.encoder.tokenizer, "split_text") else target_text.split()
        return [self.encoder.tokenizer.vocab[w] for w in words if w in self.encoder.tokenizer.vocab]

    def _ids_to_text(self, ids: List[int]) -> str:
        if not hasattr(self.encoder, "tokenizer"): return ""
        id_to_word = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        return "".join([id_to_word[tid] for tid in ids if tid in id_to_word])

    def _update_history(self, role: str, text: str, sdr: List[int]) -> None:
        self.dialogue_history.append({"role": role, "text": text, "sdr": sdr})
        if len(self.dialogue_history) > self.max_history_turns * 2:
            self.dialogue_history.pop(0)

    def _get_history_context_sdr(self) -> List[int]:
        combined_sdr: set[int] = set()
        decay_rate = 1.0
        for item in reversed(self.dialogue_history):
            sdr = item["sdr"]
            if not sdr: continue
            sample_count = int(len(sdr) * decay_rate)
            if sample_count > 0:
                combined_sdr.update(random.sample(sdr, min(sample_count, len(sdr))))
            decay_rate *= 0.6
            if decay_rate < 0.2: break
        target_on_bits = int(self.encoder.input_size * 0.05)
        sdr_list = sorted(list(combined_sdr))
        if len(sdr_list) > target_on_bits:
            return sorted(random.sample(sdr_list, target_on_bits))
        return sdr_list

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        self._register_dynamic_vocab(user_text)
        user_ids = self._text_to_ids(user_text)
        
        if user_ids and teaching_mode:
            self.episodic_snn.process_sequence(user_ids, is_training=True)
            self.llm.learn_sequence(user_ids)

        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa

        self._update_history("user", user_text, input_sdr)
        history_context_sdr = self._get_history_context_sdr()

        routing_sdr = set(pfc_input_sdr)
        if history_context_sdr:
            routing_sdr.update(random.sample(history_context_sdr, int(len(history_context_sdr) * 0.1)))

        overlaps = {comp: len(routing_sdr.intersection(set(anchor))) + (2 if comp != "general" else 0) 
                    for comp, anchor in self.prefrontal.context_anchors.items()}
        sorted_experts = sorted(overlaps.items(), key=lambda x: x[1], reverse=True)
        active_experts = [comp for comp, score in sorted_experts[:2] if score > 0]
        if not active_experts or sorted_experts[0][1] < 5:
            active_experts = ["general"]

        if teaching_mode:
            context = active_experts[0]
            if ":" in user_text:
                parts = user_text.split(":", 1)
                if parts[0].strip() in self.prefrontal.compartments:
                    context = parts[0].strip()
                    user_text = parts[1].strip()

            if context in self.prefrontal.context_anchors:
                anchor_set = set(self.prefrontal.context_anchors[context])
                sampled_bits = random.sample(pfc_input_sdr, min(max(1, int(len(pfc_input_sdr) * 0.4)), len(pfc_input_sdr)))
                anchor_set.update(sampled_bits)
                target_bits = int(self.encoder.input_size * self.encoder.density)
                self.prefrontal.context_anchors[context] = set(random.sample(list(anchor_set), target_bits) if len(anchor_set) > target_bits else anchor_set)

            self.brain.experience_and_memorize(input_sdr, content=user_text, context=context, learning=True)
            response_text = f"[MoE Router: {context} Expert] 海馬とSpikingLLMに記憶を定着させました。"
            self._update_history("system", response_text, [])
            return response_text

        associated_text = ""
        associated_sdr: List[int] = []
        if user_ids:
            associated_ids = self.episodic_snn.process_sequence(user_ids, is_training=False)
            associated_text = self._ids_to_text(associated_ids)
            if associated_text:
                self.encoder.apply_vsa = False
                associated_sdr = self.encoder.encode(associated_text)
                self.encoder.apply_vsa = True

        search_sdr = set(input_sdr)
        if associated_sdr: search_sdr.update(associated_sdr)

        all_retrieved: List[Dict[str, Any]] = []
        for comp in active_experts:
            all_retrieved.extend(self.brain.in_context_inference(current_sensory_sdr=list(search_sdr), context=comp))

        if all_retrieved:
            all_retrieved.sort(key=lambda x: x["score"], reverse=True)
            best_memories = [m["content"] for m in all_retrieved[:2]]
            blended_memory = " | ".join(best_memories)
            memory_context = best_memories[0]
        else:
            blended_memory, memory_context = "なし", associated_text

        if memory_context.strip():
            prompt_for_llm = f"{memory_context} {user_text}"
            current_prompt_ids = self._text_to_ids(prompt_for_llm)
            generated_tokens: List[int] = []
            
            for trigger_spike, tool_func in self.tools.items():
                if prompt_for_llm.strip().endswith(trigger_spike):
                    tool_result = tool_func(prompt_for_llm)
                    feedback_str = f" {tool_result} = "
                    feedback_ids = self._text_to_ids(feedback_str)
                    generated_tokens.extend(feedback_ids)
                    current_prompt_ids.extend(feedback_ids)
                    break

            max_agent_steps = 20
            for step in range(max_agent_steps):
                new_ids = self.llm.generate(prompt_tokens=current_prompt_ids, max_new_tokens=1, temperature=0.1)
                if not new_ids:
                    break
                
                token_id = new_ids[0]
                generated_tokens.append(token_id)
                current_prompt_ids.append(token_id)
                
                current_full_str = prompt_for_llm + self._ids_to_text(generated_tokens)
                
                tool_triggered = False
                for trigger_spike, tool_func in self.tools.items():
                    if current_full_str.strip().endswith(trigger_spike):
                        tool_result = tool_func(current_full_str)
                        feedback_str = f" {tool_result} = "
                        feedback_ids = self._text_to_ids(feedback_str)
                        generated_tokens.extend(feedback_ids)
                        current_prompt_ids.extend(feedback_ids)
                        tool_triggered = True
                        break
                        
                if current_full_str.strip().endswith("。") or current_full_str.strip().endswith("？"):
                    break

            final_generated_text = prompt_for_llm + self._ids_to_text(generated_tokens)
        else:
            final_generated_text = "関連する記憶が見つからず、推論を開始できませんでした。"

        full_response = f"[MoE Router: {', '.join(active_experts)} 活性化]\n"
        full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
        full_response += f" >> SNN-LLM自律生成: {final_generated_text}"

        self._update_history("system", final_generated_text, self.encoder.encode(final_generated_text))
        return full_response
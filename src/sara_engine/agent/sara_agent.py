# ファイルメタ情報
_FILE_INFO = {
    "//": "配置するディレクトリのパス: src/sara_engine/agent/sara_agent.py",
    "//": "ファイルの日本語タイトル: 統合マルチモーダル・エージェント (Retrieval-Augmented MoE + Function Calling 実装版)",
    "//": "ファイルの目的や内容: 論文「Agentic Context Engineering (ACE)」と「Retrieval-Augmented Mixture of Experts (MoE)」の概念を統合。さらに、SNNの特定の出力スパイクを「運動コマンド」として捉え、Python側で外部ツールを実行して結果を「感覚入力」として差し戻す生体模倣型のエージェンティックループを実装。"
}

import os
import random
import hashlib
import re
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
        )  # type: ignore

        self.cortex = CorticalColumn(
            input_size=input_size,
            hidden_size_per_comp=hidden_size,
            compartment_names=compartments,
        )
        
        # 外部記憶（Hippocampus & LTM）
        self.brain = CorticoHippocampalSystem(
            cortex=self.cortex,
            ltm_filepath="models/sara_multimodal_ltm.pkl",
            max_working_memory_size=15,
        )

        self.gpt = SaraGPT(self.encoder)
        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)

        self.episodic_snn = DynamicSNNMemory(vocab_size=100000, sdr_size=3)

        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history_turns = 5

        # ツール定義（キーワードに対する関数マッピング）
        self.tools = {
            "<CALC>": self._execute_calc,
            "<SEARCH>": self._execute_search
        }

        self._bootstrap()

    def _execute_calc(self, context: str) -> str:
        """直前の文脈から数式を抽出し、計算結果を返すツール"""
        target_context = context.split("<CALC>")[0] if "<CALC>" in context else context
        match = re.search(r'([0-9\+\-\*\/\s\(\)\.]+)(?:は|の計算|はいくつ|の答え|)$', target_context.strip())
        
        if not match:
            words = target_context.split()
            expression = words[-1] if words else ""
        else:
            expression = match.group(1).strip()

        allowed_chars = set("0123456789+-*/(). ")
        if not expression or not all(c in allowed_chars for c in expression):
            return "[CALC_ERROR]"
            
        try:
            result = eval(expression)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception:
            return "[CALC_ERROR]"

    def _execute_search(self, context: str) -> str:
        """直前の文脈から検索クエリを抽出し、ダミーの検索結果を返すツール"""
        return "検索結果: SARAエンジンはスパイクのみで動作する次世代AIです。"

    def _bootstrap(self):
        corpus = [
            "python_expert Python リスト 内包 表記 コード 簡潔",
            "biology ミトコンドリア 細胞 エネルギー 作り",
            "vision 画像 視覚 リンゴ 果物 赤く みかん",
            "audio 音声 音楽 聴覚 音",
            "general 挨拶 日常 会話 こんにちは おはよう 猫 犬 かわいい です ます",
            "calc 計算 ツール <CALC> = ",
            "search 検索 調べる <SEARCH>",
            "リスト 内包 表記 を 使う と コード は 簡潔 に 書け ます",
            "ミトコンドリア は 細胞 の エネルギー を 作り ます",
            "リンゴ は 赤く て 美味しい 果物 です",
        ]
        if not os.path.exists("workspace/sara_vocab.json"):
            self.encoder.tokenizer.train(corpus)
        self.encoder.train_semantic_network(corpus, window_size=3, epochs=2)

        for text in corpus[7:]:
            self.gpt.learn_sequence(text, weight=1.0)

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

        if 1 < len(text) <= 10:
            char_bits = set()
            for char in text:
                char_seed = int(hashlib.md5(char.encode("utf-8")).hexdigest(), 16)
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
        if not (
            hasattr(self.encoder, "tokenizer")
            and hasattr(self.encoder.tokenizer, "vocab")
        ):
            return

        target_text = text.split(":", 1)[1] if ":" in text else text
        words = (
            self.encoder.tokenizer.split_text(target_text)
            if hasattr(self.encoder.tokenizer, "split_text")
            else target_text.split()
        )

        for word in words:
            if not word:
                continue

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
        if not hasattr(self.encoder, "tokenizer"):
            return []

        target_text = text.split(":", 1)[1] if ":" in text else text
        words = (
            self.encoder.tokenizer.split_text(target_text)
            if hasattr(self.encoder.tokenizer, "split_text")
            else target_text.split()
        )

        ids = []
        for word in words:
            if word in self.encoder.tokenizer.vocab:
                ids.append(self.encoder.tokenizer.vocab[word])
        return ids

    def _ids_to_text(self, ids: List[int]) -> str:
        if not hasattr(self.encoder, "tokenizer"):
            return ""

        id_to_word = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        words = [id_to_word[tid] for tid in ids if tid in id_to_word]
        return " ".join(words)

    def _update_history(self, role: str, text: str, sdr: List[int]):
        self.dialogue_history.append({"role": role, "text": text, "sdr": sdr})
        if len(self.dialogue_history) > self.max_history_turns * 2:
            self.dialogue_history.pop(0)

    def _get_history_context_sdr(self) -> List[int]:
        combined_sdr = set()
        decay_rate = 1.0
        min_decay = 0.2

        for item in reversed(self.dialogue_history):
            sdr = item["sdr"]
            if not sdr:
                continue

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

        self.brain.experience_and_memorize(
            bound_sdr, content=f"[視覚入力: {label}]", context="vision", learning=True
        )
        return f"[視覚野] 画像を解析し、『{label}』の概念と結合しました。"

    def recognize_image(self, image_features: List[float], question: str = "") -> str:
        vision_sdr = self.vision.encode(image_features)
        search_sdr = set(vision_sdr)
        
        if question:
            self._register_dynamic_vocab(question)
            text_sdr = self.encoder.encode(question)
            self._update_history("user", f"[画像入力] {question}", text_sdr)
        else:
            self._update_history("user", "[画像入力のみ]", vision_sdr)

        all_results = self.brain.in_context_inference(
            current_sensory_sdr=list(search_sdr), context="vision"
        )

        vision_memories = [res for res in all_results if "[視覚入力:" in res["content"]]

        if vision_memories:
            vision_memories.sort(key=lambda x: x["score"], reverse=True)
            best_memory = vision_memories[0]
            
            if best_memory["score"] > 0.15:
                memory_content = best_memory["content"]
                
                final_response = f"[PFC: vision (Cross-Modal)]\n"
                final_response += f" >> 視覚記憶からの直接想起: {memory_content}"
                
                concept_text = memory_content.replace("[視覚入力: ", "").replace("]", "").strip()
                
                original_vsa = self.encoder.apply_vsa
                self.encoder.apply_vsa = False
                concept_sdr = self.encoder.encode(concept_text)
                self.encoder.apply_vsa = original_vsa
                
                assoc_results = []
                for comp in self.prefrontal.compartments:
                    if comp != "vision":
                        assoc_results.extend(self.brain.in_context_inference(concept_sdr, context=comp))
                    
                if assoc_results:
                    assoc_results.sort(key=lambda x: x["score"], reverse=True)
                    for assoc in assoc_results:
                        assoc_memory = assoc["content"]
                        if "視覚入力" not in assoc_memory:
                            final_response += f"\n >> クロスモーダル連想エピソード: {assoc_memory}"
                            break

                response_sdr = self.encoder.encode(final_response)
                self._update_history("system", final_response, response_sdr)
                return final_response

        fallback_msg = "この画像に関連する明確な記憶は見つかりませんでした。"
        response_sdr = self.encoder.encode(fallback_msg)
        self._update_history("system", fallback_msg, response_sdr)
        return f"[PFC: vision (Cross-Modal)]\n >> 視覚野: {fallback_msg}"

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
            routing_sample = random.sample(
                history_context_sdr, int(len(history_context_sdr) * 0.1)
            )
            routing_sdr.update(routing_sample)

        # --- Retrieval-Augmented MoE Routing (SDR Base) ---
        overlaps = {}
        for comp, anchor in self.prefrontal.context_anchors.items():
            overlap = len(routing_sdr.intersection(set(anchor)))
            if comp != "general":
                overlap += 2
            overlaps[comp] = overlap

        sorted_experts = sorted(overlaps.items(), key=lambda x: x[1], reverse=True)
        top_k = 2
        active_experts = [comp for comp, score in sorted_experts[:top_k] if score > 0]
        
        if not active_experts or sorted_experts[0][1] < 5:
            active_experts = ["general"]

        if teaching_mode:
            context = active_experts[0]
            if ":" in user_text:
                parts = user_text.split(":", 1)
                explicit_context = parts[0].strip()
                if explicit_context in self.prefrontal.compartments:
                    context = explicit_context
                    user_text = parts[1].strip()

            if context in self.prefrontal.context_anchors:
                anchor_set = set(self.prefrontal.context_anchors[context])
                sample_size = max(1, int(len(pfc_input_sdr) * 0.40))
                sampled_bits = random.sample(
                    pfc_input_sdr, min(sample_size, len(pfc_input_sdr))
                )
                anchor_set.update(sampled_bits)

                target_bits = int(self.encoder.input_size * self.encoder.density)
                if len(anchor_set) > target_bits:
                    self.prefrontal.context_anchors[context] = set(
                        random.sample(list(anchor_set), target_bits)
                    )
                else:
                    self.prefrontal.context_anchors[context] = set(anchor_set)

            self.brain.experience_and_memorize(
                input_sdr, content=user_text, context=context, learning=True
            )
            self.gpt.learn_sequence(user_text, weight=5.0)

            response_text = (
                f"[MoE Router: {context} Expertに割当] 海馬に記憶し、シーケンスとアンカーを学習しました。"
            )

            response_ids = self._text_to_ids(response_text)
            if response_ids:
                self.episodic_snn.process_sequence(response_ids, is_training=True)

            self._update_history("system", response_text, [])
            return response_text

        else:
            associated_text = ""
            associated_sdr = []
            if user_ids:
                associated_ids = self.episodic_snn.process_sequence(
                    user_ids, is_training=False
                )
                associated_text = self._ids_to_text(associated_ids)
                if associated_text:
                    original_vsa = self.encoder.apply_vsa
                    self.encoder.apply_vsa = False
                    associated_sdr = self.encoder.encode(associated_text)
                    self.encoder.apply_vsa = original_vsa

            search_sdr = set(input_sdr)
            if associated_sdr:
                search_sdr.update(associated_sdr)

            all_retrieved_memories = []
            for comp in active_experts:
                res = self.brain.in_context_inference(
                    current_sensory_sdr=list(search_sdr), context=comp
                )
                all_retrieved_memories.extend(res)

            if all_retrieved_memories:
                all_retrieved_memories.sort(key=lambda x: x["score"], reverse=True)
                best_memories = [m["content"] for m in all_retrieved_memories[:2]]
                blended_memory = " | ".join(best_memories)
                best_memory_for_gen = best_memories[0]
            else:
                blended_memory = "なし"
                best_memory_for_gen = "なし"

            memory_context = ""
            if best_memory_for_gen != "なし":
                memory_context = best_memory_for_gen
            elif associated_text:
                memory_context = associated_text

            if memory_context.strip():
                original_vsa = self.encoder.apply_vsa
                self.encoder.apply_vsa = False
                memory_sdr = self.encoder.encode(memory_context.strip())
                self.encoder.apply_vsa = original_vsa

                generation_context_sdr = set(memory_sdr)
                prompt_for_gpt = user_text
                
                # --- Agentic Loop (Action Spikes) ---
                max_agent_steps = 3
                current_prompt = prompt_for_gpt
                generated_segments = []
                
                for step in range(max_agent_steps):
                    gen_text = self.gpt.generate(
                        prompt=current_prompt,
                        context_sdr=list(generation_context_sdr),
                        max_tokens=15,
                        temperature=0.1,
                    )
                    generated_segments.append(gen_text)
                    current_full = prompt_for_gpt + "".join(generated_segments)
                    
                    # 運動コマンド（アクションスパイク）の検知と実行
                    if "<CALC>" in gen_text or ("<CALC>" in current_prompt and "=" not in current_full.split("<CALC>")[-1]):
                        print(f"\n[AGENT] Action Spike Fired: <CALC>")
                        calc_res = self._execute_calc(current_full)
                        print(f"[AGENT] Tool Executed. Result -> {calc_res}")
                        feedback = f" {calc_res} = "
                        generated_segments.append(feedback)
                        current_prompt = current_full + feedback
                        
                        # 外部入力を文脈（SDR）に差し戻す
                        self.encoder.apply_vsa = False
                        generation_context_sdr.update(self.encoder.encode(feedback))
                        self.encoder.apply_vsa = True
                        
                    elif "<SEARCH>" in gen_text or ("<SEARCH>" in current_prompt and "]" not in current_full.split("<SEARCH>")[-1]):
                        print(f"\n[AGENT] Action Spike Fired: <SEARCH>")
                        search_res = self._execute_search(current_full)
                        print(f"[AGENT] Tool Executed. Result -> {search_res}")
                        feedback = f" [{search_res}] "
                        generated_segments.append(feedback)
                        current_prompt = current_full + feedback
                        
                        self.encoder.apply_vsa = False
                        generation_context_sdr.update(self.encoder.encode(feedback))
                        self.encoder.apply_vsa = True
                    else:
                        break  # 発火がない場合は終了
                        
                final_generated_text = prompt_for_gpt + "".join(generated_segments)
            else:
                final_generated_text = (
                    "関連する明確なエピソード記憶が海馬に見つかりませんでした。"
                )

            active_experts_str = ", ".join(active_experts)
            full_response = f"[MoE Router: {active_experts_str} Experts 活性化]\n"
            full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
            if associated_text:
                full_response += f" >> SNN直感連想: {associated_text}\n"
            full_response += f" >> SNN-GPT生成: {final_generated_text}"

            response_sdr = self.encoder.encode(final_generated_text)
            self._update_history("system", final_generated_text, response_sdr)

            return full_response

    def _reflect_on_history(self) -> Dict[str, set]:
        insights_by_comp = {comp: set() for comp in self.prefrontal.compartments}
        
        for i in range(len(self.dialogue_history) - 1):
            if self.dialogue_history[i]["role"] == "user" and self.dialogue_history[i+1]["role"] == "system":
                user_sdr = set(self.dialogue_history[i]["sdr"])
                sys_sdr = set(self.dialogue_history[i+1]["sdr"])
                
                insight_sdr = user_sdr.intersection(sys_sdr)
                
                if insight_sdr:
                    best_comp = "general"
                    best_score = -1
                    for comp, anchor in self.prefrontal.context_anchors.items():
                        score = len(insight_sdr.intersection(set(anchor)))
                        if score > best_score:
                            best_score = score
                            best_comp = comp
                            
                    insights_by_comp[best_comp].update(insight_sdr)
                    
        return insights_by_comp

    def _curate_playbook(self, insights_by_comp: Dict[str, set]):
        target_bits = int(self.encoder.input_size * self.encoder.density)
        
        for comp, new_insights in insights_by_comp.items():
            if not new_insights:
                continue
                
            current_anchor = set(self.prefrontal.context_anchors[comp])
            current_anchor.update(new_insights)
            
            if len(current_anchor) > target_bits:
                core_retention = int(target_bits * 0.7)
                original_anchor = set(self.prefrontal.context_anchors[comp])
                core_candidates = list(original_anchor.intersection(current_anchor))
                
                if len(core_candidates) >= core_retention:
                    core_bits = random.sample(core_candidates, core_retention)
                else:
                    core_bits = core_candidates
                    
                remaining_novel = list(new_insights - set(core_bits))
                novel_allowance = target_bits - len(core_bits)
                novel_bits = random.sample(remaining_novel, min(len(remaining_novel), novel_allowance))
                
                final_bits = set(core_bits + novel_bits)
                
                if len(final_bits) < target_bits:
                    leftovers = list(current_anchor - final_bits)
                    fillers = random.sample(leftovers, min(len(leftovers), target_bits - len(final_bits)))
                    final_bits.update(fillers)
                    
                self.prefrontal.context_anchors[comp] = final_bits
            else:
                self.prefrontal.context_anchors[comp] = current_anchor

    def sleep(self, consolidation_epochs: int = 3) -> str:
        insights = self._reflect_on_history()
        self._curate_playbook(insights)
        
        for comp in self.prefrontal.compartments:
            self.brain.consolidate_memories(
                context=comp, replay_count=consolidation_epochs
            )

        self.dialogue_history = []
        return "[睡眠完了] ACEフレームワーク(Reflector/Curator)によるPFCアンカーの差分進化と、海馬記憶の統合を実行しました。"
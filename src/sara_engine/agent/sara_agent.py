# ディレクトリパス: src/sara_engine/agent/sara_agent.py
# ファイルの日本語タイトル: 統合マルチモーダル・エージェント
# ファイルの目的や内容: 学習時の文脈自動ラベル付け（代名詞解決用）と、指示語検知によるスマートなキーワードフィルタリングを実装した最終版。

from ..memory.million_token_snn import DynamicSNNMemory
from ..encoders.audio import AudioSpikeEncoder
from ..encoders.vision import ImageSpikeEncoder
from ..models.spiking_llm import SpikingLLM
from ..core.prefrontal import PrefrontalCortex
from ..memory.hippocampus import CorticoHippocampalSystem
from ..core.cortex import CorticalColumn
from ..memory.sdr import SDREncoder
from ..utils.dialogue import AdaptiveTopicTracker
from typing import List, Dict, Any, Callable, Optional, Tuple
import hashlib
import random
import os
import re
import pickle
import math

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

        self.llm = SpikingLLM(num_layers=2, sdr_size=128,
                              vocab_size=100000, enable_learning=True)
        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)
        self.episodic_snn = DynamicSNNMemory(vocab_size=100000, sdr_size=3)

        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history_turns = 5
        self.dialogue_state: Dict[str, Any] = {"current_topic": None, "last_intent": None}
        self.topic_tracker = AdaptiveTopicTracker()

        self.tools: Dict[str, Callable[[str], str]] = {}
        self._bootstrap()

    def save_agent(self, save_dir: str = "workspace/models/sara_agent") -> None:
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.llm, "save_pretrained"):
            try:
                self.llm.save_pretrained(save_dir)
            except Exception as e:
                print(f"⚠️ LLMの保存に失敗しました: {e}")
        try:
            with open(os.path.join(save_dir, "episodic_snn.pkl"), "wb") as f:
                pickle.dump(self.episodic_snn, f)
            with open(os.path.join(save_dir, "brain_ltm.pkl"), "wb") as f:
                pickle.dump(self.brain, f)
        except Exception as e:
            print(f"⚠️ 記憶の保存に失敗しました: {e}")

    def load_agent(self, load_dir: str = "workspace/models/sara_agent") -> None:
        if not os.path.exists(load_dir):
            return
        if hasattr(self.llm, "load_pretrained"):
            try:
                from ..models.spiking_llm import SpikingLLM
                self.llm = SpikingLLM.from_pretrained(load_dir)
            except Exception as e:
                print(f"⚠️ LLMのロードに失敗しました: {e}")
        try:
            episodic_path = os.path.join(load_dir, "episodic_snn.pkl")
            if os.path.exists(episodic_path):
                with open(episodic_path, "rb") as f:
                    self.episodic_snn = pickle.load(f)
            brain_path = os.path.join(load_dir, "brain_ltm.pkl")
            if os.path.exists(brain_path):
                with open(brain_path, "rb") as f:
                    self.brain = pickle.load(f)
        except Exception as e:
            print(f"⚠️ 記憶のロードに失敗しました: {e}")

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
            self.prefrontal.context_anchors[comp] = set(
                self.encoder.encode(comp))
        self.encoder.apply_vsa = original_vsa

    def _generate_stable_sdr(self, text: str) -> List[int]:
        hex_digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        word_seed = int(hex_digest, 16)
        rng = random.Random(word_seed)
        n = self.encoder.input_size
        target_w = int(n * self.encoder.density)
        return sorted(rng.sample(range(n), target_w))

    def _register_dynamic_vocab(self, text: str) -> None:
        if not hasattr(self.encoder, "tokenizer"):
            return
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(
            self.encoder.tokenizer, "split_text") else target_text.split()

        for word in words:
            if not word:
                continue
            if word not in self.encoder.tokenizer.vocab:
                new_id = len(self.encoder.tokenizer.vocab)
                self.encoder.tokenizer.vocab[word] = new_id
                self.encoder.token_sdr_map[new_id] = self._generate_stable_sdr(
                    word)

    def _text_to_ids(self, text: str) -> List[int]:
        if not hasattr(self.encoder, "tokenizer"):
            return []
        target_text = text.split(":", 1)[1] if ":" in text else text
        words = self.encoder.tokenizer.split_text(target_text) if hasattr(
            self.encoder.tokenizer, "split_text") else target_text.split()
        return [self.encoder.tokenizer.vocab[w] for w in words if w in self.encoder.tokenizer.vocab]

    def _ids_to_text(self, ids: List[int]) -> str:
        if not hasattr(self.encoder, "tokenizer"):
            return ""
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
            if not sdr:
                continue
            sample_count = int(len(sdr) * decay_rate)
            if sample_count > 0:
                combined_sdr.update(random.sample(
                    sdr, min(sample_count, len(sdr))))
            decay_rate *= 0.6
            if decay_rate < 0.2:
                break
        target_on_bits = int(self.encoder.input_size * 0.05)
        sdr_list = sorted(list(combined_sdr))
        if len(sdr_list) > target_on_bits:
            return sorted(random.sample(sdr_list, target_on_bits))
        return sdr_list

    def _determine_response_mode(self, text: str) -> str:
        if "とは" in text or "意味" in text or "教えて" in text:
            return "definition"
        elif "続き" in text or "その先" in text:
            return "continuation"
        elif "要約" in text or "まとめて" in text:
            return "summarize"
        elif "?" in text or "？" in text:
            return "question"
        return "general"

    def _score_candidate(self, text: str, query: str) -> float:
        score = 10.0
        if text.count("「") > text.count("」") or text.count("（") > text.count("）"):
            score -= 5.0
        if not text.endswith(("。", "！", "？", "!", "?")):
            score -= 3.0
        if re.search(r"(.{5,})\1", text):
            score -= 4.0
        if len(text) < 5:
            score -= 6.0
            
        query_words = set(query.split())
        for word in query_words:
            if len(word) > 1 and word in text:
                score += 2.0
        return score

    def _complete_sentence(self, text: str) -> str:
        if text.count("「") > text.count("」"):
            text += "」"
        if text.count("（") > text.count("）"):
            text += "）"
        if not text.endswith(("。", "！", "？", "!", "?")):
            text += "。"
        return text

    def _extract_keywords(self, text: str) -> List[str]:
        stop_terms = {
            "それ", "これ", "あれ", "その", "この", "あの", "こと", "もの", "何", "なに",
            "です", "ます", "した", "して", "いる", "ある", "なる", "ですか", "ますか",
            "メリット", "デメリット",
        }
        extracted: List[str] = []

        if hasattr(self.encoder, "tokenizer") and hasattr(self.encoder.tokenizer, "pre_tokenize"):
            for token in self.encoder.tokenizer.pre_tokenize(text):
                cleaned = token.strip().lower()
                if len(cleaned) >= 2 and cleaned not in stop_terms and re.search(r"[a-zA-Z0-9一-龥ぁ-んァ-ヴー]", cleaned):
                    extracted.append(cleaned)

        extracted.extend(w.lower() for w in re.findall(r"[a-zA-Z0-9]{2,}", text))
        extracted.extend(w for w in re.findall(r"[一-龥]{2,}|[ァ-ヴー]{2,}|[ぁ-ん]{2,}", text) if w not in stop_terms)

        seen = []
        for item in extracted:
            if item not in stop_terms and item not in seen:
                seen.append(item)
        return seen

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        self._register_dynamic_vocab(user_text)
        user_ids = self._text_to_ids(user_text)
        
        mode = self._determine_response_mode(user_text)
        self.dialogue_state["last_intent"] = mode
        extracted_now = self._extract_keywords(user_text)
        self.topic_tracker.update(extracted_now)

        if user_ids and teaching_mode:
            self.episodic_snn.process_sequence(user_ids, is_training=True)
            self.llm.learn_sequence(user_ids)

        original_vsa = self.encoder.apply_vsa
        self.encoder.apply_vsa = False
        pfc_input_sdr = self.encoder.encode(user_text)
        input_sdr = self.encoder.encode(user_text)
        self.encoder.apply_vsa = original_vsa

        if not teaching_mode:
            self._update_history("user", user_text, input_sdr)
            
        history_context_sdr = self._get_history_context_sdr()

        routing_sdr = set(pfc_input_sdr)
        if history_context_sdr:
            routing_sdr.update(random.sample(
                history_context_sdr, int(len(history_context_sdr) * 0.1)))

        overlaps = {comp: len(routing_sdr.intersection(set(anchor))) + (2 if comp != "general" else 0)
                    for comp, anchor in self.prefrontal.context_anchors.items()}
        sorted_experts = sorted(
            overlaps.items(), key=lambda x: x[1], reverse=True)
        active_experts = [comp for comp,
                          score in sorted_experts[:2] if score > 0]
        if not active_experts or sorted_experts[0][1] < 5:
            active_experts = ["general"]

        self.dialogue_state["current_topic"] = active_experts[0]

        # 💡 文脈キーワードの抽出（過去数ターンのユーザー発言から）
        context_keywords = set()
        prev_user = ""
        if len(self.dialogue_history) > 0:
            for msg in reversed(self.dialogue_history[-6:]):
                if msg["role"] == "user":
                    if not prev_user:
                        prev_user = msg["text"]
                    context_keywords.update(self._extract_keywords(msg["text"]))
        context_keywords.update(self.topic_tracker.active_terms(limit=6))

        if teaching_mode:
            context = active_experts[0]
            if ":" in user_text:
                parts = user_text.split(":", 1)
                if parts[0].strip() in self.prefrontal.compartments:
                    context = parts[0].strip()
                    user_text = parts[1].strip()

            if context in self.prefrontal.context_anchors:
                anchor_set = set(self.prefrontal.context_anchors[context])
                sampled_bits = random.sample(pfc_input_sdr, min(
                    max(1, int(len(pfc_input_sdr) * 0.4)), len(pfc_input_sdr)))
                anchor_set.update(sampled_bits)
                target_bits = int(self.encoder.input_size *
                                  self.encoder.density)
                self.prefrontal.context_anchors[context] = set(random.sample(
                    list(anchor_set), target_bits) if len(anchor_set) > target_bits else anchor_set)

            # 💡 【重要】学習時に、現在の文脈キーワードを記憶テキストに埋め込んで一緒に保存する
            kw_prefix = f"【文脈: {' '.join(context_keywords)}】 " if context_keywords else ""
            content_to_store = f"{kw_prefix}{user_text}"

            self.brain.experience_and_memorize(
                input_sdr, content=content_to_store, context=context, learning=True)
            response_text = f"[MoE Router: {context} Expert] 海馬とSpikingLLMに記憶を定着させました。"
            return response_text

        pre_tool_results = []
        for trigger_spike, tool_func in self.tools.items():
            if trigger_spike in user_text:
                try:
                    res = tool_func(user_text)
                    if res and res != "計算できませんでした":
                        pre_tool_results.append(res)
                except Exception:
                    pass
                    
        tool_triggered_by_user = len(pre_tool_results) > 0

        recent_context = f"前回の質問「{prev_user[:30]}」を踏まえ、" if prev_user else ""

        current_keywords = set(extracted_now)
        
        # 指示語（代名詞）が含まれているかをチェック
        has_demonstrative = any(d in user_text for d in ["それ", "これ", "あれ", "その", "この", "あの", "彼", "彼女"])
        if has_demonstrative or len(current_keywords) <= 1:
            current_keywords.update(context_keywords)

        associated_text = ""
        associated_sdr: List[int] = []
        if user_ids:
            associated_ids = self.episodic_snn.process_sequence(
                user_ids, is_training=False)
            associated_text = self._ids_to_text(associated_ids)
            if associated_text:
                self.encoder.apply_vsa = False
                associated_sdr = self.encoder.encode(associated_text)
                self.encoder.apply_vsa = True

        search_sdr = set(input_sdr)
        if associated_sdr:
            search_sdr.update(associated_sdr)
            
        if prev_user:
            self.encoder.apply_vsa = False
            prev_sdr = self.encoder.encode(prev_user)
            self.encoder.apply_vsa = True
            search_sdr.update(random.sample(prev_sdr, int(len(prev_sdr) * 0.4)))

        all_retrieved: List[Dict[str, Any]] = []
        for comp in self.prefrontal.compartments:
            all_retrieved.extend(self.brain.in_context_inference(
                current_sensory_sdr=list(search_sdr), context=comp))

        valid_retrievals = [m for m in all_retrieved if m.get("score", 0) > 0.05]
        filtered_retrievals = []
        
        for m in valid_retrievals:
            content = m["content"]
            content_lower = content.lower()
            
            curr_match = sum(1 for kw in current_keywords if kw.lower() in content_lower)
            ctx_match = sum(1 for kw in context_keywords if kw.lower() in content_lower)

            # 現在語が明確ならそれを優先し、指示語・短文時は文脈概念で補完
            if current_keywords and not has_demonstrative:
                req_curr = 1 if len(current_keywords) <= 2 else max(1, math.ceil(len(current_keywords) * 0.5))
                if curr_match < req_curr:
                    continue
            
            # 2. 💡 指示語が使われている場合は、過去の文脈（主語）が含まれているかを確認する
            if (has_demonstrative or not current_keywords) and context_keywords:
                if ctx_match == 0:
                    continue  # 文脈キーワードが1つも含まれていない場合は無関係な記憶として除外

            all_kw = set(current_keywords) | context_keywords
            total_match = sum(1 for kw in all_kw if kw.lower() in content_lower)
            m["keyword_score"] = total_match
                
            match_qa = re.search(r'回答は「(.*?)」', content)
            if match_qa:
                m["clean_content"] = match_qa.group(1)
            else:
                m["clean_content"] = content
                
            filtered_retrievals.append(m)

        filtered_retrievals.sort(key=lambda x: (x.get("keyword_score", 0), x.get("score", 0)), reverse=True)
        valid_retrievals = filtered_retrievals

        if tool_triggered_by_user:
            memory_context = " ".join(pre_tool_results)
            blended_memory = "[外部ツール実行結果による事実情報]"
        elif valid_retrievals:
            best_memories = [m["clean_content"] for m in valid_retrievals[:2]]
            blended_memory = " | ".join(best_memories)
            memory_context = best_memories[0]
            if len(memory_context) > 150:
                 memory_context = memory_context[:150] + "..."
        else:
            blended_memory = "なし"
            memory_context = associated_text[:100] if associated_text else ""

        if not memory_context.strip() and not tool_triggered_by_user:
            active_terms = self.topic_tracker.active_terms(limit=3)
            if active_terms:
                hint = " / ".join(active_terms)
                fallback_msg = f"関連知識は十分に取り出せませんでした。現在の話題候補は「{hint}」です。主語を補って聞き直してください。"
            else:
                fallback_msg = "申し訳ありませんが、その質問に関連する十分な知識が記憶に見つかりません。別の表現で試していただくか、関連する知識を教えていただけますか？"
            self._update_history("system", fallback_msg, self.encoder.encode(fallback_msg))
            return f"[MoE Router: {active_experts[0]} (Fallback)]\n >> {fallback_msg}"

        if tool_triggered_by_user:
            final_generated_text = f"【システム情報】 {pre_tool_results[0]}"
            full_response = f"[MoE Router: {', '.join(active_experts)} (Tool Action)]\n"
            full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
            full_response += f" >> SARA回答: {final_generated_text}"
            self._update_history("system", final_generated_text, self.encoder.encode(final_generated_text))
            return full_response

        num_candidates = 3
        best_candidate = ""
        best_score = -999.0

        for attempt in range(num_candidates):
            prompt_for_llm = f"{recent_context}{memory_context} {user_text}"
            if mode == "definition":
                prompt_for_llm += " つまり、"
            elif mode == "summarize":
                prompt_for_llm += " 要約すると、"
            elif mode == "question":
                prompt_for_llm += " 答えは、"

            current_prompt_ids = self._text_to_ids(prompt_for_llm)
            generated_tokens: List[int] = []

            max_agent_steps = 30
            for step in range(max_agent_steps):
                new_ids = self.llm.generate(
                    prompt_tokens=current_prompt_ids, max_new_tokens=1, temperature=0.5)
                if not new_ids:
                    break

                token_id = new_ids[0]
                generated_tokens.append(token_id)
                current_prompt_ids.append(token_id)

                current_full_str = prompt_for_llm + self._ids_to_text(generated_tokens)
                
                if len(generated_tokens) > 5 and current_full_str.strip().endswith(("。", "？", "！", "!", "?")):
                    break
                    
            candidate_text = self._ids_to_text(generated_tokens)
            candidate_text = self._complete_sentence(candidate_text)
            
            score = self._score_candidate(candidate_text, user_text)
            if score > best_score:
                best_score = score
                best_candidate = candidate_text

        if len(best_candidate) < 5 and memory_context:
            final_generated_text = memory_context
        else:
            final_generated_text = memory_context + "\n" + best_candidate

        full_response = f"[MoE Router: {', '.join(active_experts)} 活性化 (Mode: {mode})]\n"
        full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
        full_response += f" >> SNN-LLM自律生成: {final_generated_text}"

        self._update_history("system", final_generated_text,
                             self.encoder.encode(final_generated_text))
        return full_response

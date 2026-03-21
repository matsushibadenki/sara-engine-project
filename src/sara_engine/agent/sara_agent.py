# ディレクトリパス: src/sara_engine/agent/sara_agent.py
# ファイルの日本語タイトル: 統合マルチモーダル・エージェント (SpikingLLM Pipeline統合版)
# ファイルの目的や内容: 検索時のキーワード判定を「メタデータを含む全体」と「クリーンな本文」で分離し、代名詞（指示語）を用いたフォローアップ質問に対する文脈解決能力を回復。さらにPhase 4のTextGenerationPipelineとストリーミング生成を統合した最終版。

from ..memory.million_token_snn import DynamicSNNMemory
from ..encoders.audio import AudioSpikeEncoder
from ..encoders.vision import ImageSpikeEncoder
from ..models.spiking_llm import SpikingLLM
from ..core.prefrontal import PrefrontalCortex
from ..memory.hippocampus import CorticoHippocampalSystem
from ..core.cortex import CorticalColumn
from ..memory.sdr import SDREncoder
from ..utils.dialogue import AdaptiveTopicTracker
from ..safety.safety_guard import SafetyGuard, SafetyCheckResult
from ..pipelines.text_generation import pipeline  # 追加: パイプライン
from typing import List, Dict, Any, Callable, Union, Generator, Optional
import hashlib
import random
import os
import re
import pickle
from ..utils.project_paths import ensure_output_directory, ensure_parent_directory, model_path, workspace_path


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
        system_prompt: Optional[str] = None,
        safety_guard: Optional[SafetyGuard] = None,
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
            ltm_filepath=model_path("sara_multimodal_ltm.pkl"),
            max_working_memory_size=15,
        )

        self.llm = SpikingLLM(num_layers=2, sdr_size=128,
                              vocab_size=100000, enable_learning=True)
                              
        # 追加: TextGenerationPipelineの初期化
        self.generator = pipeline("text-generation", model=self.llm, tokenizer=self.encoder.tokenizer)

        self.vision = ImageSpikeEncoder(output_size=input_size)
        self.audio = AudioSpikeEncoder(output_size=input_size)
        self.episodic_snn = DynamicSNNMemory(vocab_size=100000, sdr_size=3)

        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history_turns = 5
        self.dialogue_state: Dict[str, Any] = {
            "current_topic": None, "last_intent": None}
        self.topic_tracker = AdaptiveTopicTracker()

        self.tools: Dict[str, Callable[[str], str]] = {}
        self.runtime_issues: List[Dict[str, str]] = []
        self.retrieval_diagnostics: List[Dict[str, Any]] = []
        self.system_prompt = system_prompt or ""
        self.safety_guard = safety_guard
        self._bootstrap()

    def save_agent(self, save_dir: str = model_path("sara_agent")) -> None:
        save_dir = ensure_output_directory(save_dir)
        if hasattr(self.llm, "save_pretrained"):
            try:
                self.llm.save_pretrained(save_dir)
            except Exception as e:
                self._record_issue("save_agent", f"Failed to save LLM weights: {e}")
        try:
            with open(os.path.join(save_dir, "episodic_snn.pkl"), "wb") as f:
                pickle.dump(self.episodic_snn, f)
            with open(os.path.join(save_dir, "brain_ltm.pkl"), "wb") as f:
                pickle.dump(self.brain, f)
        except Exception as e:
            self._record_issue("save_agent", f"Failed to save memory state: {e}")

    def load_agent(self, load_dir: str = model_path("sara_agent")) -> None:
        if not os.path.exists(load_dir):
            return
        if hasattr(self.llm, "load_pretrained"):
            try:
                from ..models.spiking_llm import SpikingLLM
                self.llm = SpikingLLM.from_pretrained(load_dir)
                # LLM再ロード後にパイプラインも更新
                self.generator = pipeline("text-generation", model=self.llm, tokenizer=self.encoder.tokenizer)
            except Exception as e:
                self._record_issue("load_agent", f"Failed to load LLM weights: {e}")
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
            self._record_issue("load_agent", f"Failed to load memory state: {e}")

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
        bootstrap_vocab_path = workspace_path("tokenizers", "sara_vocab.json")
        if not os.path.exists(bootstrap_vocab_path):
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

    def save_session(self, session_path: str) -> None:
        session_path = ensure_parent_directory(session_path)
        payload = {
            "dialogue_history": self.dialogue_history,
            "dialogue_state": self.dialogue_state,
            "max_history_turns": self.max_history_turns,
            "system_prompt": self.system_prompt,
            "topic_tracker": self.topic_tracker.to_dict(),
            "runtime_issues": list(self.runtime_issues),
            "retrieval_diagnostics": list(self.retrieval_diagnostics),
        }
        with open(session_path, "wb") as handle:
            pickle.dump(payload, handle)

    def load_session(self, session_path: str) -> None:
        if not os.path.exists(session_path):
            return
        with open(session_path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            self.dialogue_history = payload.get("dialogue_history", [])
            self.dialogue_state = payload.get("dialogue_state", {})
            self.max_history_turns = int(payload.get("max_history_turns", self.max_history_turns))
            self.system_prompt = payload.get("system_prompt", self.system_prompt)
            topic_state = payload.get("topic_tracker")
            if isinstance(topic_state, dict):
                self.topic_tracker.load_state(topic_state)
            runtime_issues = payload.get("runtime_issues", [])
            if isinstance(runtime_issues, list):
                self.runtime_issues = [
                    {
                        "stage": str(item.get("stage", "")),
                        "message": str(item.get("message", "")),
                    }
                    for item in runtime_issues
                    if isinstance(item, dict)
                ][-20:]
            retrieval_diagnostics = payload.get("retrieval_diagnostics", [])
            if isinstance(retrieval_diagnostics, list):
                self.retrieval_diagnostics = [
                    item for item in retrieval_diagnostics
                    if isinstance(item, dict)
                ][-10:]

    def get_recent_issues(self, limit: int = 5) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        return self.runtime_issues[-limit:]

    def get_recent_retrieval_diagnostics(self, limit: int = 3) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return self.retrieval_diagnostics[-limit:]

    def clear_runtime_issues(self) -> None:
        self.runtime_issues.clear()
        self.retrieval_diagnostics.clear()

    def format_recent_issues(self, limit: int = 5) -> str:
        issues = self.get_recent_issues(limit=limit)
        if not issues:
            return "No runtime issues recorded."
        lines = ["Recent runtime issues:"]
        for issue in issues:
            lines.append(f"- [{issue['stage']}] {issue['message']}")
        return "\n".join(lines)

    def format_recent_retrieval_diagnostics(self, limit: int = 3) -> str:
        diagnostics = self.get_recent_retrieval_diagnostics(limit=limit)
        if not diagnostics:
            return "No retrieval diagnostics recorded."

        lines = ["Recent retrieval diagnostics:"]
        for item in diagnostics:
            lines.append(
                "- "
                f"{item.get('clean_content', '')[:60]} | "
                f"total={item.get('keyword_score', 0.0):.2f} "
                f"current={item.get('current_keyword_coverage', 0.0):.2f} "
                f"context={item.get('context_keyword_coverage', 0.0):.2f} "
                f"metadata={item.get('metadata_keyword_coverage', 0.0):.2f}"
            )
        return "\n".join(lines)

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
            "は", "が", "の", "に", "を", "と", "や"
        }
        extracted: List[str] = []

        if hasattr(self.encoder, "tokenizer") and hasattr(self.encoder.tokenizer, "pre_tokenize"):
            for token in self.encoder.tokenizer.pre_tokenize(text):
                cleaned = token.strip().lower()
                if len(cleaned) >= 2 and cleaned not in stop_terms and re.search(r"[a-zA-Z0-9一-龥ぁ-んァ-ヴー]", cleaned):
                    extracted.append(cleaned)

        extracted.extend(w.lower()
                         for w in re.findall(r"[a-zA-Z0-9]{2,}", text))
        extracted.extend(w for w in re.findall(
            r"[一-龥]{2,}|[ァ-ヴー]{2,}|[ぁ-ん]{2,}", text) if w not in stop_terms)

        seen = []
        for item in extracted:
            if item not in stop_terms and item not in seen:
                seen.append(item)
        return seen

    def _keyword_overlap_stats(self, query_keywords: set[str], candidate_text: str) -> Dict[str, float]:
        if not query_keywords:
            return {
                "match_count": 0.0,
                "coverage": 0.0,
                "density": 0.0,
            }

        candidate_lower = candidate_text.lower()
        match_count = sum(1 for kw in query_keywords if kw.lower() in candidate_lower)
        coverage = match_count / max(1, len(query_keywords))
        density = match_count / max(1, len(candidate_text.split()))
        return {
            "match_count": float(match_count),
            "coverage": coverage,
            "density": density,
        }

    def _prepare_teaching_memory(
        self,
        user_text: str,
        context: str,
        context_keywords: set[str],
        current_keywords: set[str],
    ) -> tuple[str, Dict[str, Any]]:
        selected_keywords: List[str] = []
        for keyword in list(current_keywords) + list(context_keywords):
            normalized = keyword.strip().lower()
            if not normalized or normalized in selected_keywords:
                continue
            selected_keywords.append(normalized)
            if len(selected_keywords) >= 6:
                break

        metadata = {
            "context": context,
            "keywords": selected_keywords,
        }
        keyword_prefix = " ".join(selected_keywords)
        content_to_store = f"【文脈: {keyword_prefix}】 {user_text}" if keyword_prefix else user_text
        return content_to_store, metadata

    def _score_retrieval_candidate(
        self,
        candidate: Dict[str, Any],
        current_keywords: set[str],
        context_keywords: set[str],
        has_demonstrative: bool,
    ) -> Optional[Dict[str, Any]]:
        content = str(candidate.get("content", ""))
        content_lower = content.lower()

        clean_content = content
        match_ctx = re.search(r'【文脈:.*?】\s*(.*)', content)
        if match_ctx:
            clean_content = match_ctx.group(1)

        match_qa = re.search(r'回答は「(.*?)」', clean_content)
        if match_qa:
            clean_content = match_qa.group(1)

        clean_content = clean_content.strip()
        clean_lower = clean_content.lower()
        metadata = candidate.get("metadata", {})
        metadata_keywords = set()
        if isinstance(metadata, dict):
            raw_keywords = metadata.get("keywords", [])
            if isinstance(raw_keywords, list):
                metadata_keywords = {
                    str(keyword).strip().lower()
                    for keyword in raw_keywords
                    if str(keyword).strip()
                }

        current_stats = self._keyword_overlap_stats(current_keywords, clean_lower)
        context_stats = self._keyword_overlap_stats(context_keywords, content_lower)
        all_keywords = set(current_keywords) | set(context_keywords)
        total_stats = self._keyword_overlap_stats(all_keywords, content_lower)
        metadata_stats = self._keyword_overlap_stats(all_keywords, " ".join(sorted(metadata_keywords)))

        if current_keywords and not has_demonstrative and current_stats["match_count"] <= 0:
            return None
        if (has_demonstrative or not current_keywords) and context_keywords and context_stats["match_count"] <= 0:
            return None

        retrieval_score = float(candidate.get("score", 0.0))
        candidate["clean_content"] = clean_content
        candidate["keyword_score"] = (
            current_stats["match_count"] * 3.0
            + current_stats["coverage"] * 4.0
            + context_stats["match_count"] * 1.5
            + context_stats["coverage"] * 2.0
            + metadata_stats["match_count"] * 2.0
            + metadata_stats["coverage"] * 3.0
            + total_stats["density"] * 2.5
            + retrieval_score * 5.0
        )
        candidate["match_count"] = total_stats["match_count"]
        candidate["current_keyword_coverage"] = current_stats["coverage"]
        candidate["context_keyword_coverage"] = context_stats["coverage"]
        candidate["metadata_keyword_coverage"] = metadata_stats["coverage"]
        candidate["retrieval_score_base"] = retrieval_score
        return candidate

    def _capture_retrieval_diagnostics(self, retrievals: List[Dict[str, Any]], limit: int = 3) -> None:
        captured: List[Dict[str, Any]] = []
        for item in retrievals[:limit]:
            captured.append(
                {
                    "clean_content": str(item.get("clean_content", "")),
                    "keyword_score": float(item.get("keyword_score", 0.0)),
                    "current_keyword_coverage": float(item.get("current_keyword_coverage", 0.0)),
                    "context_keyword_coverage": float(item.get("context_keyword_coverage", 0.0)),
                    "metadata_keyword_coverage": float(item.get("metadata_keyword_coverage", 0.0)),
                    "retrieval_score_base": float(item.get("retrieval_score_base", 0.0)),
                }
            )
        self.retrieval_diagnostics = captured[-10:]

    def _record_issue(self, stage: str, message: str) -> None:
        issue = {"stage": stage, "message": message}
        self.runtime_issues.append(issue)
        if len(self.runtime_issues) > 20:
            self.runtime_issues.pop(0)

    def _score_expert_routing(
        self,
        routing_sdr: set[int],
        current_keywords: set[str],
        context_keywords: set[str],
    ) -> List[tuple[str, float]]:
        scored_experts: List[tuple[str, float]] = []
        combined_keywords = {keyword.lower() for keyword in (current_keywords | context_keywords) if keyword}

        for comp, anchor in self.prefrontal.context_anchors.items():
            anchor_overlap = float(len(routing_sdr.intersection(set(anchor))))
            lexical_hits = 0.0

            comp_terms = {
                token.lower()
                for token in comp.replace("_", " ").split()
                if token
            }
            lexical_hits += sum(1.0 for token in comp_terms if token in combined_keywords)

            for keyword in combined_keywords:
                if keyword in comp.lower():
                    lexical_hits += 1.5

            topic_bonus = 0.0
            if comp == self.dialogue_state.get("current_topic"):
                topic_bonus += 1.0

            if comp != "general":
                anchor_overlap += 2.0 if anchor_overlap > 0 else 0.0
                if anchor_overlap > 0 or lexical_hits > 0.0:
                    topic_bonus += 0.5
            else:
                lexical_hits *= 0.5

            score = anchor_overlap + lexical_hits + topic_bonus
            scored_experts.append((comp, score))

        scored_experts.sort(key=lambda item: item[1], reverse=True)
        return scored_experts

    def _blend_retrieval_memories(
        self,
        retrievals: List[Dict[str, Any]],
        current_keywords: set[str],
        context_keywords: set[str],
    ) -> tuple[str, str]:
        if not retrievals:
            return "", ""

        deduped: List[Dict[str, Any]] = []
        seen_contents: set[str] = set()
        for item in retrievals:
            clean_content = str(item.get("clean_content", "")).strip()
            if not clean_content or clean_content in seen_contents:
                continue
            seen_contents.add(clean_content)
            deduped.append(item)

        if not deduped:
            return "", ""

        selected = [deduped[0]["clean_content"]]
        query_terms = {term.lower() for term in (current_keywords | context_keywords) if term}
        primary_terms = set(self._extract_keywords(deduped[0]["clean_content"]))
        support_terms = primary_terms | query_terms

        for item in deduped[1:4]:
            clean_content = item["clean_content"]
            candidate_terms = set(self._extract_keywords(clean_content))
            if not candidate_terms:
                continue

            overlap = len(candidate_terms.intersection(support_terms))
            coverage = overlap / max(1, len(candidate_terms))
            if overlap <= 0 or coverage < 0.2:
                continue

            selected.append(clean_content)
            support_terms.update(candidate_terms)
            if len(selected) >= 2:
                break

        blended_memory = " | ".join(selected)
        memory_context = selected[0]
        if len(selected) > 1:
            memory_context = f"{selected[0]} 補足: {selected[1]}"
        if len(memory_context) > 150:
            memory_context = memory_context[:150] + "..."
        return memory_context, blended_memory

    def _build_topic_aware_fallback(
        self,
        user_text: str,
        current_keywords: set[str],
        context_keywords: set[str],
        active_experts: List[str],
        mode: str,
        has_demonstrative: bool,
    ) -> str:
        if len(user_text) <= 10 and mode == "general":
            return "はい、SARAです。何かお手伝いできることはありますか？"

        active_terms = self.topic_tracker.active_terms(limit=3)
        topic_hint = active_terms[0] if active_terms else active_experts[0]
        query_terms = [term for term in list(current_keywords)[:2] if term]
        context_terms = [term for term in list(context_keywords)[:2] if term]

        if has_demonstrative and context_terms:
            detail_hint = " / ".join(context_terms)
            return (
                f"関連知識は十分に取り出せませんでした。現在の話題候補は「{topic_hint}」です。"
                f"「{detail_hint}」のどの点を知りたいか、主語を補って聞き直してください。"
            )

        if query_terms:
            detail_hint = " / ".join(query_terms)
            return (
                f"関連知識は十分に取り出せませんでした。現在の話題候補は「{topic_hint}」です。"
                f"「{detail_hint}」について、対象や条件をもう少し具体的にしてください。"
            )

        if active_terms:
            detail_hint = " / ".join(active_terms)
            return (
                f"関連知識は十分に取り出せませんでした。現在の話題候補は「{detail_hint}」です。"
                "主語や対象を補って聞き直してください。"
            )

        return "申し訳ありませんが、その質問に関連する十分な知識が記憶に見つかりません。別の表現で試していただくか、関連する知識を教えていただけますか？"

    def _execute_tools(self, user_text: str) -> tuple[List[str], List[str]]:
        tool_results: List[str] = []
        tool_failures: List[str] = []
        for trigger_spike, tool_func in self.tools.items():
            if trigger_spike not in user_text:
                continue
            try:
                res = tool_func(user_text)
            except Exception as exc:
                failure = f"{trigger_spike}: {exc}"
                tool_failures.append(failure)
                self._record_issue("tool_execution", failure)
                continue
            if res and res != "計算できませんでした":
                tool_results.append(res)
        return tool_results, tool_failures

    def chat(self, user_text: str, teaching_mode: bool = False, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        safety_input: Optional[SafetyCheckResult] = None
        if self.safety_guard is not None:
            safety_input = self.safety_guard.check_input(user_text)
            if not safety_input.is_safe:
                return "Input was rejected by safety guard."
            user_text = safety_input.sanitized_text
        self._register_dynamic_vocab(user_text)
        user_ids = self._text_to_ids(user_text)

        mode = self._determine_response_mode(user_text)
        self.dialogue_state["last_intent"] = mode
        extracted_now = self._extract_keywords(user_text)
        current_keywords = set(extracted_now)
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
        else:
            self._update_history("user", f"[学習] {user_text}", input_sdr)

        history_context_sdr = self._get_history_context_sdr()

        routing_sdr = set(pfc_input_sdr)
        if history_context_sdr:
            routing_sdr.update(random.sample(
                history_context_sdr, int(len(history_context_sdr) * 0.1)))

        context_keywords = set()
        prev_user = ""
        if len(self.dialogue_history) > 0:
            for msg in reversed(self.dialogue_history[-6:]):
                if msg["role"] == "user":
                    if not prev_user:
                        prev_user = msg["text"]
                    context_keywords.update(
                        self._extract_keywords(msg["text"]))
        context_keywords.update(self.topic_tracker.active_terms(limit=6))
        sorted_experts = self._score_expert_routing(
            routing_sdr=routing_sdr,
            current_keywords=current_keywords,
            context_keywords=context_keywords,
        )
        active_experts = [comp for comp, score in sorted_experts[:2] if score > 0]
        if not active_experts:
            active_experts = ["general"]

        self.dialogue_state["current_topic"] = active_experts[0]

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

            content_to_store, memory_metadata = self._prepare_teaching_memory(
                user_text=user_text,
                context=context,
                context_keywords=context_keywords,
                current_keywords=current_keywords,
            )

            self.brain.experience_and_memorize(
                input_sdr,
                content=content_to_store,
                context=context,
                learning=True,
                metadata=memory_metadata,
            )
            response_text = f"[MoE Router: {context} Expert] 海馬とSpikingLLMに記憶を定着させました。"
            return response_text

        pre_tool_results, tool_failures = self._execute_tools(user_text)

        tool_triggered_by_user = len(pre_tool_results) > 0

        recent_context = f"前回の質問「{prev_user[:30]}」を踏まえ、" if prev_user else ""
        has_demonstrative = any(d in user_text for d in [
                                "それ", "これ", "あれ", "その", "この", "あの", "彼", "彼女"])
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
            search_sdr.update(random.sample(
                prev_sdr, int(len(prev_sdr) * 0.4)))

        all_retrieved: List[Dict[str, Any]] = []
        for comp in self.prefrontal.compartments:
            all_retrieved.extend(self.brain.in_context_inference(
                current_sensory_sdr=list(search_sdr), context=comp))

        valid_retrievals = [
            m for m in all_retrieved if m.get("score", 0) > 0.01]
        filtered_retrievals = []
        for m in valid_retrievals:
            scored = self._score_retrieval_candidate(
                m,
                current_keywords=current_keywords,
                context_keywords=context_keywords,
                has_demonstrative=has_demonstrative,
            )
            if scored is not None:
                filtered_retrievals.append(scored)

        filtered_retrievals.sort(key=lambda x: (
            x.get("keyword_score", 0.0),
            x.get("current_keyword_coverage", 0.0),
            x.get("context_keyword_coverage", 0.0),
            x.get("metadata_keyword_coverage", 0.0),
            x.get("score", 0.0),
        ), reverse=True)
        valid_retrievals = filtered_retrievals
        self._capture_retrieval_diagnostics(valid_retrievals)

        if not valid_retrievals and not tool_triggered_by_user:
            fallback_msg = self._build_topic_aware_fallback(
                user_text=user_text,
                current_keywords=current_keywords,
                context_keywords=context_keywords,
                active_experts=active_experts,
                mode=mode,
                has_demonstrative=has_demonstrative,
            )
            self._update_history("system", fallback_msg,
                                 self.encoder.encode(fallback_msg))
            router_suffix = "" if len(user_text) <= 10 and mode == "general" else " (Fallback)"
            return f"[MoE Router: {active_experts[0]}{router_suffix}]\n >> {fallback_msg}"

        if tool_triggered_by_user:
            memory_context = " ".join(pre_tool_results)
            blended_memory = "[外部ツール実行結果による事実情報]"
        else:
            memory_context, blended_memory = self._blend_retrieval_memories(
                valid_retrievals,
                current_keywords=current_keywords,
                context_keywords=context_keywords,
            )

        if tool_triggered_by_user:
            final_generated_text = f"【システム情報】 {pre_tool_results[0]}"
            full_response = f"[MoE Router: {', '.join(active_experts)} (Tool Action)]\n"
            full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
            full_response += f" >> SARA回答: {final_generated_text}"
            if tool_failures:
                full_response += f"\n >> Tool warnings: {' | '.join(tool_failures[:2])}"
            self._update_history("system", final_generated_text,
                                 self.encoder.encode(final_generated_text))
            return full_response

        # プロンプトの構築
        prompt_for_llm = f"{recent_context}{memory_context} {user_text}"
        if self.system_prompt:
            prompt_for_llm = f"{self.system_prompt}\n{prompt_for_llm}"
        if mode == "definition":
            prompt_for_llm += " つまり、"
        elif mode == "summarize":
            prompt_for_llm += " 要約すると、"
        elif mode == "question":
            prompt_for_llm += " 答えは、"

        # =============== Phase 4: TextGenerationPipeline の利用 ===============
        stop_conditions = ["。", "？", "！", "!", "?"]

        if stream:
            def response_generator() -> Generator[str, None, None]:
                # ヘッダ情報と記憶情報を最初に送る
                yield f"[MoE Router: {', '.join(active_experts)} 活性化 (Mode: {mode})]\n"
                yield f" >> 海馬ブレンド記憶: {blended_memory}\n"
                yield f" >> SARA回答: {memory_context}\n"
                yield "[SNN生成による追記] "

                generated_full = ""
                # パイプラインストリーミングによる逐次生成
                for chunk in self.generator.stream(
                    prompt_for_llm, 
                    max_new_tokens=30, 
                    temperature=0.5, 
                    stop_conditions=stop_conditions
                ):
                    text_val = chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk)
                    generated_full += text_val
                    yield text_val

                final_text = memory_context + "\n[SNN生成による追記] " + generated_full
                if self.safety_guard is not None:
                    safety_output = self.safety_guard.check_output(final_text)
                    if not safety_output.is_safe:
                        yield "\n[Output blocked by safety guard]"
                        return
                self._update_history("system", final_text, self.encoder.encode(final_text))
            
            return response_generator()
        else:
            # 非ストリーミング時は、複数の候補からスコアの一番高いものを選択する (ノイズ除去ロジックを継承)
            num_candidates = 3
            best_candidate = ""
            best_score = -999.0

            for attempt in range(num_candidates):
                # パイプライン経由でのテキスト生成
                gen_result = self.generator(
                    prompt_for_llm,
                    max_new_tokens=30,
                    temperature=0.5 + (attempt * 0.1),  # 試行ごとに多様性を持たせる
                    stop_conditions=stop_conditions
                )

                # 生成されたテキストからプロンプト部分を削除する
                if isinstance(gen_result, str):
                    if gen_result.startswith(prompt_for_llm):
                        candidate_text = gen_result[len(prompt_for_llm):]
                    else:
                        candidate_text = gen_result.replace(prompt_for_llm, "")
                else:
                    candidate_text = ""

                candidate_text = self._complete_sentence(candidate_text)
                score = self._score_candidate(candidate_text, user_text)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate_text

            if len(best_candidate) < 10 or best_score < 15.0:
                final_generated_text = memory_context
            else:
                final_generated_text = memory_context + "\n[SNN生成による追記] " + best_candidate

            full_response = f"[MoE Router: {', '.join(active_experts)} 活性化 (Mode: {mode})]\n"
            full_response += f" >> 海馬ブレンド記憶: {blended_memory}\n"
            full_response += f" >> SARA回答: {final_generated_text}"

            self._update_history("system", final_generated_text, self.encoder.encode(final_generated_text))
            if self.safety_guard is not None:
                safety_output = self.safety_guard.check_output(full_response)
                if not safety_output.is_safe:
                    return "Output was blocked by safety guard."
            return full_response

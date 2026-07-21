# ディレクトリパス: src/sara_engine/inference.py
# ファイルの日本語タイトル: SARA汎用推論エンジンクラス
# ファイルの目的や内容: SNNベースの汎用推論エンジン。空の脳状態を防ぐため、オンライン学習(Hebbian学習)とモデル保存機能を追加実装。
import msgpack
import os
import random
import math
import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .utils.direct_map import restore_direct_map, serialize_direct_map
from .utils.tokenizer import SaraTokenizer
from .dynamics.fluid_field import FluidFieldDynamics
from .utils.future_state_runtime import LightweightFutureStateRuntime
from .utils.retrieval_diagnostics import format_retrieval_diagnostics, normalize_retrieval_diagnostic
from .utils.turboquant import HybridTurboQuantEngine, create_turboquant_engine, turboquant_metadata

# Try to import Rust core for Phase 3 (LIF Model)
try:
    import sara_rust_core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False


class _TokenIdList(list):
    def tolist(self) -> List[int]:
        return list(self)


class _NativeTokenizerAdapter:
    """Expose the small tokenizer subset used by the sparse inference path."""

    def __init__(self, tokenizer: SaraTokenizer) -> None:
        self._tokenizer = tokenizer

    def __call__(self, text: str, *, return_tensors: str = "pt") -> Dict[str, List[_TokenIdList]]:
        del return_tensors
        return {"input_ids": [_TokenIdList(self._tokenizer.encode(text))]}

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode([int(item) for item in ids])


class SaraInference:
    """
    SNN-based inference engine, designed to act as a lightweight replacement for 
    traditional Transformers AutoModelForCausalLM generation methods.
    Does not use backpropagation, matrix multiplication, or GPUs.
    """

    def __init__(
        self,
        model_path="models/distilled_sara_llm.msgpack",
        tokenizer_name: Optional[str] = None,
        tokenizer: Any = None,
        enable_turboquant: bool = False,
        turboquant_main_bits: int = 3,
        turboquant_residual_scale: Optional[float] = None,
    ):
        self.model_path = model_path
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "The optional 'ann-reference' dependencies are required for a Hugging Face tokenizer."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = _NativeTokenizerAdapter(SaraTokenizer())
        self.direct_map: Dict[Tuple[int, ...], Dict[int, float]] = {}
        self.context_index: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self.retrieval_diagnostics: List[Dict[str, object]] = []
        self.refractory_buffer: List[int] = []
        self.session_memory: Dict[str, str] = {}
        self.fast_intent_cache: Dict[Tuple[str, str], str] = {}
        self.predictor_state: Dict[str, object] = {}
        self.adaptation_state: Dict[str, object] = {}
        self.future_state_runtime_state: Dict[str, object] = {}
        self.context_encoding = "stable_v1"
        self.quantization_enabled = bool(enable_turboquant)
        self._turboquant_engine = create_turboquant_engine(
            main_bits=turboquant_main_bits,
            residual_scale=turboquant_residual_scale,
        )
        self._future_state_runtime = LightweightFutureStateRuntime()
        self._fluid_field_dynamics = FluidFieldDynamics()

        # Rust LIF Network for long context understanding (Phase 3)
        self.lif_network = None
        if HAS_RUST_CORE:
            # Emulating biological neuron decay and threshold
            self.lif_network = sara_rust_core.LIFNetwork(
                decay_rate=0.9, threshold=1.0)

        self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.model_path):
            print(
                f"[Warning] Memory file not found: {self.model_path}. Starting with an empty brain.")
            self.direct_map = {}
            self.context_index = {}
            self.retrieval_diagnostics = []
            self.session_memory = {}
            self.fast_intent_cache = {}
            self.predictor_state = {}
            self.adaptation_state = {}
            self.future_state_runtime_state = {}
            return

        with open(self.model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        if not isinstance(state, dict):
            state = {}
        quantized_direct_map = state.get("quantized_direct_map")
        if isinstance(quantized_direct_map, dict):
            self.quantization_enabled = bool(state.get("quantization_enabled", True))
            self._turboquant_engine = create_turboquant_engine(
                metadata=state.get("quantization") if isinstance(state.get("quantization"), dict) else None,
            )
            self.direct_map = self._get_turboquant_engine().restore_direct_map(quantized_direct_map)
        else:
            self.quantization_enabled = bool(state.get("quantization_enabled", False))
            self.direct_map = restore_direct_map(state.get("direct_map", {}))
        raw_context_encoding = state.get("context_encoding", "legacy_python_hash")
        if isinstance(raw_context_encoding, str) and raw_context_encoding:
            self.context_encoding = raw_context_encoding
        else:
            self.context_encoding = "legacy_python_hash"
        self.context_index = self._restore_context_index(state.get("context_index", {}))
        retrieval_diagnostics = state.get("retrieval_diagnostics", [])
        if isinstance(retrieval_diagnostics, list):
            self.retrieval_diagnostics = [
                item for item in retrieval_diagnostics if isinstance(item, dict)
            ][-10:]
        else:
            self.retrieval_diagnostics = []
        raw_session_memory = state.get("session_memory", {})
        if isinstance(raw_session_memory, dict):
            self.session_memory = {
                str(key): str(value)
                for key, value in raw_session_memory.items()
                if isinstance(key, str) and isinstance(value, (str, int, float))
            }
        else:
            self.session_memory = {}
        self.fast_intent_cache = {}
        raw_predictor_state = state.get("predictor_state", {})
        if isinstance(raw_predictor_state, dict):
            self.predictor_state = {
                str(key): value
                for key, value in raw_predictor_state.items()
                if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            }
        else:
            self.predictor_state = {}
        raw_adaptation_state = state.get("adaptation_state", {})
        if isinstance(raw_adaptation_state, dict):
            self.adaptation_state = {
                str(key): value
                for key, value in raw_adaptation_state.items()
                if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            }
        else:
            self.adaptation_state = {}
        self._sanitize_session_memory()
        self._refresh_predictor_state()

    def _encode_context_sdr(self, context_tokens):
        """
        Convert context tokens into a sparse representation key.
        Uses a deterministic context hash for new artifacts.
        """
        context_tuple = tuple(int(token) for token in context_tokens)
        if getattr(self, "context_encoding", "stable_v1") == "legacy_python_hash":
            return (hash(context_tuple),)

        hasher = hashlib.blake2b(digest_size=8)
        for token in context_tuple:
            hasher.update(int(token).to_bytes(8, byteorder="big", signed=True))
        return (int.from_bytes(hasher.digest(), byteorder="big", signed=False),)

    def _ensure_runtime_state(self) -> None:
        if getattr(self, "direct_map", None) is None:
            self.direct_map = {}
        if getattr(self, "context_index", None) is None:
            self.context_index = {}
        if getattr(self, "refractory_buffer", None) is None:
            self.refractory_buffer = []
        if getattr(self, "session_memory", None) is None:
            self.session_memory = {}
        if getattr(self, "fast_intent_cache", None) is None:
            self.fast_intent_cache = {}
        if getattr(self, "predictor_state", None) is None:
            self.predictor_state = {}
        if getattr(self, "adaptation_state", None) is None:
            self.adaptation_state = {}
        if getattr(self, "future_state_runtime_state", None) is None:
            self.future_state_runtime_state = {}
        if getattr(self, "retrieval_diagnostics", None) is None:
            self.retrieval_diagnostics = []
        if not getattr(self, "context_encoding", None):
            self.context_encoding = "stable_v1"
        self._sanitize_session_memory()
        self._refresh_predictor_state()

    def _session_memory_signature(self) -> str:
        goal = str(self.session_memory.get("goal", "")).strip().lower()
        task = str(self.session_memory.get("task", "")).strip().lower()
        location = str(self.session_memory.get("location", "")).strip().lower()
        return "|".join([goal, task, location])

    def _fast_intent_cache_key(self, prompt: str) -> Tuple[str, str]:
        normalized_prompt = str(prompt).strip().lower()
        return normalized_prompt, self._session_memory_signature()

    def _get_cached_fast_intent_response(self, prompt: str) -> Optional[str]:
        cache = getattr(self, "fast_intent_cache", None)
        if not isinstance(cache, dict):
            return None
        return cache.get(self._fast_intent_cache_key(prompt))

    def _put_cached_fast_intent_response(self, prompt: str, response: str) -> None:
        cache = getattr(self, "fast_intent_cache", None)
        if not isinstance(cache, dict):
            self.fast_intent_cache = {}
            cache = self.fast_intent_cache
        cache[self._fast_intent_cache_key(prompt)] = str(response)
        # Keep cache bounded for energy/memory efficiency.
        if len(cache) > 32:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)

    def _is_ultra_fast_prompt(self, prompt: str) -> bool:
        user_text = self._extract_latest_user_text(str(prompt))
        lowered = user_text.lower()

        if (
            "who are you" in lowered
            or "あなたは誰" in user_text
            or "あなたはだれ" in user_text
        ):
            return True

        if (
            re.search(r"\b(hello|hi|hey)\b", lowered)
            or "こんにちは" in user_text
            or "こんばんは" in user_text
        ):
            return True

        if any(term in lowered for term in ["日本語はわかりますか", "日本語わかりますか", "can you understand japanese"]):
            return True

        if any(term in lowered for term in ["do you have", "持っていますか", "ありますか"]):
            return True

        return False

    def _sanitize_session_memory(self) -> None:
        memory = getattr(self, "session_memory", None)
        if not isinstance(memory, dict):
            self.session_memory = {}
            return

        cleaned: Dict[str, str] = {}
        for key, value in memory.items():
            if not isinstance(key, str):
                continue
            normalized_value = str(value).strip()
            if not normalized_value:
                continue
            cleaned[key] = normalized_value

        profession = cleaned.get("profession", "")
        if any(marker in profession for marker in ["好き", "住んで", "名前", "出身"]):
            cleaned.pop("profession", None)

        self.session_memory = cleaned

    def _get_adaptation_state(self) -> Dict[str, object]:
        self._ensure_runtime_state()
        state = getattr(self, "adaptation_state", None)
        if not isinstance(state, dict):
            self.adaptation_state = {}
            state = self.adaptation_state
        return state

    def _update_adaptation_state(self, prompt: str) -> None:
        user_text = self._extract_latest_user_text(prompt)
        if not user_text:
            return

        state = self._get_adaptation_state()
        turns = int(state.get("adaptation_turns", 0) or 0) + 1
        next_step_requests = int(state.get("next_step_requests", 0) or 0)
        memory_requests = int(state.get("memory_requests", 0) or 0)
        lowered = user_text.lower()
        last_intent = "general"

        if any(term in lowered for term in ["what should i do next", "what do i do next", "what is the next step"]) or any(
            term in user_text for term in ["次に何をすればいい", "次に何をしたらいい", "次の一歩は何"]
        ):
            next_step_requests += 1
            last_intent = "next_step"
        elif any(
            term in lowered
            for term in [
                "do you remember me",
                "what is my goal",
                "what am i working on",
                "what is my name",
                "where do i live",
            ]
        ) or any(
            term in user_text
            for term in ["覚えていますか", "目標は何", "何をしている", "名前は何", "どこに住んで"]
        ):
            memory_requests += 1
            last_intent = "memory"

        has_plan_context = bool(self.session_memory.get("goal") and self.session_memory.get("task"))
        response_mode = "directive" if has_plan_context and next_step_requests >= 2 else "guided" if has_plan_context else "neutral"
        command_preference = bool(has_plan_context and next_step_requests >= 1)
        planning_confidence = min(
            1.0,
            (0.25 if has_plan_context else 0.0)
            + (0.3 * min(next_step_requests, 2))
            + (0.15 * min(memory_requests, 2)),
        )
        memory_weight = min(
            1.5,
            1.0
            + (0.12 * min(memory_requests, 2))
            + (0.18 * min(next_step_requests, 2)),
        )
        fallback_relaxation = min(
            0.10,
            (0.02 * min(memory_requests, 2))
            + (0.03 * min(next_step_requests, 2))
            + (0.02 if response_mode == "directive" else 0.0),
        )

        self.adaptation_state = {
            "adaptation_turns": turns,
            "next_step_requests": next_step_requests,
            "memory_requests": memory_requests,
            "response_mode": response_mode,
            "command_preference": command_preference,
            "planning_confidence": float(planning_confidence),
            "memory_weight": float(memory_weight),
            "fallback_relaxation": float(fallback_relaxation),
            "last_intent": last_intent,
        }

    def _remember_context(self, sdr_key: Tuple[int, ...], context_tokens: Sequence[int]) -> None:
        self.context_index[sdr_key] = tuple(int(token) for token in context_tokens)

    def _get_future_state_runtime(self) -> LightweightFutureStateRuntime:
        runtime = getattr(self, "_future_state_runtime", None)
        if runtime is None:
            runtime = LightweightFutureStateRuntime()
            self._future_state_runtime = runtime
        return runtime

    def _get_fluid_field_dynamics(self) -> FluidFieldDynamics:
        dynamics = getattr(self, "_fluid_field_dynamics", None)
        if dynamics is None:
            dynamics = FluidFieldDynamics()
            self._fluid_field_dynamics = dynamics
        return dynamics

    def _get_future_state_runtime_snapshot(self) -> Dict[str, object]:
        snapshot = getattr(self, "future_state_runtime_state", None)
        if isinstance(snapshot, dict):
            return dict(snapshot)
        self.future_state_runtime_state = {}
        return {}

    def _describe_future_state_shift(self, language: str = "en") -> str:
        snapshot = self._get_future_state_runtime_snapshot()
        previous_target = str(snapshot.get("last_shift_from", ""))
        current_target = str(snapshot.get("last_shift_to", ""))
        shift_count = int(snapshot.get("shift_count", 0) or 0)

        if not previous_target or not current_target or shift_count <= 0:
            return ""

        if language == "ja":
            return f"予測の焦点は {previous_target} から {current_target} へ切り替わっています。"
        return f"The predictive focus has shifted from {previous_target} to {current_target}."

    def _extract_transition_operator(self, action: str, command: str, response: str = "") -> str:
        text = f"{action} {command} {response}".lower()
        if any(token in text for token in ["highest-risk", "高リスク", "prioritize"]):
            return "release.risk_prioritize"
        if any(token in text for token in ["rollback condition", "ロールバック条件"]):
            return "release.rollback_guard"
        if any(token in text for token in ["release check", "リリース確認"]):
            return "release.check"
        if any(token in text for token in ["compare two candidate", "候補を2つ", "compare one alternative"]):
            return "research.compare"
        if any(token in text for token in ["contradictory source", "反対側の材料"]):
            return "research.contradictory_probe"
        if any(token in text for token in ["narrow", "問いを1つ", "one question"]):
            return "research.narrow_question"
        if any(token in text for token in ["concrete change", "変更点", "small unfinished action", "未完了タスク"]):
            return "development.change"
        if any(token in text for token in ["reproducible failure", "不具合", "原因を確認"]):
            return "debug.isolate"
        if any(token in text for token in ["first heading", "paragraph", "見出し", "段落"]):
            return "writing.draft"
        if any(token in text for token in ["rough sketch", "ラフ"]):
            return "visual.sketch"
        if any(token in text for token in ["pytest", "release_soak.py", "release_gate.py", "db-list"]):
            return "command.operational"
        return ""

    def _calculate_action_response_overlap(self, action: str, response: str) -> float:
        action_tokens = {token for token in re.split(r"[^a-z0-9]+", action.lower()) if token}
        response_tokens = {token for token in re.split(r"[^a-z0-9]+", response.lower()) if token}
        if not action_tokens:
            return 0.0
        return len(action_tokens.intersection(response_tokens)) / len(action_tokens)

    def _encode_fluid_token_ids(self, text: str) -> List[int]:
        token_ids: List[int] = []
        for char in text:
            if char.isspace():
                continue
            token_ids.append((ord(char) * 131) % 257)
        return token_ids[:16]

    def _build_fluid_trace(
        self,
        *,
        action: str,
        target_state: str,
        command: str,
    ) -> Dict[str, object]:
        token_ids = self._encode_fluid_token_ids(f"{action} {target_state} {command}")
        if not token_ids:
            return {}
        summary = self._get_fluid_field_dynamics().run(token_ids, steps=6)
        return {
            "active_columns": int(summary.get("active_columns", 0) or 0),
            "total_spikes": int(summary.get("total_spikes", 0) or 0),
            "peak_amplitude": int(summary.get("peak_amplitude", 0) or 0),
            "bounded": bool(summary.get("bounded", False)),
            "support_score": float(summary.get("support_score", 0.0) or 0.0),
        }

    def _build_speculative_trace(
        self,
        *,
        action: str,
        command: str,
        target_state: str,
        response: str,
        chosen_plan: str,
        alternative_action: str,
        secondary_alternative_action: str,
        runtime_state: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        predicted_operator = self._extract_transition_operator(action, command)
        verified_operator = self._extract_transition_operator(action, command, response)
        operator_match = bool(
            predicted_operator
            and verified_operator
            and predicted_operator == verified_operator
        )
        target_grounded = bool(target_state and target_state in response)
        action_overlap = self._calculate_action_response_overlap(action, response)
        draft_verify_accepted = bool(operator_match and target_grounded and action_overlap >= 0.30)
        rollback_required = bool(chosen_plan and chosen_plan != "primary")
        runtime_snapshot = runtime_state if isinstance(runtime_state, dict) else {}
        rollback_observable = bool(
            rollback_required
            and (
                (
                    int(runtime_snapshot.get("last_simulated_branch_count", 0) or 0) >= 2
                    and str(runtime_snapshot.get("last_best_simulated_branch", "")).strip()
                )
                or (alternative_action and secondary_alternative_action)
            )
        )
        counterfactual_branch_viable = bool(
            rollback_observable
            and alternative_action
            and secondary_alternative_action
            and alternative_action != secondary_alternative_action
        )
        return {
            "predicted_operator": predicted_operator,
            "verified_operator": verified_operator,
            "operator_match": operator_match,
            "action_response_overlap": float(action_overlap),
            "target_grounded": target_grounded,
            "draft_verify_accepted": draft_verify_accepted,
            "rollback_required": rollback_required,
            "rollback_observable": rollback_observable,
            "counterfactual_branch_viable": counterfactual_branch_viable,
        }

    def _should_run_adaptive_refinement(
        self,
        simulated_candidates: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        candidates = simulated_candidates if isinstance(simulated_candidates, list) else []
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            predictor_state = {}

        primary_confidence = float(predictor_state.get("confidence", 0.0) or 0.0)
        category = str(predictor_state.get("category", ""))
        top_score = float(candidates[0].get("simulation_score", 0.0) or 0.0) if candidates else 0.0
        second_score = float(candidates[1].get("simulation_score", 0.0) or 0.0) if len(candidates) > 1 else 0.0
        top_gap = top_score - second_score if len(candidates) > 1 else top_score

        reasons: List[str] = []
        uncertainty = max(0.0, min(1.0, 1.0 - top_gap)) if candidates else 1.0
        retrieval_conflict = bool(len(candidates) >= 2 and top_gap < 0.12)
        verifier_failure = bool(primary_confidence < 0.85)
        if not candidates:
            return {
                "triggered": False,
                "reasons": reasons,
                "top_score": float(top_score),
                "second_score": float(second_score),
                "top_gap": float(top_gap),
                "adaptive_depth_budget": {
                    "base_loop_budget": 1,
                    "allocated_loop_budget": 1,
                    "max_loop_budget": 2,
                    "depth_increase_applied": False,
                    "uncertainty": float(uncertainty),
                    "retrieval_conflict": False,
                    "verifier_failure": False,
                    "budget_reason": "no_candidates",
                },
            }
        if retrieval_conflict:
            reasons.append("narrow_score_gap")
        if verifier_failure:
            reasons.append("low_primary_confidence")
        if category == "research":
            reasons.append("research_compare_mode")
        triggered = bool(reasons)
        allocated_loop_budget = 2 if triggered else 1

        return {
            "triggered": triggered,
            "reasons": reasons,
            "top_score": float(top_score),
            "second_score": float(second_score),
            "top_gap": float(top_gap),
            "adaptive_depth_budget": {
                "base_loop_budget": 1,
                "allocated_loop_budget": allocated_loop_budget,
                "max_loop_budget": 2,
                "depth_increase_applied": bool(allocated_loop_budget > 1),
                "uncertainty": float(uncertainty),
                "retrieval_conflict": retrieval_conflict,
                "verifier_failure": verifier_failure,
                "budget_reason": ",".join(reasons) if reasons else "early_stop",
            },
        }

    def _refine_simulated_branch_candidates(
        self,
        simulated_candidates: Optional[List[Dict[str, object]]] = None,
        language: str = "en",
    ) -> Dict[str, object]:
        candidates = [
            dict(item)
            for item in (simulated_candidates if isinstance(simulated_candidates, list) else [])
            if isinstance(item, dict)
        ]
        if not candidates:
            candidates = self._simulate_future_state_branch_candidates(language=language)
        if not candidates:
            return {
                "triggered": False,
                "loop_count": 1,
                "reasons": [],
                "selected_branch_before": "primary",
                "selected_branch_after": "primary",
                "score_gap_before": 0.0,
                "score_gap_after": 0.0,
                "adaptive_depth_budget": {
                    "base_loop_budget": 1,
                    "allocated_loop_budget": 1,
                    "max_loop_budget": 2,
                    "depth_increase_applied": False,
                    "uncertainty": 1.0,
                    "retrieval_conflict": False,
                    "verifier_failure": False,
                    "budget_reason": "no_candidates",
                },
                "refined_candidates": [],
            }

        gating = self._should_run_adaptive_refinement(candidates)
        if not bool(gating.get("triggered", False)):
            chosen = str(candidates[0].get("label", "primary") or "primary") if candidates else "primary"
            return {
                "triggered": False,
                "loop_count": 1,
                "reasons": list(gating.get("reasons", [])),
                "selected_branch_before": chosen,
                "selected_branch_after": chosen,
                "score_gap_before": float(gating.get("top_gap", 0.0) or 0.0),
                "score_gap_after": float(gating.get("top_gap", 0.0) or 0.0),
                "adaptive_depth_budget": dict(gating.get("adaptive_depth_budget", {}))
                if isinstance(gating.get("adaptive_depth_budget"), dict)
                else {},
                "refined_candidates": candidates,
            }

        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            predictor_state = {}
        category = str(predictor_state.get("category", ""))
        fluid_trace = predictor_state.get("fluid_trace", {})
        fluid_support = (
            float(fluid_trace.get("support_score", 0.0) or 0.0)
            if isinstance(fluid_trace, dict)
            else 0.0
        )
        selected_before = str(candidates[0].get("label", "primary") or "primary")

        refined_candidates: List[Dict[str, object]] = []
        for item in candidates:
            refined = dict(item)
            label = str(item.get("label", ""))
            action = str(item.get("action", ""))
            command = str(item.get("command", ""))
            operator = self._extract_transition_operator(action, command)
            command_bonus = 0.03 if command else 0.0
            operator_bonus = 0.03 if operator else 0.0
            category_bonus = 0.0
            if category == "release" and label == "alternative":
                category_bonus += 0.04
            if category == "research" and label == "alternative":
                category_bonus += 0.05
            if label == "primary" and fluid_support >= 0.75:
                category_bonus += 0.02
            refined_score = min(
                1.0,
                float(item.get("simulation_score", 0.0) or 0.0)
                + command_bonus
                + operator_bonus
                + category_bonus,
            )
            refined["refined_simulation_score"] = float(refined_score)
            refined["refinement_operator"] = operator
            refined["refinement_bonus"] = float(command_bonus + operator_bonus + category_bonus)
            refined_candidates.append(refined)

        refined_candidates.sort(
            key=lambda item: (
                float(item.get("refined_simulation_score", item.get("simulation_score", 0.0)) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        top_after = (
            float(refined_candidates[0].get("refined_simulation_score", 0.0) or 0.0)
            if refined_candidates
            else 0.0
        )
        second_after = (
            float(refined_candidates[1].get("refined_simulation_score", 0.0) or 0.0)
            if len(refined_candidates) > 1
            else 0.0
        )
        selected_after = str(refined_candidates[0].get("label", "primary") or "primary") if refined_candidates else "primary"
        return {
            "triggered": True,
            "loop_count": 2,
            "reasons": list(gating.get("reasons", [])),
            "selected_branch_before": selected_before,
            "selected_branch_after": selected_after,
            "score_gap_before": float(gating.get("top_gap", 0.0) or 0.0),
            "score_gap_after": float(top_after - second_after),
            "adaptive_depth_budget": dict(gating.get("adaptive_depth_budget", {}))
            if isinstance(gating.get("adaptive_depth_budget"), dict)
            else {},
            "refined_candidates": refined_candidates,
        }

    def _build_primary_transition_response(
        self,
        *,
        action: str,
        target_state: str,
        language: str = "en",
    ) -> str:
        if not action:
            return ""
        if language == "ja":
            if target_state:
                return f"Step 1: {action}。 Step 2: それが{target_state}につながるか確認します。"
            return f"Step 1: {action}。"
        if target_state:
            return f"Step 1: {action}. Step 2: finish it and check that it moves you toward {target_state}."
        return f"Step 1: {action}."

    def _refresh_predictor_state(self) -> None:
        memory = getattr(self, "session_memory", None)
        if not isinstance(memory, dict) or not memory:
            self.predictor_state = {}
            self.future_state_runtime_state = {}
            self._get_future_state_runtime().reset()
            return

        language_seed = " ".join(str(value) for value in memory.values())
        inferred_language = "ja" if any(ord(char) > 127 for char in language_seed) else "en"
        prediction = self._predict_lightweight_future_state(language=inferred_language)
        category = str(prediction.get("category", ""))
        action = str(prediction.get("action", ""))
        target_state = str(prediction.get("target_state", ""))
        command = str(prediction.get("command", ""))
        confidence = float(prediction.get("confidence", 0.0))
        counterfactual = self._predict_counterfactual_future_state(language=inferred_language)
        alternative_action = str(counterfactual.get("action", ""))
        alternative_target_state = str(counterfactual.get("target_state", ""))
        alternative_command = str(counterfactual.get("command", ""))
        alternative_confidence = float(counterfactual.get("confidence", 0.0) or 0.0)
        secondary_counterfactual = self._predict_secondary_counterfactual_future_state(language=inferred_language)
        secondary_alternative_action = str(secondary_counterfactual.get("action", ""))
        secondary_alternative_target_state = str(secondary_counterfactual.get("target_state", ""))
        secondary_alternative_command = str(secondary_counterfactual.get("command", ""))
        secondary_alternative_confidence = float(secondary_counterfactual.get("confidence", 0.0) or 0.0)

        if not action and not target_state and not command and confidence <= 0.0:
            self.predictor_state = {}
            self.future_state_runtime_state = {}
            self._get_future_state_runtime().reset()
            return

        self.predictor_state = {
            "category": category,
            "action": action,
            "target_state": target_state,
            "command": command,
            "confidence": confidence,
            "alternative_action": alternative_action,
            "alternative_target_state": alternative_target_state,
            "alternative_command": alternative_command,
            "alternative_confidence": alternative_confidence,
            "secondary_alternative_action": secondary_alternative_action,
            "secondary_alternative_target_state": secondary_alternative_target_state,
            "secondary_alternative_command": secondary_alternative_command,
            "secondary_alternative_confidence": secondary_alternative_confidence,
        }
        branch_candidates = self._build_future_state_branch_candidates()
        simulated_branch_candidates = self._simulate_future_state_branch_candidates()
        refinement_trace = self._refine_simulated_branch_candidates(
            simulated_candidates=simulated_branch_candidates,
            language=inferred_language,
        )
        if bool(refinement_trace.get("triggered", False)):
            refined_candidates = refinement_trace.get("refined_candidates", [])
            if isinstance(refined_candidates, list) and refined_candidates:
                simulated_branch_candidates = refined_candidates
        best_simulated_branch = self._choose_best_simulated_branch(simulated_branch_candidates)
        self.predictor_state["branch_candidates"] = branch_candidates
        self.predictor_state["simulated_branch_candidates"] = simulated_branch_candidates
        self.predictor_state["best_simulated_branch"] = best_simulated_branch
        self.predictor_state["refinement_trace"] = refinement_trace
        preferred_branch = self._choose_preferred_next_step_plan()
        self.predictor_state["preferred_branch"] = preferred_branch
        reward_trace = self._build_reward_trace(
            selected_branch=preferred_branch,
            best_simulated_branch=best_simulated_branch,
            simulated_candidates=simulated_branch_candidates,
        )
        policy_trace = self._build_policy_trace(
            selected_branch=preferred_branch,
            best_simulated_branch=best_simulated_branch,
            reward_trace=reward_trace,
        )
        self.predictor_state["reward_trace"] = reward_trace
        self.predictor_state["policy_trace"] = policy_trace
        ranked_branch_candidates = self._rank_future_state_branch_candidates()
        self.predictor_state["ranked_branch_candidates"] = ranked_branch_candidates
        primary_response = self._task_hint(language=inferred_language, compact=True)
        if not primary_response:
            primary_response = self._build_primary_transition_response(
                action=action,
                target_state=target_state,
                language=inferred_language,
            )
        speculative_trace = self._build_speculative_trace(
            action=action,
            command=command,
            target_state=target_state,
            response=primary_response,
            chosen_plan=preferred_branch,
            alternative_action=alternative_action,
            secondary_alternative_action=secondary_alternative_action,
        )
        self.predictor_state["transition_operator"] = speculative_trace.get("predicted_operator", "")
        self.predictor_state["alternative_transition_operator"] = self._extract_transition_operator(
            alternative_action,
            alternative_command,
        )
        self.predictor_state["secondary_alternative_transition_operator"] = self._extract_transition_operator(
            secondary_alternative_action,
            secondary_alternative_command,
        )
        self.predictor_state["speculative_trace"] = speculative_trace
        self.predictor_state["fluid_trace"] = self._build_fluid_trace(
            action=action,
            target_state=target_state,
            command=command,
        )
        self.future_state_runtime_state = self._get_future_state_runtime().advance(
            self.predictor_state,
            language=inferred_language,
        )
        if isinstance(self.future_state_runtime_state, dict):
            runtime_trace = {
                **speculative_trace,
                "rollback_observable": bool(
                    self.future_state_runtime_state.get("last_rollback_observable", False)
                    or speculative_trace.get("rollback_observable", False)
                ),
                "counterfactual_branch_viable": bool(
                    self.future_state_runtime_state.get("last_counterfactual_viable", False)
                    or speculative_trace.get("counterfactual_branch_viable", False)
                ),
            }
            self.predictor_state["speculative_trace"] = runtime_trace

    def _get_turboquant_engine(self) -> HybridTurboQuantEngine:
        engine = getattr(self, "_turboquant_engine", None)
        if engine is None:
            engine = create_turboquant_engine()
            self._turboquant_engine = engine
        return engine

    def _maybe_quantize_direct_row(self, sdr_key: Tuple[int, ...]) -> None:
        if not getattr(self, "quantization_enabled", False):
            return
        if sdr_key not in self.direct_map:
            return
        payload = self._get_turboquant_engine().quantize_weight_row(self.direct_map[sdr_key])
        self.direct_map[sdr_key] = self._get_turboquant_engine().reconstruct_weight_row(payload)

    def _restore_context_index(self, raw_index: object) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        restored: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        if not isinstance(raw_index, dict):
            return restored

        restored_map = restore_direct_map(raw_index)
        for key, values in restored_map.items():
            ordered_tokens = sorted((int(position), int(token_id)) for position, token_id in values.items())
            restored[key] = tuple(token_id for _, token_id in ordered_tokens)
        return restored

    def _serialize_context_index(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[Tuple[int, ...], Dict[int, float]] = {}
        for key, context_tokens in self.context_index.items():
            payload[key] = {idx: float(token_id) for idx, token_id in enumerate(context_tokens)}
        return serialize_direct_map(payload)

    def _score_direct_context_alignment(
        self,
        query_context: Sequence[int],
        candidate_context: Sequence[int],
    ) -> Dict[str, float]:
        query_tuple = tuple(int(token) for token in query_context)
        candidate_tuple = tuple(int(token) for token in candidate_context)
        if not query_tuple or not candidate_tuple:
            return {
                "overlap": 0.0,
                "coverage": 0.0,
                "precision": 0.0,
                "jaccard": 0.0,
                "length_ratio": 0.0,
                "suffix_match": 0.0,
                "drift_penalty": 0.0,
                "stability_score": 0.0,
            }

        query_set = set(query_tuple)
        candidate_set = set(candidate_tuple)
        overlap = len(query_set.intersection(candidate_set))
        coverage = overlap / max(1, len(query_set))
        precision = overlap / max(1, len(candidate_set))
        jaccard = overlap / max(1, len(query_set.union(candidate_set)))
        length_ratio = min(len(query_tuple), len(candidate_tuple)) / max(1, max(len(query_tuple), len(candidate_tuple)))

        suffix_overlap = 0
        for q_token, c_token in zip(reversed(query_tuple), reversed(candidate_tuple)):
            if q_token != c_token:
                break
            suffix_overlap += 1
        suffix_match = suffix_overlap / max(1, min(len(query_tuple), len(candidate_tuple)))

        drift_tokens = len(candidate_set.difference(query_set))
        drift_penalty = min(0.4, drift_tokens / max(1, len(candidate_set)))

        stability_score = (
            (coverage * 0.35)
            + (precision * 0.15)
            + (jaccard * 0.15)
            + (length_ratio * 0.10)
            + (suffix_match * 0.25)
            - (drift_penalty * 0.15)
        )
        stability_score = max(0.0, min(1.0, stability_score))
        return {
            "overlap": float(overlap),
            "coverage": coverage,
            "precision": precision,
            "jaccard": jaccard,
            "length_ratio": length_ratio,
            "suffix_match": suffix_match,
            "drift_penalty": drift_penalty,
            "stability_score": stability_score,
        }

    def _find_best_matching_key(
        self,
        context_tokens: Sequence[int],
        capture_diagnostics: bool = True,
    ) -> Optional[Tuple[int, ...]]:
        if not context_tokens:
            return None

        context_tuple = tuple(int(token) for token in context_tokens)
        strict_key = self._encode_context_sdr(context_tuple)
        if strict_key in self.direct_map:
            self._remember_context(strict_key, context_tuple)
            if capture_diagnostics:
                self._capture_matching_diagnostic(
                    {
                        "content_preview": " ".join(str(token) for token in context_tuple),
                        "base_score": 1.0,
                        "stability_score": 1.0,
                        "suffix_match": 1.0,
                        "drift_penalty": 0.0,
                        "metadata_keyword_overlap": 1.0,
                        "context_match": True,
                        "role_match": True,
                    }
                )
            return strict_key

        context_set = set(context_tuple)
        if not context_set:
            return None

        best_key: Optional[Tuple[int, ...]] = None
        best_score = 0.0
        best_alignment: Optional[Dict[str, float]] = None
        best_context: Tuple[int, ...] = ()

        for candidate_key, candidate_context in self.context_index.items():
            if candidate_key not in self.direct_map or not candidate_context:
                continue

            alignment = self._score_direct_context_alignment(context_tuple, candidate_context)
            overlap = alignment["overlap"]
            if overlap <= 0:
                continue

            score = alignment["stability_score"]
            if alignment["coverage"] < 0.5 and alignment["suffix_match"] <= 0.0:
                continue

            if overlap == len(context_set):
                score += 0.15
            if alignment["suffix_match"] >= 0.5:
                score += 0.08
            score = min(1.0, score)

            if score > best_score:
                best_score = score
                best_key = candidate_key
                best_alignment = alignment
                best_context = tuple(int(token) for token in candidate_context)

        if best_score >= 0.45:
            if best_alignment is not None and capture_diagnostics:
                self._capture_matching_diagnostic(
                    {
                        "content_preview": " ".join(str(token) for token in best_context),
                        "base_score": best_alignment.get("coverage", 0.0),
                        "stability_score": best_alignment.get("stability_score", 0.0),
                        "suffix_match": best_alignment.get("suffix_match", 0.0),
                        "drift_penalty": best_alignment.get("drift_penalty", 0.0),
                        "metadata_keyword_overlap": best_alignment.get("jaccard", 0.0),
                        "context_match": best_alignment.get("coverage", 0.0) >= 0.5,
                        "role_match": best_alignment.get("suffix_match", 0.0) > 0.0,
                    }
                )
            return best_key
        if capture_diagnostics:
            self._capture_matching_diagnostic(
                {
                    "content_preview": " ".join(str(token) for token in context_tuple),
                    "base_score": 0.0,
                    "stability_score": 0.0,
                    "suffix_match": 0.0,
                    "drift_penalty": 1.0,
                    "metadata_keyword_overlap": 0.0,
                    "context_match": False,
                    "role_match": False,
                }
            )
        return None

    def learn_sequence(self, input_ids):
        """
        Online learning using Hebbian principles (cells that fire together, wire together).
        Updates synaptic weights continuously without backpropagation. O(1) dictionary updates.
        """
        self._ensure_runtime_state()
        if not input_ids:
            return

        for i in range(1, len(input_ids)):
            next_id = input_ids[i]
            max_window = min(8, i)
            for window in range(max_window, 0, -1):
                context = input_ids[i-window:i]
                sdr_k = self._encode_context_sdr(context)
                self._remember_context(sdr_k, context)

                if sdr_k not in self.direct_map:
                    self.direct_map[sdr_k] = {}

                # STDP-like reinforcement: strengthen synapse based on co-occurrence
                self.direct_map[sdr_k][next_id] = self.direct_map[sdr_k].get(next_id, 0.0) + 1.0
                self._maybe_quantize_direct_row(sdr_k)

    def save_pretrained(self, save_path):
        """
        Save the current synaptic weights (direct_map) to MessagePack.
        """
        self._ensure_runtime_state()
        # ファイルパスが直接指定された場合（拡張子で判定）
        if save_path.endswith(".msgpack"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_path = save_path
        else:
            # ディレクトリパスが指定された場合
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(save_path, "dummy_model.msgpack")

        with open(out_path, "wb") as f:
            payload = {
                "context_index": self._serialize_context_index(),
                "retrieval_diagnostics": list(self.retrieval_diagnostics[-10:]),
                "session_memory": dict(self.session_memory),
                "predictor_state": dict(self.predictor_state),
                "adaptation_state": dict(self.adaptation_state),
                "context_encoding": getattr(self, "context_encoding", "stable_v1"),
                "quantization_enabled": bool(getattr(self, "quantization_enabled", False)),
            }
            if getattr(self, "quantization_enabled", False):
                engine = self._get_turboquant_engine()
                payload["quantized_direct_map"] = engine.quantize_direct_map(self.direct_map)
                payload["quantization"] = turboquant_metadata(engine)
            else:
                payload["direct_map"] = serialize_direct_map(self.direct_map)
            msgpack.pack(
                payload,
                f,
            )
        print(f"[INFO] Brain state saved to {out_path}")

    def generate(self, prompt, max_new_tokens=50, top_k=3, temperature=0.5,
                 stop_conditions=None, refractory_penalty=1.2, refractory_period=10):
        """
        Generates text using pure SNN principles (Sparse Distributed Representations and LIF).
        """
        self._ensure_runtime_state()
        if self._is_ultra_fast_prompt(prompt):
            cached_fast = self._get_cached_fast_intent_response(prompt)
            if cached_fast is not None:
                self._capture_fast_path_diagnostic(prompt, cached_fast)
                return cached_fast

            fast_intent_response = self._fast_intent_response(prompt)
            if fast_intent_response is not None:
                self._put_cached_fast_intent_response(prompt, fast_intent_response)
                self._capture_fast_path_diagnostic(prompt, fast_intent_response)
                return fast_intent_response

        self._update_session_memory(prompt)
        self._update_adaptation_state(prompt)
        fast_intent_response = self._fast_intent_response(prompt)
        if fast_intent_response is not None:
            self._put_cached_fast_intent_response(prompt, fast_intent_response)
            self._capture_fast_path_diagnostic(prompt, fast_intent_response)
            return fast_intent_response

        if stop_conditions is None:
            # Multilingual stop conditions support
            stop_conditions = ["。", "！", "？", "!", "?", "\n", ".", "!", "?"]

        string_variations = [prompt, " " + prompt, "　" + prompt, "「" + prompt]

        current_tokens = []
        next_id = None
        initial_key = None

        # Initial context parsing
        for text_var in string_variations:
            base_tokens = self.tokenizer(text_var, return_tensors="pt")[
                "input_ids"][0].tolist()

            # Feed to LIF network if available (Phase 3: long context maintenance)
            if self.lif_network:
                self.lif_network.forward(base_tokens)

            for drop_last in [False, True]:
                search_tokens = base_tokens[:-1] if drop_last and len(
                    base_tokens) > 2 else base_tokens
                if not search_tokens:
                    continue

                max_window = min(8, len(search_tokens))
                for window in range(max_window, 0, -1):
                    context = search_tokens[-window:]
                    sdr_k = self._find_best_matching_key(context)

                    if sdr_k is not None and sdr_k in self.direct_map:
                        next_id = self._sample_next_token(
                            sdr_k, top_k, temperature, refractory_penalty)
                        if next_id is not None:
                            current_tokens = search_tokens
                            initial_key = sdr_k
                            break
                if next_id is not None:
                    break
            if next_id is not None:
                break

        if next_id is None:
            return ""

        next_id = self._select_best_opening_candidate(
            prompt=prompt,
            current_tokens=current_tokens,
            sdr_k=initial_key,
            sampled_next_id=next_id,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            refractory_penalty=refractory_penalty,
        )
        if next_id is None:
            return ""
        next_id = int(next_id)

        generated_text = ""
        for step in range(max_new_tokens):
            if step > 0:
                next_id = None
                max_window = min(8, len(current_tokens))
                for window in range(max_window, 0, -1):
                    context = current_tokens[-window:]
                    sdr_k = self._find_best_matching_key(context)

                    if sdr_k is not None and sdr_k in self.direct_map:
                        next_id = self._sample_next_token(
                            sdr_k, top_k, temperature, refractory_penalty)
                        if next_id is not None:
                            break

                # Fallback to Rust LIF context if strict match fails (Phase 3 Fuzzy retrieval)
                if next_id is None and self.lif_network:
                    # Simplified selection using biological context potential
                    next_id = random.choice(
                        current_tokens) if current_tokens else None

                if next_id is None:
                    break
                next_id = int(next_id)

            current_tokens.append(next_id)
            if self.lif_network:
                self.lif_network.forward([next_id])

            word = self.tokenizer.decode([next_id])
            generated_text += word

            # Biological refractory period mechanism (Memory of recent firings)
            self.refractory_buffer.append(next_id)
            if len(self.refractory_buffer) > refractory_period:
                self.refractory_buffer.pop(0)

            if any(generated_text.endswith(stop_word) for stop_word in stop_conditions):
                break

        return self._apply_practical_relevance_gate(prompt, generated_text)

    def _rank_next_token_candidates(self, sdr_k, refractory_penalty, refractory_buffer: Optional[Sequence[int]] = None):
        valid_candidates = []
        recent_tokens = list(self.refractory_buffer if refractory_buffer is None else refractory_buffer)
        for cid, w in self.direct_map[sdr_k].items():
            weight = float(w)

            # Apply biological refractory penalty for recently fired tokens
            if cid in recent_tokens:
                count = recent_tokens.count(cid)
                weight = weight / (refractory_penalty ** count)

            valid_candidates.append((cid, weight))

        if not valid_candidates:
            return []

        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_candidates

    def _sample_next_token(self, sdr_k, top_k, temperature, refractory_penalty):
        valid_candidates = self._rank_next_token_candidates(sdr_k, refractory_penalty)
        if not valid_candidates:
            return None

        if temperature <= 0.01 or top_k == 1:
            return valid_candidates[0][0]

        top_candidates = valid_candidates[:top_k]

        # Pure Python probability sampling without Matrix Operations
        probs = [math.pow(w, 1.0 / temperature) for _, w in top_candidates]
        probs_sum = sum(probs)

        if probs_sum == 0 or math.isnan(probs_sum):
            return top_candidates[0][0]

        probs = [p / probs_sum for p in probs]

        rand_val = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if rand_val <= cumulative:
                return top_candidates[i][0]

        return top_candidates[-1][0]

    def _prompt_needs_relevance_assist(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        return any(
            term in prompt_lower
            for term in [
                "who are you",
                "hello",
                "hi",
                "do you have",
                "こんにちは",
                "こんばんは",
                "ありますか",
                "持っていますか",
            ]
        )

    def _preview_response_from_candidate(
        self,
        current_tokens: Sequence[int],
        candidate_id: int,
        max_new_tokens: int,
        stop_conditions: Sequence[str],
        refractory_penalty: float,
    ) -> str:
        preview_tokens = list(int(token) for token in current_tokens)
        preview_refractory = list(getattr(self, "refractory_buffer", []))
        generated_ids = [int(candidate_id)]
        preview_tokens.append(int(candidate_id))
        preview_refractory.append(int(candidate_id))

        preview_limit = max(1, min(int(max_new_tokens), 12))
        for _ in range(preview_limit - 1):
            max_window = min(8, len(preview_tokens))
            next_id = None
            for window in range(max_window, 0, -1):
                context = preview_tokens[-window:]
                sdr_k = self._find_best_matching_key(context, capture_diagnostics=False)
                if sdr_k is None or sdr_k not in self.direct_map:
                    continue
                ranked_candidates = self._rank_next_token_candidates(
                    sdr_k,
                    refractory_penalty,
                    refractory_buffer=preview_refractory,
                )
                if ranked_candidates:
                    next_id = ranked_candidates[0][0]
                    break
            if next_id is None:
                break
            generated_ids.append(int(next_id))
            preview_tokens.append(int(next_id))
            preview_refractory.append(int(next_id))
            if len(preview_refractory) > 10:
                preview_refractory.pop(0)

            preview_text = self.tokenizer.decode(generated_ids)
            if any(preview_text.endswith(stop_word) for stop_word in stop_conditions):
                break

        return self.tokenizer.decode(generated_ids)

    def _select_best_opening_candidate(
        self,
        prompt: str,
        current_tokens: Sequence[int],
        sdr_k: Optional[Tuple[int, ...]],
        sampled_next_id: Optional[int],
        max_new_tokens: int,
        stop_conditions: Sequence[str],
        refractory_penalty: float,
    ) -> Optional[int]:
        if sampled_next_id is None or sdr_k is None or not self._prompt_needs_relevance_assist(prompt):
            return sampled_next_id

        ranked_candidates = self._rank_next_token_candidates(sdr_k, refractory_penalty)
        if not ranked_candidates:
            return sampled_next_id

        best_candidate_id = int(sampled_next_id)
        best_preview = self._preview_response_from_candidate(
            current_tokens=current_tokens,
            candidate_id=int(sampled_next_id),
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            refractory_penalty=refractory_penalty,
        )
        best_score = self._response_relevance_score(prompt, best_preview)

        for candidate_id, _weight in ranked_candidates[: min(5, len(ranked_candidates))]:
            candidate_id = int(candidate_id)
            preview = self._preview_response_from_candidate(
                current_tokens=current_tokens,
                candidate_id=candidate_id,
                max_new_tokens=max_new_tokens,
                stop_conditions=stop_conditions,
                refractory_penalty=refractory_penalty,
            )
            preview_score = self._response_relevance_score(prompt, preview)
            if preview_score > best_score + 0.05:
                best_candidate_id = candidate_id
                best_score = preview_score
                best_preview = preview

        if best_score >= 0.2:
            return best_candidate_id
        return sampled_next_id

    def _extract_prompt_keywords(self, prompt: str) -> List[str]:
        if not prompt:
            return []
        user_text = prompt
        if "You:" in prompt and "SARA:" in prompt:
            user_text = prompt.split("You:", 1)[-1].split("SARA:", 1)[0]
        keywords = []
        for token in re.findall(r"[A-Za-z]{3,}|[一-龥]{2,}|[ァ-ヴー]{2,}|[ぁ-ん]{2,}", user_text):
            normalized = token.strip().lower()
            if normalized and normalized not in keywords:
                keywords.append(normalized)
        return keywords

    def _extract_latest_user_text(self, prompt: str) -> str:
        if not prompt:
            return ""
        if "You:" in prompt and "SARA:" in prompt:
            return prompt.rsplit("You:", 1)[-1].split("SARA:", 1)[0].strip()
        return prompt.strip()

    def _looks_like_question(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "?" in text
            or "？" in text
            or lowered.startswith(("where ", "what ", "who ", "do you ", "can you ", "are you "))
            or any(
                phrase in text
                for phrase in ["どこ", "何", "誰", "ですか", "ますか", "覚えていますか", "わかりますか"]
            )
        )

    def _update_session_memory(self, prompt: str) -> None:
        self._ensure_runtime_state()
        user_text = self._extract_latest_user_text(prompt)
        if not user_text:
            return

        user_lower = user_text.lower().strip()
        if self._looks_like_question(user_text):
            return

        live_match = re.search(r"\bi live in ([a-z][a-z\s' -]{1,40})\b", user_lower)
        if live_match:
            self.session_memory["location"] = live_match.group(1).strip(" .!?").title()

        japanese_live_match = re.search(r"(?:私は|ぼくは|僕は|俺は)?(.{1,30})に住んでいます", user_text)
        if japanese_live_match:
            self.session_memory["location"] = japanese_live_match.group(1).strip(" 　。.!?")

        from_match = re.search(r"\bi am from ([a-z][a-z\s' -]{1,40})\b", user_lower)
        if from_match:
            self.session_memory["origin"] = from_match.group(1).strip(" .!?").title()

        japanese_from_match = re.search(r"(?:私は|ぼくは|僕は|俺は)?(.{1,30})出身です", user_text)
        if japanese_from_match:
            self.session_memory["origin"] = japanese_from_match.group(1).strip(" 　。.!?")

        name_match = re.search(r"\bmy name is ([a-z][a-z\s' -]{0,30})\b", user_lower)
        if name_match:
            self.session_memory["name"] = name_match.group(1).strip(" .!?").title()

        japanese_name_match = re.search(r"(?:私の名前は|名前は)(.{1,30})です", user_text)
        if japanese_name_match:
            self.session_memory["name"] = japanese_name_match.group(1).strip(" 　。.!?")

        profession_match = re.search(r"\bi am an? ([a-z][a-z\s' -]{1,40})\b", user_lower)
        if profession_match:
            self.session_memory["profession"] = profession_match.group(1).strip(" .!?")

        japanese_profession_match = re.search(
            r"(?:私は|ぼくは|僕は|俺は)?(.{1,30}(?:です|をしています))",
            user_text,
        )
        if japanese_profession_match:
            candidate = japanese_profession_match.group(1).strip(" 　。.!?")
            if (
                "名前" not in candidate
                and "どこ" not in candidate
                and "好き" not in candidate
                and any(
                    marker in candidate
                    for marker in [
                        "エンジニア",
                        "開発者",
                        "デザイナー",
                        "学生",
                        "会社員",
                        "研究者",
                        "教師",
                        "医師",
                        "看護師",
                        "イラストレーター",
                        "プログラマー",
                        "仕事",
                    ]
                )
            ):
                self.session_memory["profession"] = candidate

        like_match = re.search(r"\bi like ([a-z][a-z\s' -]{1,40})\b", user_lower)
        if like_match:
            self.session_memory["preference"] = like_match.group(1).strip(" .!?")

        japanese_like_match = re.search(r"私は(.{1,30})が好きです", user_text)
        if japanese_like_match:
            self.session_memory["preference"] = japanese_like_match.group(1).strip(" 　。.!?")
            if "profession" in self.session_memory and "好き" in self.session_memory["profession"]:
                self.session_memory.pop("profession", None)

        goal_match = re.search(r"\bi want to ([a-z][a-z\s' -]{1,60})\b", user_lower)
        if goal_match:
            self.session_memory["goal"] = goal_match.group(1).strip(" .!?")

        task_match = re.search(r"\bi am working on ([a-z][a-z0-9\s' -]{1,60})\b", user_lower)
        if task_match:
            self.session_memory["task"] = task_match.group(1).strip(" .!?")

        japanese_goal_match = re.search(r"私は(.{1,40})したいです", user_text)
        if japanese_goal_match:
            self.session_memory["goal"] = japanese_goal_match.group(1).strip(" 　。.!?")

        japanese_task_match = re.search(r"私は(.{1,40})をしています", user_text)
        if japanese_task_match:
            candidate = japanese_task_match.group(1).strip(" 　。.!?")
            if "仕事" not in candidate:
                self.session_memory["task"] = candidate
        self._refresh_predictor_state()

    def _describe_session_memory(self) -> str:
        self._ensure_runtime_state()
        facts = []
        if self.session_memory.get("name"):
            facts.append(f"your name is {self.session_memory['name']}")
        if self.session_memory.get("location"):
            facts.append(f"you live in {self.session_memory['location']}")
        if self.session_memory.get("origin"):
            facts.append(f"you are from {self.session_memory['origin']}")
        if self.session_memory.get("profession"):
            facts.append(f"you are {self.session_memory['profession']}")
        if self.session_memory.get("preference"):
            facts.append(f"you like {self.session_memory['preference']}")
        if self.session_memory.get("goal"):
            facts.append(f"you want to {self.session_memory['goal']}")
        if self.session_memory.get("task"):
            facts.append(f"you are working on {self.session_memory['task']}")
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        return ", and ".join([", ".join(facts[:-1]), facts[-1]]) if len(facts) > 2 else " and ".join(facts)

    def _build_natural_session_summary(self) -> str:
        self._ensure_runtime_state()
        summary_parts = []
        priority_fields = [
            ("name", "your name is {value}"),
            ("location", "you live in {value}"),
            ("task", "you are working on {value}"),
            ("goal", "you want to {value}"),
            ("preference", "you like {value}"),
            ("profession", "you are {value}"),
            ("origin", "you are from {value}"),
        ]
        for key, template in priority_fields:
            value = self.session_memory.get(key)
            if value:
                summary_parts.append(template.format(value=value))
        summary_parts = summary_parts[:4]

        if not summary_parts:
            return ""
        if len(summary_parts) == 1:
            return summary_parts[0]
        if len(summary_parts) == 2:
            return " and ".join(summary_parts)
        return ", ".join(summary_parts[:-1]) + f", and {summary_parts[-1]}"

    def _format_session_value(self, key: str, language: str = "en") -> str:
        value = self.session_memory.get(key, "")
        if not value:
            return ""
        if language != "ja":
            return value
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s'._-]*", value):
            return f"英語の「{value}」"
        if re.search(r"[A-Za-z]", value):
            return f"「{value}」"
        return value

    def _format_session_label_ja(self, key: str) -> str:
        value = self._format_session_value(key, language="ja")
        if not value:
            return ""
        if key in {"goal", "task"} and not value.startswith("「") and not value.startswith("英語の「"):
            return f"「{value}」"
        return value

    def _build_alternative_next_step_response(self, language: str = "en") -> str:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            return ""

        alternative_action = str(predictor_state.get("alternative_action", ""))
        alternative_target_state = str(predictor_state.get("alternative_target_state", ""))
        alternative_command = str(predictor_state.get("alternative_command", ""))

        if not alternative_action:
            return ""

        if language == "ja":
            response = f"別案としては、まず{alternative_action}のがよいです。"
            if alternative_target_state:
                response += f" そうすると{alternative_target_state}にも近づけます。"
            if alternative_command:
                response += f" 提案コマンド: `{alternative_command}`"
            return response

        response = f"An alternative next step is to {alternative_action}."
        if alternative_target_state:
            response += f" That can also move you toward {alternative_target_state}."
        if alternative_command:
            response += f" Suggested command: `{alternative_command}`"
        return response

    def _build_secondary_alternative_next_step_response(self, language: str = "en") -> str:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            return ""

        alternative_action = str(predictor_state.get("secondary_alternative_action", ""))
        alternative_target_state = str(predictor_state.get("secondary_alternative_target_state", ""))
        alternative_command = str(predictor_state.get("secondary_alternative_command", ""))

        if not alternative_action:
            return ""

        if language == "ja":
            response = f"もう一つの別案としては、まず{alternative_action}のがよいです。"
            if alternative_target_state:
                response += f" そうすると{alternative_target_state}にも近づけます。"
            if alternative_command:
                response += f" 提案コマンド: `{alternative_command}`"
            return response

        response = f"A second alternative next step is to {alternative_action}."
        if alternative_target_state:
            response += f" That can also move you toward {alternative_target_state}."
        if alternative_command:
            response += f" Suggested command: `{alternative_command}`"
        return response

    def _build_future_state_branch_candidates(self, language: str = "en") -> List[Dict[str, object]]:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            return []

        action = str(predictor_state.get("action", ""))
        target_state = str(predictor_state.get("target_state", ""))
        command = str(predictor_state.get("command", ""))
        if not action:
            primary_prediction = self._predict_lightweight_future_state(language=language)
            action = str(primary_prediction.get("action", ""))
            if not target_state:
                target_state = str(primary_prediction.get("target_state", ""))
            if not command:
                command = str(primary_prediction.get("command", ""))
        primary_text = ""
        if action:
            if language == "ja":
                primary_text = f"次の一歩として、まず{action}のがよいです。"
                if target_state:
                    primary_text += f" そうすると{target_state}に近づけます。"
                if command:
                    primary_text += f" 提案コマンド: `{command}`"
            else:
                primary_text = f"Step 1: {action}. Step 2: finish it and check that it moves you toward {target_state}."
                if command:
                    primary_text += f" Suggested command: `{command}`"
        alternative_text = self._build_alternative_next_step_response(language=language)
        secondary_text = self._build_secondary_alternative_next_step_response(language=language)

        candidates: List[Dict[str, object]] = []
        if primary_text:
            candidates.append(
                {
                    "kind": "primary",
                    "label": "primary",
                    "action": action,
                    "target_state": target_state,
                    "command": command,
                    "response": primary_text,
                    "confidence": float(predictor_state.get("confidence", 0.0) or 0.0),
                }
            )
        if alternative_text:
            candidates.append(
                {
                    "kind": "alternative",
                    "label": "alternative",
                    "action": str(predictor_state.get("alternative_action", "")),
                    "target_state": str(predictor_state.get("alternative_target_state", "")),
                    "command": str(predictor_state.get("alternative_command", "")),
                    "response": alternative_text,
                    "confidence": float(predictor_state.get("alternative_confidence", 0.0) or 0.0),
                }
            )
        if secondary_text:
            candidates.append(
                {
                    "kind": "secondary",
                    "label": "secondary",
                    "action": str(predictor_state.get("secondary_alternative_action", "")),
                    "target_state": str(predictor_state.get("secondary_alternative_target_state", "")),
                    "command": str(predictor_state.get("secondary_alternative_command", "")),
                    "response": secondary_text,
                    "confidence": float(predictor_state.get("secondary_alternative_confidence", 0.0) or 0.0),
                }
            )
        return candidates

    def _simulate_future_state_branch_candidates(self, language: str = "en") -> List[Dict[str, object]]:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            predictor_state = {}
        category = str(predictor_state.get("category", ""))
        candidates = self._build_future_state_branch_candidates(language=language)
        simulated: List[Dict[str, object]] = []

        for item in candidates:
            action = str(item.get("action", ""))
            command = str(item.get("command", ""))
            label = str(item.get("label", ""))
            lowered_action = action.lower()
            goal_text = str(self.session_memory.get("goal", ""))
            task_text = str(self.session_memory.get("task", ""))
            release_like_context = any(
                marker in f"{goal_text} {task_text} {action}"
                for marker in ["release", "ship", "公開", "リリース"]
            )

            progress_score = 0.55
            if command:
                progress_score += 0.20
            if label == "primary":
                progress_score += 0.15
            elif label == "secondary":
                progress_score -= 0.10
            if any(marker in lowered_action for marker in ["finish", "complete", "ship", "release", "実行", "仕上げ"]):
                progress_score += 0.10
            if any(marker in lowered_action for marker in ["compare", "check", "review", "inspect", "比較", "確認"]):
                progress_score += 0.05

            risk_reduction_score = 0.45
            if any(
                marker in lowered_action
                for marker in [
                    "highest-risk",
                    "most fragile",
                    "rollback",
                    "risk",
                    "最も壊れやすい",
                    "高リスク",
                    "ロールバック",
                    "影響が最も大きい",
                ]
            ):
                risk_reduction_score += 0.40
            elif any(marker in lowered_action for marker in ["compare", "check", "review", "比較", "確認"]):
                risk_reduction_score += 0.25
            elif label == "primary":
                risk_reduction_score += 0.15

            reversibility_score = 0.40
            if any(marker in lowered_action for marker in ["compare", "check", "review", "inspect", "比較", "確認"]):
                reversibility_score += 0.45
            elif any(marker in lowered_action for marker in ["rollback", "guard", "safety", "ロールバック", "安全"]):
                reversibility_score += 0.35
            elif label == "primary":
                reversibility_score += 0.20

            category_bonus = 0.0
            if (category == "release" or release_like_context) and any(
                marker in lowered_action
                for marker in ["highest-risk", "most fragile", "rollback", "高リスク", "ロールバック", "影響が最も大きい"]
            ):
                category_bonus += 0.15
            if category == "research" and any(
                marker in lowered_action
                for marker in ["compare", "candidate", "direction", "比較", "候補"]
            ):
                category_bonus += 0.15
            if label == "alternative" and any(
                marker in lowered_action
                for marker in ["highest-risk", "most fragile", "compare", "影響が最も大きい", "比較"]
            ):
                category_bonus += 0.05
            if label == "secondary":
                category_bonus -= 0.05
            if category not in {"release", "research"} and label == "primary":
                category_bonus += 0.10

            confidence = float(item.get("confidence", 0.0) or 0.0)
            simulation_score = min(
                1.0,
                (0.30 * min(progress_score, 1.0))
                + (0.30 * min(risk_reduction_score, 1.0))
                + (0.20 * min(reversibility_score, 1.0))
                + (0.20 * min(confidence, 1.0))
                + category_bonus,
            )

            rationale = (
                "進みやすさ、リスク低減、戻しやすさを軽量シミュレーションで比較した結果です。"
                if language == "ja"
                else "This lightweight simulation compares expected progress, risk reduction, and reversibility."
            )
            simulated.append(
                {
                    **dict(item),
                    "progress_score": float(min(progress_score, 1.0)),
                    "risk_reduction_score": float(min(risk_reduction_score, 1.0)),
                    "reversibility_score": float(min(reversibility_score, 1.0)),
                    "simulation_score": float(simulation_score),
                    "simulation_rationale": rationale,
                }
            )

        simulated.sort(
            key=lambda item: (
                float(item.get("simulation_score", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return simulated

    def _choose_best_simulated_branch(self, simulated_candidates: Optional[List[Dict[str, object]]] = None) -> str:
        candidates = simulated_candidates
        if not isinstance(candidates, list):
            predictor_state = getattr(self, "predictor_state", {})
            if isinstance(predictor_state, dict) and isinstance(predictor_state.get("simulated_branch_candidates"), list):
                candidates = predictor_state.get("simulated_branch_candidates")
            else:
                candidates = self._simulate_future_state_branch_candidates()
        if not candidates:
            return "primary"
        return str(candidates[0].get("label", "primary") or "primary")

    def _build_reward_trace(
        self,
        *,
        selected_branch: str,
        best_simulated_branch: str,
        simulated_candidates: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        candidates = simulated_candidates if isinstance(simulated_candidates, list) else []
        selected_candidate: Dict[str, object] = {}
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get("label", "")) == selected_branch:
                selected_candidate = item
                break
        if not selected_candidate and candidates and isinstance(candidates[0], dict):
            selected_candidate = candidates[0]

        progress_score = float(selected_candidate.get("progress_score", 0.0) or 0.0)
        risk_reduction_score = float(selected_candidate.get("risk_reduction_score", 0.0) or 0.0)
        reversibility_score = float(selected_candidate.get("reversibility_score", 0.0) or 0.0)
        simulation_score = float(selected_candidate.get("simulation_score", 0.0) or 0.0)
        confidence = float(selected_candidate.get("confidence", 0.0) or 0.0)
        energy_cost_proxy = max(
            0.0,
            min(1.0, 1.0 - (0.50 * reversibility_score + 0.25 * risk_reduction_score + 0.25 * confidence)),
        )
        branch_alignment = 1.0 if selected_branch == best_simulated_branch else 0.5
        user_feedback_signal = branch_alignment * 0.1
        total_reward = max(
            0.0,
            min(
                1.0,
                (0.35 * progress_score)
                + (0.25 * risk_reduction_score)
                + (0.20 * reversibility_score)
                + (0.10 * simulation_score)
                + (0.10 * (1.0 - energy_cost_proxy))
                + user_feedback_signal,
            ),
        )
        return {
            "progress_score": float(progress_score),
            "risk_reduction_score": float(risk_reduction_score),
            "reversibility_score": float(reversibility_score),
            "energy_cost_proxy": float(energy_cost_proxy),
            "user_feedback_signal": float(user_feedback_signal),
            "total_reward": float(total_reward),
            "selected_branch": str(selected_branch or "primary"),
        }

    def _build_policy_trace(
        self,
        *,
        selected_branch: str,
        best_simulated_branch: str,
        reward_trace: Dict[str, object],
    ) -> Dict[str, object]:
        selected = str(selected_branch or "").strip() or "primary"
        best = str(best_simulated_branch or "").strip() or selected
        selection_consistent = selected == best
        policy_shift_applied = not selection_consistent
        reward_signal = float(reward_trace.get("total_reward", 0.0) or 0.0)
        policy_stability = max(0.0, min(1.0, reward_signal if selection_consistent else reward_signal * 0.8))
        return {
            "selected_branch": selected,
            "best_simulated_branch": best,
            "selection_consistent": bool(selection_consistent),
            "policy_shift_applied": bool(policy_shift_applied),
            "policy_stability": float(policy_stability),
        }

    def _rank_future_state_branch_candidates(self, language: str = "en") -> List[Dict[str, object]]:
        predictor_state = getattr(self, "predictor_state", {})
        if isinstance(predictor_state, dict) and isinstance(predictor_state.get("simulated_branch_candidates"), list):
            candidates = [dict(item) for item in predictor_state.get("simulated_branch_candidates", []) if isinstance(item, dict)]
        else:
            candidates = self._simulate_future_state_branch_candidates(language=language)
        if len(candidates) < 2:
            return candidates

        chosen_plan = self._choose_preferred_next_step_plan()
        priority_order = {
            "primary": 1 if chosen_plan == "primary" else 0,
            "alternative": 1 if chosen_plan == "alternative" else 0,
            "secondary": 0,
        }
        ranked = [dict(item) for item in candidates]
        ranked.sort(
            key=lambda item: (
                priority_order.get(str(item.get("kind", "")), 0),
                float(item.get("simulation_score", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return ranked

    def _build_next_step_comparison_response(self, language: str = "en") -> str:
        primary = self._build_next_step_response(language=language)
        alternative = self._build_alternative_next_step_response(language=language)
        if not primary or not alternative:
            return ""

        if language == "ja":
            return f"主案: {primary}\n別案: {alternative}"
        return f"Primary: {primary}\nAlternative: {alternative}"

    def _build_next_step_options_response(self, language: str = "en") -> str:
        primary = self._build_next_step_response(language=language)
        alternative = self._build_alternative_next_step_response(language=language)
        secondary_alternative = self._build_secondary_alternative_next_step_response(language=language)

        options = [item for item in [primary, alternative, secondary_alternative] if item]
        if len(options) < 2:
            return ""

        if language == "ja":
            lines = [f"主案: {primary}"]
            if alternative:
                lines.append(f"別案: {alternative}")
            if secondary_alternative:
                lines.append(f"追加案: {secondary_alternative}")
            return "\n".join(lines)

        lines = [f"Primary: {primary}"]
        if alternative:
            lines.append(f"Alternative: {alternative}")
        if secondary_alternative:
            lines.append(f"Additional: {secondary_alternative}")
        return "\n".join(lines)

    def _build_ranked_next_step_options_response(self, language: str = "en") -> str:
        ranked_items = self._rank_future_state_branch_candidates(language=language)
        if len(ranked_items) < 2:
            return ""

        if language == "ja":
            label_map = {"primary": "主案", "alternative": "別案", "secondary": "追加案"}
            return "\n".join(
                f"{index}位 ({label_map.get(str(item.get('kind', '')), '候補')}): {str(item.get('response', ''))}"
                for index, item in enumerate(ranked_items, start=1)
            )

        label_map = {"primary": "Primary", "alternative": "Alternative", "secondary": "Additional"}
        return "\n".join(
            f"{index}. {label_map.get(str(item.get('kind', '')), 'Option')}: {str(item.get('response', ''))}"
            for index, item in enumerate(ranked_items, start=1)
        )

    def _choose_preferred_next_step_plan(self) -> str:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            return "primary"

        category = str(predictor_state.get("category", ""))
        best_simulated_branch = str(predictor_state.get("best_simulated_branch", "")).strip()
        alternative_action = str(predictor_state.get("alternative_action", ""))
        if not alternative_action:
            return "primary"
        if best_simulated_branch in {"primary", "alternative", "secondary"}:
            return best_simulated_branch

        lowered_alternative = alternative_action.lower()
        if any(
            marker in lowered_alternative
            for marker in ["highest-risk", "most fragile", "影響が最も大きい", "最も壊れやすい"]
        ):
            return "alternative"
        if category == "research" or any(
            marker in lowered_alternative
            for marker in ["compare two candidate directions", "比較したい候補を2つ"]
        ):
            return "alternative"
        return "primary"

    def _build_next_step_choice_reason(self, chosen_plan: str, language: str = "en") -> str:
        predictor_state = getattr(self, "predictor_state", {})
        if not isinstance(predictor_state, dict):
            return ""

        category = str(predictor_state.get("category", ""))
        alternative_action = str(predictor_state.get("alternative_action", ""))
        lowered_alternative = alternative_action.lower()

        if chosen_plan == "alternative":
            if any(
                marker in lowered_alternative
                for marker in ["highest-risk", "most fragile", "影響が最も大きい", "最も壊れやすい"]
            ):
                if language == "ja":
                    return "理由: 先に高リスク側を確認したほうが、手戻りを減らしやすいからです。"
                return "Reason: checking the highest-risk path first is more likely to reduce rework."
            if category == "research" or any(
                marker in lowered_alternative
                for marker in ["compare two candidate directions", "比較したい候補を2つ"]
            ):
                if language == "ja":
                    return "理由: 候補を先に比べると、次の判断がしやすくなるからです。"
                return "Reason: comparing candidates first makes the next decision clearer."
        if language == "ja":
            return "理由: まずは最短で前進できる案から進めるのが安定しやすいからです。"
        return "Reason: starting with the most direct plan is the most stable way to make progress."

    def _build_next_step_simulation_response(self, language: str = "en") -> str:
        simulated = self._simulate_future_state_branch_candidates(language=language)
        if len(simulated) < 2:
            return ""

        if language == "ja":
            label_map = {"primary": "主案", "alternative": "別案", "secondary": "追加案"}
            lines = ["軽量シミュレーション:"]
            for item in simulated:
                label = label_map.get(str(item.get("label", "")), "候補")
                lines.append(
                    f"- {label}: score={float(item.get('simulation_score', 0.0) or 0.0):.3f}, "
                    f"progress={float(item.get('progress_score', 0.0) or 0.0):.3f}, "
                    f"risk={float(item.get('risk_reduction_score', 0.0) or 0.0):.3f}, "
                    f"reversible={float(item.get('reversibility_score', 0.0) or 0.0):.3f}"
                )
            return "\n".join(lines)

        label_map = {"primary": "Primary", "alternative": "Alternative", "secondary": "Additional"}
        lines = ["Lightweight simulation:"]
        for item in simulated:
            label = label_map.get(str(item.get("label", "")), "Option")
            lines.append(
                f"- {label}: score={float(item.get('simulation_score', 0.0) or 0.0):.3f}, "
                f"progress={float(item.get('progress_score', 0.0) or 0.0):.3f}, "
                f"risk={float(item.get('risk_reduction_score', 0.0) or 0.0):.3f}, "
                f"reversible={float(item.get('reversibility_score', 0.0) or 0.0):.3f}"
            )
        return "\n".join(lines)

    def _build_next_step_choice_response(self, language: str = "en") -> str:
        primary = self._build_next_step_response(language=language)
        alternative = self._build_alternative_next_step_response(language=language)
        if not primary:
            return ""

        chosen_plan = self._choose_preferred_next_step_plan()
        reason = self._build_next_step_choice_reason(chosen_plan, language=language)
        if chosen_plan == "alternative" and alternative:
            if language == "ja":
                return f"まずは別案から進めるのがよいです: {alternative} {reason}".strip()
            return f"I would start with the alternative plan: {alternative} {reason}".strip()

        if language == "ja":
            return f"まずは主案から進めるのがよいです: {primary} {reason}".strip()
        return f"I would start with the primary plan: {primary} {reason}".strip()

    def _build_next_step_decision_brief(self, language: str = "en") -> str:
        choice = self._build_next_step_choice_response(language=language)
        ranked = self._build_ranked_next_step_options_response(language=language)
        if not choice or not ranked:
            return ""

        ranked_lines = [line for line in ranked.splitlines() if line.strip()][:2]
        if language == "ja":
            lines = ["判断メモ:", choice]
            lines.extend(ranked_lines)
            return "\n".join(lines)

        lines = ["Decision brief:", choice]
        lines.extend(ranked_lines)
        return "\n".join(lines)

    def _fast_intent_response(self, prompt: str) -> Optional[str]:
        user_text = self._extract_latest_user_text(prompt)
        if not user_text:
            return None

        user_lower = user_text.lower()
        if any(term in user_lower for term in ["who are you", "あなたは誰", "君は誰"]):
            return "I am SARA, a CPU-first spiking neural network assistant."

        if (
            re.search(r"\b(hello|hi|hey)\b", user_lower)
            or "こんにちは" in user_text
            or "こんばんは" in user_text
        ):
            return "Hello. I am SARA. How can I help you?"

        if any(term in user_lower for term in ["do you have", "持っていますか", "ありますか"]):
            return "I am a software agent, so I do not have physical objects, but I can help with information."

        if any(term in user_lower for term in ["do you remember me", "覚えていますか", "remember me"]):
            remembered = self._build_natural_session_summary()
            if remembered:
                return f"Yes. In this conversation, I remember that {remembered}."
            return "I can use what you share in the current conversation, but I do not reliably remember personal details across every session yet."

        if any(term in user_lower for term in ["日本語はわかりますか", "日本語わかりますか", "can you understand japanese"]):
            return "Yes. I can understand Japanese and English, although my answers may still be simpler than a large language model."

        if any(term in user_lower for term in ["where do i live", "what city do i live in", "where am i from"]):
            if self.session_memory.get("location"):
                return f"In this conversation, you told me that you live in {self.session_memory['location']}."
            if self.session_memory.get("origin"):
                return f"In this conversation, you told me that you are from {self.session_memory['origin']}."
            return "You have not told me your location in this conversation yet."

        if any(term in user_lower for term in ["どこに住んで", "どこに住んでいます", "どこ出身"]):
            if self.session_memory.get("location"):
                return f"この会話では、あなたは{self._format_session_value('location', language='ja')}に住んでいると教えてくれました。"
            if self.session_memory.get("origin"):
                return f"この会話では、あなたは{self._format_session_value('origin', language='ja')}出身だと教えてくれました。"
            return "この会話では、まだ住んでいる場所を聞いていません。"

        if any(term in user_lower for term in ["what is my name", "what's my name", "do you know my name"]):
            if self.session_memory.get("name"):
                return f"In this conversation, you told me that your name is {self.session_memory['name']}."
            return "You have not told me your name in this conversation yet."

        if any(term in user_lower for term in ["私の名前は", "名前を覚えて", "名前は何"]):
            if self.session_memory.get("name"):
                return f"この会話では、あなたの名前は{self._format_session_value('name', language='ja')}だと教えてくれました。"
            return "この会話では、まだ名前を聞いていません。"

        if any(term in user_lower for term in ["what do i like", "what do i like?"]):
            if self.session_memory.get("preference"):
                return f"In this conversation, you told me that you like {self.session_memory['preference']}."
            return "You have not told me what you like in this conversation yet."

        if any(term in user_lower for term in ["何が好き", "好きなものは"]):
            if self.session_memory.get("preference"):
                return f"この会話では、あなたは{self._format_session_value('preference', language='ja')}が好きだと教えてくれました。"
            return "この会話では、まだ好きなものを聞いていません。"

        if any(term in user_lower for term in ["what do i do", "what is my job", "what is my profession"]):
            if self.session_memory.get("profession"):
                return f"In this conversation, you told me that you are {self.session_memory['profession']}."
            return "You have not told me your work or role in this conversation yet."

        if any(term in user_lower for term in ["仕事は何", "職業は何", "私は何をしている"]):
            if self.session_memory.get("profession"):
                return f"この会話では、あなたは{self._format_session_value('profession', language='ja')}だと教えてくれました。"
            return "この会話では、まだ仕事や役割を聞いていません。"

        if any(term in user_lower for term in ["what do i want", "what is my goal", "what am i trying to do"]):
            if self.session_memory.get("goal"):
                suggestion = self._build_goal_suggestion()
                return (
                    f"In this conversation, you told me that you want to {self.session_memory['goal']}. "
                    f"{suggestion}"
                )
            return "You have not told me your goal in this conversation yet."

        if any(term in user_lower for term in ["what am i working on", "what is my task"]):
            if self.session_memory.get("task"):
                suggestion = self._build_task_suggestion()
                return (
                    f"In this conversation, you told me that you are working on {self.session_memory['task']}. "
                    f"{suggestion}"
                )
            return "You have not told me your current task in this conversation yet."

        if any(term in user_lower for term in ["目標は何", "何をしたい", "やりたいことは何"]):
            if self.session_memory.get("goal"):
                return f"この会話では、あなたは{self._format_session_label_ja('goal')}を目標にしていると教えてくれました。{self._build_goal_suggestion(language='ja')}"
            return "この会話では、まだ目標を聞いていません。"

        if any(term in user_lower for term in ["何をしている", "今の作業は何", "作業は何"]):
            if self.session_memory.get("task"):
                return f"この会話では、あなたは{self._format_session_label_ja('task')}をしていると教えてくれました。{self._build_task_suggestion(language='ja')}"
            return "この会話では、まだ今の作業を聞いていません。"

        if any(term in user_lower for term in ["which one should i choose", "which next step should i choose", "which plan should i choose"]):
            choice = self._build_next_step_choice_response()
            if choice:
                return choice
            return "Tell me your current task or goal, and I can suggest which next step to choose first."

        if any(term in user_lower for term in ["show all next step options", "list the next step options", "show my next step options"]):
            options = self._build_next_step_options_response()
            if options:
                return options
            return "Tell me your current task or goal, and I can list the current next-step options."

        if any(term in user_lower for term in ["summarize the next step decision", "give me a decision brief", "summarize the next step options"]):
            brief = self._build_next_step_decision_brief()
            if brief:
                return brief
            return "Tell me your current task or goal, and I can summarize the current next-step decision."

        if any(term in user_lower for term in ["simulate the next step options", "run a lightweight simulation", "simulate the next steps", "run an offline simulation"]):
            simulation = self._build_next_step_simulation_response()
            if simulation:
                return simulation
            return "Tell me your current task or goal, and I can simulate the current next-step options."

        if any(term in user_lower for term in ["rank the next step options", "rank my next steps", "prioritize the next step options"]):
            ranked_options = self._build_ranked_next_step_options_response()
            if ranked_options:
                return ranked_options
            return "Tell me your current task or goal, and I can rank the current next-step options."

        if any(term in user_lower for term in ["compare the next steps", "compare next steps", "compare my next steps"]):
            comparison = self._build_next_step_comparison_response()
            if comparison:
                return comparison
            return "Tell me your current task or goal, and I can compare two next-step options."

        if any(term in user_lower for term in ["what should i do next", "what do i do next", "what is the next step"]):
            next_step = self._build_next_step_response()
            if next_step:
                return next_step
            return "Tell me your current task or goal, and I can suggest a next step."

        if any(term in user_lower for term in ["what else could i do next", "what is another next step", "what is an alternative next step"]):
            alternative_next_step = self._build_alternative_next_step_response()
            if alternative_next_step:
                return alternative_next_step
            return "Tell me your current task or goal, and I can suggest an alternative next step."

        if any(term in user_lower for term in ["what is a second alternative next step", "what is another alternative after that", "what is a second option"]):
            secondary_alternative_next_step = self._build_secondary_alternative_next_step_response()
            if secondary_alternative_next_step:
                return secondary_alternative_next_step
            return "Tell me your current task or goal, and I can suggest a second alternative next step."

        if any(term in user_lower for term in ["どちらがよい", "どちらを選ぶべき", "主案と別案のどちら"]):
            choice = self._build_next_step_choice_response(language="ja")
            if choice:
                return choice
            return "今の作業や目標を教えてくれれば、どちらを先に選ぶべきか一緒に考えられます。"

        if any(term in user_lower for term in ["次の一歩の候補を見せて", "次の一歩の案を一覧で", "候補を全部見せて"]):
            options = self._build_next_step_options_response(language="ja")
            if options:
                return options
            return "今の作業や目標を教えてくれれば、次の一歩の候補を一覧で出せます。"

        if any(term in user_lower for term in ["次の一歩を要約して", "判断メモを出して", "次の一歩の判断を要約して"]):
            brief = self._build_next_step_decision_brief(language="ja")
            if brief:
                return brief
            return "今の作業や目標を教えてくれれば、次の一歩の判断を短く要約できます。"

        if any(term in user_lower for term in ["次の一歩をシミュレーションして", "候補を軽くシミュレーションして", "オフラインで比較して"]):
            simulation = self._build_next_step_simulation_response(language="ja")
            if simulation:
                return simulation
            return "今の作業や目標を教えてくれれば、次の一歩の候補を軽くシミュレーションできます。"

        if any(term in user_lower for term in ["候補に順位を付けて", "次の一歩を順位付けして", "優先順を見せて"]):
            ranked_options = self._build_ranked_next_step_options_response(language="ja")
            if ranked_options:
                return ranked_options
            return "今の作業や目標を教えてくれれば、次の一歩の候補を優先順で出せます。"

        if any(term in user_lower for term in ["次の一歩を比較して", "主案と別案", "次の一歩の比較"]):
            comparison = self._build_next_step_comparison_response(language="ja")
            if comparison:
                return comparison
            return "今の作業や目標を教えてくれれば、次の一歩を比較できます。"

        if any(term in user_lower for term in ["もう一つの別案", "第二の別案", "もう一つの代替案"]):
            secondary_alternative_next_step = self._build_secondary_alternative_next_step_response(language="ja")
            if secondary_alternative_next_step:
                return secondary_alternative_next_step
            return "今の作業や目標を教えてくれれば、もう一つの別案も一緒に考えられます。"

        if any(term in user_lower for term in ["別の次の一歩", "他に何をすればいい", "別案は何", "他の案は何"]):
            alternative_next_step = self._build_alternative_next_step_response(language="ja")
            if alternative_next_step:
                return alternative_next_step
            return "今の作業や目標を教えてくれれば、別案も一緒に考えられます。"

        if any(term in user_lower for term in ["次に何をすればいい", "次に何をしたらいい", "次の一歩は何"]):
            next_step = self._build_next_step_response(language="ja")
            if next_step:
                return next_step
            return "今の作業や目標を教えてくれれば、次の一歩を一緒に考えられます。"

        live_match = re.search(r"\bi live in ([a-z][a-z\s'-]{1,40})\b", user_lower)
        if live_match:
            location = live_match.group(1).strip(" .!?")
            return f"Thank you for telling me. In this conversation, I understand that you live in {location.title()}."

        name_match = re.search(r"\bmy name is ([a-z][a-z\s' -]{0,30})\b", user_lower)
        if name_match:
            name = name_match.group(1).strip(" .!?").title()
            return f"Thank you for telling me. In this conversation, I understand that your name is {name}."

        if "住んでいます" in user_text:
            return "ありがとうございます。この会話では、あなたが住んでいる場所の情報を文脈として使えます。"

        if "出身です" in user_text:
            return "ありがとうございます。この会話では、あなたの出身地の情報を文脈として使えます。"

        if re.search(r"\bi like ([a-z][a-z\s' -]{1,40})\b", user_lower):
            return "Thank you for telling me. In this conversation, I can use that preference as context."

        if "好きです" in user_text:
            return "ありがとうございます。この会話では、あなたの好みの情報を文脈として使えます。"

        if re.search(r"\bi want to ([a-z][a-z\s' -]{1,60})\b", user_lower):
            return "Thank you for telling me. In this conversation, I can use your goal as context."

        if re.search(r"\bi am working on ([a-z][a-z0-9\s' -]{1,60})\b", user_lower):
            return "Thank you for telling me. In this conversation, I can use your current task as context."

        if "したいです" in user_text:
            return "ありがとうございます。この会話では、あなたの目標を文脈として使えます。"

        if "をしています" in user_text:
            return "ありがとうございます。この会話では、あなたの今の作業を文脈として使えます。"

        return None

    def _capture_fast_path_diagnostic(self, prompt: str, response: str) -> None:
        self._ensure_runtime_state()
        user_text = self._extract_latest_user_text(prompt)
        memory_hit = "session_memory" if self._looks_like_session_memory_reply(prompt, response) else "fast_path"
        diagnostic = normalize_retrieval_diagnostic(
            {
                "content_preview": user_text[:80] if user_text else "fast_intent",
                "base_score": 1.0,
                "stability_score": 1.0,
                "suffix_match": 1.0,
                "drift_penalty": 0.0,
                "metadata_keyword_overlap": 1.0,
                "context_match": True,
                "role_match": True,
                "keyword_score": self._response_relevance_score(prompt, response),
                "memory_hit": memory_hit,
            },
            source="inference_fast_path",
            content_key="content_preview",
        )
        self.retrieval_diagnostics.append(diagnostic)
        self.retrieval_diagnostics = self.retrieval_diagnostics[-10:]

    def _build_goal_suggestion(self, language: str = "en") -> str:
        task = self.session_memory.get("task")
        task_hint = self._task_hint(language=language)
        if language == "ja":
            if task:
                if task_hint:
                    return task_hint
                return f"必要なら、まず{self._format_session_label_ja('task')}を終えるための次の一歩を一緒に整理できます。"
            return "必要なら、その目標を小さなステップに分けて一緒に整理できます。"
        if task:
            if task_hint:
                return task_hint
            future_state = self._build_future_state_label(language=language)
            if future_state:
                return f"I can help you turn that goal into the next step for {task}, with the future state of {future_state} in mind."
            return f"I can help you turn that goal into the next step for {task}."
        return "I can help you break that goal into smaller steps."

    def _build_task_suggestion(self, language: str = "en") -> str:
        goal = self.session_memory.get("goal")
        task_hint = self._task_hint(language=language)
        if language == "ja":
            if goal:
                if task_hint:
                    return task_hint
                return f"必要なら、{self._format_session_label_ja('goal')}につながる次の一歩を一緒に考えられます。"
            return "必要なら、次に何をするか一緒に考えられます。"
        if goal:
            if task_hint:
                return task_hint
            future_state = self._build_future_state_label(language=language)
            if future_state:
                return f"I can help you choose the next step that moves you toward {future_state}."
            return f"I can help you choose the next step that moves you toward {goal}."
        return "I can help you think through the next step."

    def _build_future_state_label(self, language: str = "en") -> str:
        goal = self.session_memory.get("goal")
        task = self.session_memory.get("task")

        if language == "ja":
            if goal:
                return self._format_session_label_ja("goal")
            if task:
                task_label = self._format_session_label_ja("task")
                if task_label:
                    return f"{task_label}を前に進めること"
            return ""

        if goal:
            return str(goal)
        if task:
            return f"making progress on {task}"
        return ""

    def _predict_lightweight_future_state(self, language: str = "en") -> dict[str, str | float]:
        task_value = self.session_memory.get("task") or ""
        future_state = self._build_future_state_label(language=language)
        task = task_value.lower()
        goal = str(self.session_memory.get("goal") or "").lower()
        combined = f"{task} {goal}".strip()
        category = "generic"
        action = ""

        if language == "ja":
            task_label = self._format_session_label_ja("task") if task_value else ""
            if any(marker in combined for marker in ["release", "deploy", "ship", "packaging", "publish", "リリース", "公開", "デプロイ", "出荷"]):
                category = "release"
                action = f"{task_label}で最初に確認するリリース確認項目を1つ決める"
            elif any(marker in task for marker in ["test", "pytest", "spec", "qa", "verification", "テスト", "検証", "確認"]):
                category = "testing"
                action = f"{task_label}で最初に確認する1つのケースを決める"
            elif any(marker in task for marker in ["debug", "fix", "error", "issue", "trace", "failure", "bugfix", "デバッグ", "不具合", "修正", "障害", "エラー"]):
                category = "debugging"
                action = f"{task_label}で一番小さく再現できる不具合を1つ決める"
            elif any(marker in task for marker in ["research", "paper", "study", "investigation", "survey", "調査", "研究", "論文", "リサーチ"]):
                category = "research"
                action = f"{task_label}で答えたい問いを1つに絞る"
            elif any(marker in task for marker in ["engine", "project", "code", "api", "feature", "bug"]):
                category = "development"
                action = f"{task_label}で直近の1つの変更点を決める"
            elif task_label:
                action = f"{task_label}を進めるための一番小さな未完了タスクを1つ決める"
            command_hint = self._predict_operational_command_hint()
            confidence = 1.0 if action and future_state else 0.6 if action or future_state else 0.0
            return {
                "category": category,
                "action": action,
                "target_state": future_state,
                "command": command_hint,
                "confidence": confidence,
            }

        if any(marker in combined for marker in ["release", "deploy", "ship", "packaging", "publish"]):
            category = "release"
            action = f"choose one release check to complete for {task_value}"
        elif any(marker in task for marker in ["test", "pytest", "spec", "qa", "verification"]):
            category = "testing"
            action = f"choose one concrete case to verify in {task_value}"
        elif any(marker in task for marker in ["debug", "fix", "error", "issue", "trace", "failure", "bugfix"]):
            category = "debugging"
            action = f"isolate one small reproducible failure in {task_value}"
        elif any(marker in task for marker in ["research", "paper", "study", "investigation", "survey"]):
            category = "research"
            action = f"narrow {task_value} to one question to answer"
        elif any(marker in task for marker in ["engine", "project", "code", "api", "feature", "bug"]):
            category = "development"
            action = f"choose one concrete change to make in {task_value}"
        elif task_value:
            action = f"choose one small unfinished action for {task_value}"
        command_hint = self._predict_operational_command_hint()
        confidence = 1.0 if action and future_state else 0.6 if action or future_state else 0.0
        return {
            "category": category,
            "action": action,
            "target_state": future_state,
            "command": command_hint,
            "confidence": confidence,
        }

    def _predict_future_state_transition(self, language: str = "en") -> dict[str, str]:
        prediction = self._predict_lightweight_future_state(language=language)
        return {
            "action": str(prediction.get("action", "")),
            "target_state": str(prediction.get("target_state", "")),
        }

    def _predict_counterfactual_future_state(self, language: str = "en") -> dict[str, str | float]:
        task_value = self.session_memory.get("task") or ""
        goal_value = self.session_memory.get("goal") or ""
        task = task_value.lower()
        goal = goal_value.lower()
        combined = f"{task} {goal}".strip()

        alternative_action = ""
        alternative_target_state = self._build_future_state_label(language=language)
        alternative_command = ""

        if any(marker in combined for marker in ["release", "deploy", "ship", "packaging", "publish", "リリース", "公開", "出荷"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「リリース準備」"
                alternative_action = f"{task_label}で影響が最も大きいリリース確認項目を1つ先に確認する"
            else:
                alternative_action = f"prioritize the highest-risk release check in {task_value}"
            alternative_command = "python scripts/eval/release_soak.py --include-accuracy"
        elif any(marker in combined for marker in ["research", "paper", "study", "investigation", "survey", "調査", "研究", "論文", "リサーチ"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「調査」"
                alternative_action = f"{task_label}で比較したい候補を2つだけ選ぶ"
            else:
                alternative_action = f"compare two candidate directions inside {task_value}"
            alternative_command = "python scripts/sara_cli.py db-list --category research --format json"
        elif any(marker in combined for marker in ["test", "pytest", "spec", "qa", "verification", "テスト", "検証", "確認"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「テスト確認」"
                alternative_action = f"{task_label}で最も壊れやすいケースを先に確認する"
            else:
                alternative_action = f"verify the highest-risk case first in {task_value}"
            alternative_command = "pytest -q"
        elif task_value:
            if language == "ja":
                task_label = self._format_session_label_ja("task")
                alternative_action = f"{task_label}で別の小さな進め方を1つ比較する"
            else:
                alternative_action = f"compare one alternative small action for {task_value}"
            alternative_command = self._predict_operational_command_hint()

        alternative_confidence = 0.5 if alternative_action and alternative_target_state else 0.0
        return {
            "action": alternative_action,
            "target_state": alternative_target_state,
            "command": alternative_command,
            "confidence": alternative_confidence,
        }

    def _predict_secondary_counterfactual_future_state(self, language: str = "en") -> dict[str, str | float]:
        task_value = self.session_memory.get("task") or ""
        goal_value = self.session_memory.get("goal") or ""
        task = task_value.lower()
        goal = goal_value.lower()
        combined = f"{task} {goal}".strip()

        alternative_action = ""
        alternative_target_state = self._build_future_state_label(language=language)
        alternative_command = ""

        if any(marker in combined for marker in ["release", "deploy", "ship", "packaging", "publish", "リリース", "公開", "出荷"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「リリース準備」"
                alternative_action = f"{task_label}でロールバック条件を1つ先に確認する"
            else:
                alternative_action = f"check one rollback condition first in {task_value}"
            alternative_command = "python scripts/eval/release_gate.py"
        elif any(marker in combined for marker in ["research", "paper", "study", "investigation", "survey", "調査", "研究", "論文", "リサーチ"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「調査」"
                alternative_action = f"{task_label}で反対側の材料を1つ探す"
            else:
                alternative_action = f"find one contradictory source inside {task_value}"
            alternative_command = "python scripts/sara_cli.py db-list --category research --format json"
        elif any(marker in combined for marker in ["test", "pytest", "spec", "qa", "verification", "テスト", "検証", "確認"]):
            if language == "ja":
                task_label = self._format_session_label_ja("task") if task_value else "「テスト確認」"
                alternative_action = f"{task_label}で最小のスモーク確認を先に1つ行う"
            else:
                alternative_action = f"run the smallest smoke check first in {task_value}"
            alternative_command = "pytest -q"
        elif task_value:
            if language == "ja":
                task_label = self._format_session_label_ja("task")
                alternative_action = f"{task_label}で依存の少ない小タスクを1つ先に終える"
            else:
                alternative_action = f"finish one low-dependency small action first for {task_value}"
            alternative_command = self._predict_operational_command_hint()

        alternative_confidence = 0.4 if alternative_action and alternative_target_state else 0.0
        return {
            "action": alternative_action,
            "target_state": alternative_target_state,
            "command": alternative_command,
            "confidence": alternative_confidence,
        }

    def _predict_operational_command_hint(self) -> str:
        task = (self.session_memory.get("task") or "").lower()
        goal = (self.session_memory.get("goal") or "").lower()
        combined = f"{task} {goal}".strip()

        if any(marker in combined for marker in ["release", "ship", "publish", "packaging", "deploy", "リリース", "公開", "出荷"]):
            return "python scripts/eval/release_soak.py --include-accuracy"
        if any(marker in combined for marker in ["test", "pytest", "spec", "qa", "verification", "テスト", "検証", "確認"]):
            return "pytest -q"
        if any(marker in combined for marker in ["research", "paper", "study", "investigation", "survey", "調査", "研究", "論文", "リサーチ"]):
            return "python scripts/sara_cli.py db-list --category research --format json"
        if any(marker in combined for marker in ["dataset", "material", "corpus", "export", "素材", "データ", "コーパス"]):
            return "python scripts/sara_cli.py db-export --dry-run"
        if any(marker in combined for marker in ["debug", "fix", "error", "issue", "bug", "デバッグ", "不具合", "修正", "障害", "エラー"]):
            return "pytest -q"
        return ""

    def _append_operational_command_hint(self, response: str, language: str = "en") -> str:
        command_hint = self._predict_operational_command_hint()
        if not response or not command_hint:
            return response
        if language == "ja":
            return f"{response} 提案コマンド: `{command_hint}`"
        return f"{response} Suggested command: `{command_hint}`"

    def _append_future_state_shift_note(self, response: str, language: str = "en") -> str:
        shift_note = self._describe_future_state_shift(language=language)
        if not response or not shift_note:
            return response
        return f"{response} {shift_note}"

    def _apply_adapted_response_mode(self, response: str, language: str = "en") -> str:
        if not response:
            return response
        if str(self._get_adaptation_state().get("response_mode", "")) != "directive":
            return response
        if language == "ja":
            return f"まずこれを進めましょう: {response}"
        return f"Do this now: {response}"

    def _build_next_step_response(self, language: str = "en") -> str:
        task = self.session_memory.get("task")
        goal = self.session_memory.get("goal")
        task_hint = self._task_hint(language=language, compact=True)
        prediction = self._predict_lightweight_future_state(language=language)
        future_state = str(prediction.get("target_state", ""))
        transition = {
            "action": str(prediction.get("action", "")),
            "target_state": future_state,
        }

        if task_hint:
            return self._append_operational_command_hint(
                self._apply_adapted_response_mode(
                    self._append_future_state_shift_note(task_hint, language=language),
                    language=language,
                ),
                language=language,
            )

        if language == "ja":
            task_label = self._format_session_label_ja("task") if task else ""
            goal_label = future_state if future_state else self._format_session_label_ja("goal")
            if task and goal:
                return self._apply_adapted_response_mode(
                    self._append_future_state_shift_note(
                        f"次の一歩として、まず{task_label}を進めるために一番小さな未完了タスクを1つ決めるのがよいです。そうすると{goal_label}に近づけます。",
                        language=language,
                    ),
                    language=language,
                )
            if task:
                if future_state:
                    action = transition.get("action", "")
                    if action:
                        return self._append_operational_command_hint(
                            self._apply_adapted_response_mode(
                                self._append_future_state_shift_note(
                                    f"次の一歩として、まず{action}のがよいです。そうすると{future_state}につながります。",
                                    language=language,
                                ),
                                language=language,
                            ),
                            language=language,
                        )
                    return self._append_operational_command_hint(
                        self._apply_adapted_response_mode(
                            self._append_future_state_shift_note(
                                f"次の一歩として、まず{task_label}を進めるための一番小さな未完了タスクを1つ決めるのがよいです。そうすると{future_state}につながります。",
                                language=language,
                            ),
                            language=language,
                        ),
                        language=language,
                    )
                return self._append_operational_command_hint(
                    self._apply_adapted_response_mode(
                        self._append_future_state_shift_note(
                            f"次の一歩として、まず{task_label}を進めるための一番小さな未完了タスクを1つ決めるのがよいです。",
                            language=language,
                        ),
                        language=language,
                    ),
                    language=language,
                )
            if goal:
                return self._append_operational_command_hint(
                    self._apply_adapted_response_mode(
                        self._append_future_state_shift_note(
                            f"次の一歩として、{goal_label}に向けた最小の作業を1つ決めるのがよいです。",
                            language=language,
                        ),
                        language=language,
                    ),
                    language=language,
                )
            return ""

        if task and goal:
            return self._append_operational_command_hint(
                self._apply_adapted_response_mode(
                    self._append_future_state_shift_note(
                        (
                        f"The next step is to choose one small unfinished action for {task}. "
                        f"That will move you toward {future_state or goal}."
                        ),
                        language=language,
                    ),
                    language=language,
                ),
                language=language,
            )
        if task:
            if future_state:
                action = transition.get("action", "")
                if action:
                    return self._append_operational_command_hint(
                        self._apply_adapted_response_mode(
                            self._append_future_state_shift_note(
                                f"The next step is to {action}. That will support the future state of {future_state}.",
                                language=language,
                            ),
                            language=language,
                        ),
                        language=language,
                    )
                return self._append_operational_command_hint(
                    self._apply_adapted_response_mode(
                        self._append_future_state_shift_note(
                            (
                            f"The next step is to choose one small unfinished action for {task}. "
                            f"That will support the future state of {future_state}."
                            ),
                            language=language,
                        ),
                        language=language,
                    ),
                    language=language,
                )
            return self._append_operational_command_hint(
                self._apply_adapted_response_mode(
                    self._append_future_state_shift_note(
                        f"The next step is to choose one small unfinished action for {task}.",
                        language=language,
                    ),
                    language=language,
                ),
                language=language,
            )
        if goal:
            return self._append_operational_command_hint(
                self._apply_adapted_response_mode(
                    self._append_future_state_shift_note(
                        f"The next step is to choose one small concrete action that moves you toward {future_state or goal}.",
                        language=language,
                    ),
                    language=language,
                ),
                language=language,
            )
        return ""

    def _task_hint(self, language: str = "en", compact: bool = False) -> str:
        task = (self.session_memory.get("task") or "").lower()
        goal = self.session_memory.get("goal") or ""
        task_value_ja = self._format_session_label_ja("task")
        goal_value_ja = self._format_session_label_ja("goal")

        if any(marker in task for marker in ["test", "pytest", "spec", "qa", "verification", "テスト", "検証", "確認"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}で最初に確認する1つのケースを決めます。 Step 2: その結果が{goal_value_ja}に近づくか確認します。"
                    return f"Step 1: {task_value_ja}で最初に確認する1つのケースを決めます。 Step 2: そのケースを実際に確認します。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}で最初に確認する1つのケースを決めるのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}で最初に確認する1つのケースを決めるのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: choose one concrete case to verify in {self.session_memory.get('task')}. Step 2: run it and check that it moves you toward {goal}."
                return f"Step 1: choose one concrete case to verify in {self.session_memory.get('task')}. Step 2: run that check."
            if goal:
                return f"The next step is to choose one concrete case to verify in {self.session_memory.get('task')}. That will move you toward {goal}."
            return f"The next step is to choose one concrete case to verify in {self.session_memory.get('task')}."

        if any(marker in task for marker in ["debug", "fix", "error", "issue", "trace", "failure", "bugfix", "デバッグ", "不具合", "修正", "障害", "エラー"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}で一番小さく再現できる不具合を1つ決めます。 Step 2: その原因を確認して{goal_value_ja}に近づく修正を進めます。"
                    return f"Step 1: {task_value_ja}で一番小さく再現できる不具合を1つ決めます。 Step 2: その原因を確認します。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}で一番小さく再現できる不具合を1つ決めるのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}で一番小さく再現できる不具合を1つ決めるのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: isolate one small reproducible failure in {self.session_memory.get('task')}. Step 2: confirm the cause and check that the fix moves you toward {goal}."
                return f"Step 1: isolate one small reproducible failure in {self.session_memory.get('task')}. Step 2: confirm the cause."
            if goal:
                return f"The next step is to isolate one small reproducible failure in {self.session_memory.get('task')}. That will help move you toward {goal}."
            return f"The next step is to isolate one small reproducible failure in {self.session_memory.get('task')}."

        if any(marker in task for marker in ["research", "paper", "study", "investigation", "survey", "調査", "研究", "論文", "リサーチ"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}で答えたい問いを1つに絞ります。 Step 2: その問いへの材料を集めて{goal_value_ja}に近づくか確認します。"
                    return f"Step 1: {task_value_ja}で答えたい問いを1つに絞ります。 Step 2: その問いへの材料を集めます。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}で答えたい問いを1つに絞るのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}で答えたい問いを1つに絞るのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: narrow {self.session_memory.get('task')} to one question to answer. Step 2: gather the material that moves you toward {goal}."
                return f"Step 1: narrow {self.session_memory.get('task')} to one question to answer. Step 2: gather material for that question."
            if goal:
                return f"The next step is to narrow {self.session_memory.get('task')} to one question to answer. That will move you toward {goal}."
            return f"The next step is to narrow {self.session_memory.get('task')} to one question to answer."

        if any(marker in task for marker in ["release", "deploy", "ship", "packaging", "publish", "リリース", "公開", "デプロイ", "出荷"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}で最初に確認するリリース確認項目を1つ決めます。 Step 2: それを終えて{goal_value_ja}に近づくか確認します。"
                    return f"Step 1: {task_value_ja}で最初に確認するリリース確認項目を1つ決めます。 Step 2: その項目を完了します。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}で最初に確認するリリース確認項目を1つ決めるのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}で最初に確認するリリース確認項目を1つ決めるのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: choose one release check to complete for {self.session_memory.get('task')}. Step 2: finish it and check that it moves you toward {goal}."
                return f"Step 1: choose one release check to complete for {self.session_memory.get('task')}. Step 2: finish that check."
            if goal:
                return f"The next step is to choose one release check to complete for {self.session_memory.get('task')}. That will move you toward {goal}."
            return f"The next step is to choose one release check to complete for {self.session_memory.get('task')}."

        if any(marker in task for marker in ["engine", "project", "code", "api", "feature", "bug"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}で直近の1つの変更点を決めます。 Step 2: それを終えて{goal_value_ja}に近づくか確認します。"
                    return f"Step 1: {task_value_ja}で直近の1つの変更点を決めます。 Step 2: その変更を終えるところまで進めます。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}で直近の1つの変更点を決めるのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}で直近の1つの変更点を決めるのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: choose one concrete change to make in {self.session_memory.get('task')}. Step 2: finish it and check that it moves you toward {goal}."
                return f"Step 1: choose one concrete change to make in {self.session_memory.get('task')}. Step 2: finish that change."
            if goal:
                return f"The next step is to choose one concrete change to make in {self.session_memory.get('task')}. That will move you toward {goal}."
            return f"The next step is to choose one concrete change to make in {self.session_memory.get('task')}."

        if any(marker in task for marker in ["write", "article", "blog", "document", "draft", "essay"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}の最初の見出しか最初の段落を書きます。 Step 2: それが{goal_value_ja}に近づく内容か確認します。"
                    return f"Step 1: {task_value_ja}の最初の見出しか最初の段落を書きます。 Step 2: その方向で下書きを広げます。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}の最初の見出しか最初の段落を書くのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}の最初の見出しか最初の段落を書くのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: draft the first heading or paragraph for {self.session_memory.get('task')}. Step 2: check that it moves you toward {goal}."
                return f"Step 1: draft the first heading or paragraph for {self.session_memory.get('task')}. Step 2: extend that draft."
            if goal:
                return f"The next step is to draft the first heading or paragraph for {self.session_memory.get('task')}. That will move you toward {goal}."
            return f"The next step is to draft the first heading or paragraph for {self.session_memory.get('task')}."

        if any(marker in task for marker in ["illustration", "draw", "art", "design", "sketch"]):
            if language == "ja":
                if compact:
                    if goal:
                        return f"Step 1: {task_value_ja}のラフを1つ作ります。 Step 2: それが{goal_value_ja}に近づく方向か確認します。"
                    return f"Step 1: {task_value_ja}のラフを1つ作ります。 Step 2: 一番良い案を選びます。"
                if goal:
                    return f"次の一歩として、まず{task_value_ja}のラフを1つ作るのがよいです。そうすると{goal_value_ja}に近づけます。"
                return f"次の一歩として、まず{task_value_ja}のラフを1つ作るのがよいです。"
            if compact:
                if goal:
                    return f"Step 1: make one rough sketch for {self.session_memory.get('task')}. Step 2: check that it moves you toward {goal}."
                return f"Step 1: make one rough sketch for {self.session_memory.get('task')}. Step 2: pick the strongest direction."
            if goal:
                return f"The next step is to make one rough sketch for {self.session_memory.get('task')}. That will move you toward {goal}."
            return f"The next step is to make one rough sketch for {self.session_memory.get('task')}."

        return ""

    def _looks_like_session_memory_reply(self, prompt: str, response: str) -> bool:
        user_text = self._extract_latest_user_text(prompt)
        if not user_text or not response:
            return False
        user_lower = user_text.lower()
        if not any(
            marker in user_lower or marker in user_text
            for marker in [
                "remember",
                "where do i live",
                "what city do i live in",
                "what is my name",
                "what do i like",
                "what is my goal",
                "what am i working on",
                "what should i do next",
                "what do i do next",
                "what is the next step",
                "覚えていますか",
                "どこに住んで",
                "名前は何",
                "何が好き",
                "目標は何",
                "何をしている",
                "次に何をすればいい",
                "次に何をしたらいい",
                "次の一歩",
            ]
        ):
            return False
        text = f"{user_text}\n{response}"
        return any(
            marker in text
            for marker in [
                "remember",
                "name",
                "live",
                "like",
                "goal",
                "working on",
                "next step",
                "Step 1:",
                "Step 2:",
                "concrete change",
                "名前",
                "住んで",
                "好き",
                "目標",
                "作業",
                "次の一歩",
                "変更点",
            ]
        )

    def _response_relevance_score(self, prompt: str, response: str) -> float:
        if not response.strip():
            return 0.0
        prompt_keywords = self._extract_prompt_keywords(prompt)
        response_lower = response.lower()
        overlap = sum(1 for keyword in prompt_keywords if keyword in response_lower)
        keyword_score = overlap / max(1, len(prompt_keywords)) if prompt_keywords else 0.0

        citation_like = 1.0 if re.search(r"\d{4}年|\(\d{4}\)|閲覧|オリジナル|et al\.|pp?\.\d", response) else 0.0
        quoted_definition = 1.0 if response.strip().startswith('"') else 0.0
        short_greeting = 1.0 if len(response.strip()) <= 24 else 0.0
        prompt_lower = prompt.lower()

        intent_score = 0.0
        if any(term in prompt_lower for term in ["who are you", "あなたは誰", "君は誰"]):
            if any(term in response_lower for term in ["i am sara", "私はsara", "i'm sara"]):
                intent_score = 1.0
        elif any(term in prompt_lower for term in ["hello", "hi", "こんにちは", "こんばんは"]):
            if any(term in response_lower for term in ["hello", "how can i help", "こんにちは"]):
                intent_score = 1.0
        elif any(term in prompt_lower for term in ["do you have", "持っていますか", "ありますか"]):
            if any(term in response_lower for term in ["software agent", "do not have physical", "physical objects"]):
                intent_score = 1.0

        adaptation_state = self._get_adaptation_state()
        planning_confidence = float(adaptation_state.get("planning_confidence", 0.0) or 0.0)
        memory_weight = float(adaptation_state.get("memory_weight", 1.0) or 1.0)
        memory_requests = int(adaptation_state.get("memory_requests", 0) or 0)
        next_step_requests = int(adaptation_state.get("next_step_requests", 0) or 0)
        adaptation_bonus = 0.0
        if self._looks_like_session_memory_reply(prompt, response):
            adaptation_bonus += min(
                0.24,
                (0.04 * memory_requests + 0.06 * next_step_requests) * max(1.0, memory_weight),
            )
            adaptation_bonus += min(0.12, planning_confidence * 0.12)

        score = keyword_score
        score += 0.2 if short_greeting and any(term in prompt_lower for term in ["hello", "hi", "こんにちは"]) else 0.0
        score += intent_score * 0.65
        score += adaptation_bonus
        score -= citation_like * 0.45
        score -= quoted_definition * 0.1
        return max(0.0, min(1.0, score))

    def _practical_fallback_for_prompt(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(term in prompt_lower for term in ["who are you", "あなたは誰", "君は誰"]):
            return "I am SARA, a CPU-first spiking neural network assistant."
        if any(term in prompt_lower for term in ["hello", "hi", "こんにちは", "こんばんは"]):
            return "Hello. I am SARA. How can I help you?"
        if any(term in prompt_lower for term in ["do you have", "持っていますか", "ありますか"]):
            return "I am a software agent, so I do not have physical objects, but I can help with information."
        return "I do not have a reliable answer yet. Please ask in a more specific way."

    def _apply_practical_relevance_gate(self, prompt: str, response: str) -> str:
        relevance = self._response_relevance_score(prompt, response)
        prompt_lower = prompt.lower()
        adaptation_state = self._get_adaptation_state()
        planning_confidence = float(adaptation_state.get("planning_confidence", 0.0) or 0.0)
        fallback_relaxation = float(adaptation_state.get("fallback_relaxation", 0.0) or 0.0)
        response_mode = str(adaptation_state.get("response_mode", ""))
        memory_like = self._looks_like_session_memory_reply(prompt, response)
        relevance_floor = 0.18
        if memory_like and response_mode in {"guided", "directive"}:
            relevance_floor = max(
                0.06,
                relevance_floor
                - min(0.08, planning_confidence * 0.08)
                - min(0.06, fallback_relaxation),
            )

        should_gate = relevance < relevance_floor
        should_gate = should_gate or (
            relevance < max(0.14, 0.28 - (0.05 if memory_like else 0.0) - min(0.04, fallback_relaxation))
            and any(term in prompt_lower for term in ["who are you", "hello", "hi", "do you have", "こんにちは", "ありますか"])
        )
        if should_gate:
            return self._practical_fallback_for_prompt(prompt)
        return response

    def reset_state(self):
        self.refractory_buffer = []
        if self.lif_network:
            self.lif_network.reset()

    def get_recent_retrieval_diagnostics(self, limit: int = 3) -> List[Dict[str, object]]:
        self._ensure_runtime_state()
        if limit <= 0:
            return []
        return self.retrieval_diagnostics[-limit:]

    def format_recent_retrieval_diagnostics(self, limit: int = 3) -> str:
        return format_retrieval_diagnostics(self.get_recent_retrieval_diagnostics(limit=limit))

    def _capture_matching_diagnostic(self, payload: Dict[str, object]) -> None:
        self._ensure_runtime_state()
        preview = payload.get("content_preview", "")
        if isinstance(preview, str) and preview.strip() and hasattr(self, "tokenizer"):
            try:
                preview_tokens = [int(item) for item in preview.split()]
                decoded_preview = self.tokenizer.decode(preview_tokens)
                if isinstance(decoded_preview, str) and decoded_preview.strip():
                    payload = dict(payload)
                    payload["content_preview"] = decoded_preview.strip()
            except (TypeError, ValueError):
                pass
        normalized = normalize_retrieval_diagnostic(
            payload,
            source="inference_direct_map",
            content_key="content_preview",
        )
        self.retrieval_diagnostics.append(normalized)
        self.retrieval_diagnostics = self.retrieval_diagnostics[-10:]

    # Backward-compatible alias used by older CLI/example code.
    def reset_buffer(self):
        self.reset_state()

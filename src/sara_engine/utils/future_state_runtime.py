from typing import Dict, Mapping


class LightweightFutureStateRuntime:
    """
    Tracks short-horizon future-state predictor transitions without relying on
    heavyweight latent models. The runtime state is intentionally lightweight
    and non-persistent so it can act as a bridge toward richer predictors.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._transition_count = 0
        self._stable_transition_count = 0
        self._shift_count = 0
        self._operator_match_count = 0
        self._speculative_acceptance_count = 0
        self._rollback_observable_count = 0
        self._counterfactual_viable_count = 0
        self._rewarded_selection_count = 0
        self._policy_stable_count = 0
        self._energy_aware_preference_count = 0
        self._previous_category = ""
        self._previous_target_state = ""
        self._last_shift_from = ""
        self._last_shift_to = ""
        self._last_category = ""
        self._last_target_state = ""
        self._last_language = ""
        self._last_transition_operator = ""
        self._last_verified_operator = ""
        self._last_operator_match = False
        self._last_speculative_acceptance = False
        self._last_rollback_observable = False
        self._last_counterfactual_viable = False
        self._last_branch_count = 0
        self._last_branch_labels: list[str] = []
        self._last_preferred_branch = ""
        self._last_simulated_branch_count = 0
        self._last_best_simulated_branch = ""
        self._last_best_simulation_score = 0.0
        self._last_reward_trace: Dict[str, object] = {}
        self._last_policy_trace: Dict[str, object] = {}

    def advance(
        self,
        prediction: Mapping[str, object],
        language: str = "en",
    ) -> Dict[str, object]:
        category = str(prediction.get("category", ""))
        target_state = str(prediction.get("target_state", ""))
        action = str(prediction.get("action", ""))
        command = str(prediction.get("command", ""))
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        branch_candidates = prediction.get("branch_candidates", [])
        simulated_branch_candidates = prediction.get("simulated_branch_candidates", [])
        speculative_trace = prediction.get("speculative_trace", {})
        reward_trace = prediction.get("reward_trace", {})
        policy_trace = prediction.get("policy_trace", {})

        branch_labels: list[str] = []
        if isinstance(branch_candidates, list):
            for item in branch_candidates:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("label", "")).strip()
                if label:
                    branch_labels.append(label)
        preferred_branch = str(prediction.get("preferred_branch", "")).strip()
        best_simulated_branch = str(prediction.get("best_simulated_branch", "")).strip()
        best_simulation_score = 0.0
        if isinstance(simulated_branch_candidates, list):
            for item in simulated_branch_candidates:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("label", "")).strip() != best_simulated_branch:
                    continue
                best_simulation_score = float(item.get("simulation_score", 0.0) or 0.0)
                break

        transition_operator = ""
        verified_operator = ""
        operator_match = False
        speculative_acceptance = False
        rollback_observable = False
        counterfactual_branch_viable = False
        if isinstance(speculative_trace, Mapping):
            transition_operator = str(speculative_trace.get("predicted_operator", "")).strip()
            verified_operator = str(speculative_trace.get("verified_operator", "")).strip()
            operator_match = bool(
                speculative_trace.get("operator_match", False)
                or (
                    transition_operator
                    and verified_operator
                    and transition_operator == verified_operator
                )
            )
            speculative_acceptance = bool(
                speculative_trace.get("draft_verify_accepted", False)
            )
            rollback_observable = bool(
                speculative_trace.get("rollback_observable", False)
            )
            counterfactual_branch_viable = bool(
                speculative_trace.get("counterfactual_branch_viable", False)
            )
        reward_signal = 0.0
        energy_cost_proxy = 1.0
        rewarded_selection = False
        energy_aware_preference = False
        normalized_reward_trace: Dict[str, object] = {}
        if isinstance(reward_trace, Mapping):
            reward_signal = float(reward_trace.get("total_reward", 0.0) or 0.0)
            energy_cost_proxy = float(reward_trace.get("energy_cost_proxy", 1.0) or 1.0)
            rewarded_selection = bool(reward_signal >= 0.55)
            energy_aware_preference = bool(
                rewarded_selection and energy_cost_proxy <= 0.45
            )
            normalized_reward_trace = {
                "progress_score": float(reward_trace.get("progress_score", 0.0) or 0.0),
                "risk_reduction_score": float(reward_trace.get("risk_reduction_score", 0.0) or 0.0),
                "reversibility_score": float(reward_trace.get("reversibility_score", 0.0) or 0.0),
                "energy_cost_proxy": float(energy_cost_proxy),
                "user_feedback_signal": float(reward_trace.get("user_feedback_signal", 0.0) or 0.0),
                "total_reward": float(reward_signal),
                "selected_branch": str(reward_trace.get("selected_branch", "")),
            }

        policy_stable = False
        normalized_policy_trace: Dict[str, object] = {}
        if isinstance(policy_trace, Mapping):
            policy_shift_applied = bool(policy_trace.get("policy_shift_applied", False))
            policy_stability = float(policy_trace.get("policy_stability", 0.0) or 0.0)
            policy_stable = bool((not policy_shift_applied) or policy_stability >= 0.75)
            normalized_policy_trace = {
                "selected_branch": str(policy_trace.get("selected_branch", "")),
                "best_simulated_branch": str(policy_trace.get("best_simulated_branch", "")),
                "policy_shift_applied": bool(policy_shift_applied),
                "policy_stability": float(policy_stability),
                "selection_consistent": bool(policy_trace.get("selection_consistent", False)),
            }

        if not action and not target_state and not command and confidence <= 0.0:
            self.reset()
            return {}

        previous_category = self._last_category
        previous_target_state = self._last_target_state
        self._transition_count += 1
        if (
            self._last_category
            and self._last_target_state
            and self._last_category == category
            and self._last_target_state == target_state
        ):
            self._stable_transition_count += 1
        elif (
            self._last_target_state
            and self._last_target_state != target_state
        ):
            self._shift_count += 1
            self._last_shift_from = self._last_target_state
            self._last_shift_to = target_state

        if operator_match:
            self._operator_match_count += 1
        if speculative_acceptance:
            self._speculative_acceptance_count += 1
        if rollback_observable:
            self._rollback_observable_count += 1
        if counterfactual_branch_viable:
            self._counterfactual_viable_count += 1
        if rewarded_selection:
            self._rewarded_selection_count += 1
        if policy_stable:
            self._policy_stable_count += 1
        if energy_aware_preference:
            self._energy_aware_preference_count += 1

        self._previous_category = previous_category
        self._previous_target_state = previous_target_state
        self._last_category = category
        self._last_target_state = target_state
        self._last_language = language
        self._last_transition_operator = transition_operator
        self._last_verified_operator = verified_operator
        self._last_operator_match = operator_match
        self._last_speculative_acceptance = speculative_acceptance
        self._last_rollback_observable = rollback_observable
        self._last_counterfactual_viable = counterfactual_branch_viable
        self._last_branch_count = len(branch_labels)
        self._last_branch_labels = list(branch_labels)
        self._last_preferred_branch = preferred_branch
        self._last_simulated_branch_count = len(simulated_branch_candidates) if isinstance(simulated_branch_candidates, list) else 0
        self._last_best_simulated_branch = best_simulated_branch
        self._last_best_simulation_score = float(best_simulation_score)
        self._last_reward_trace = normalized_reward_trace
        self._last_policy_trace = normalized_policy_trace

        denominator = max(self._transition_count - 1, 1)
        stability_ratio = self._stable_transition_count / denominator
        if self._transition_count == 1:
            stability_ratio = 1.0
        operator_consistency_ratio = self._operator_match_count / max(self._transition_count, 1)
        speculative_acceptance_ratio = self._speculative_acceptance_count / max(self._transition_count, 1)
        speculative_rollback_ratio = self._rollback_observable_count / max(self._transition_count, 1)
        counterfactual_viability_ratio = self._counterfactual_viable_count / max(self._transition_count, 1)
        rewarded_selection_ratio = self._rewarded_selection_count / max(self._transition_count, 1)
        policy_stability_ratio = self._policy_stable_count / max(self._transition_count, 1)
        energy_aware_preference_ratio = self._energy_aware_preference_count / max(self._transition_count, 1)

        return {
            "transition_count": self._transition_count,
            "stable_transition_count": self._stable_transition_count,
            "shift_count": self._shift_count,
            "stability_ratio": float(stability_ratio),
            "operator_consistency_ratio": float(operator_consistency_ratio),
            "speculative_acceptance_ratio": float(speculative_acceptance_ratio),
            "speculative_rollback_ratio": float(speculative_rollback_ratio),
            "counterfactual_viability_ratio": float(counterfactual_viability_ratio),
            "rewarded_selection_ratio": float(rewarded_selection_ratio),
            "policy_stability_ratio": float(policy_stability_ratio),
            "energy_aware_preference_ratio": float(energy_aware_preference_ratio),
            "previous_category": previous_category,
            "previous_target_state": previous_target_state,
            "last_shift_from": self._last_shift_from,
            "last_shift_to": self._last_shift_to,
            "last_category": category,
            "last_target_state": target_state,
            "last_language": language,
            "last_transition_operator": self._last_transition_operator,
            "last_verified_operator": self._last_verified_operator,
            "last_operator_match": self._last_operator_match,
            "last_speculative_acceptance": self._last_speculative_acceptance,
            "last_rollback_observable": self._last_rollback_observable,
            "last_counterfactual_viable": self._last_counterfactual_viable,
            "last_branch_count": self._last_branch_count,
            "last_branch_labels": list(self._last_branch_labels),
            "last_preferred_branch": self._last_preferred_branch,
            "last_simulated_branch_count": self._last_simulated_branch_count,
            "last_best_simulated_branch": self._last_best_simulated_branch,
            "last_best_simulation_score": float(self._last_best_simulation_score),
            "last_reward_trace": dict(self._last_reward_trace),
            "last_policy_trace": dict(self._last_policy_trace),
        }

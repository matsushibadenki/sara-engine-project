from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class ComplianceDecision:
    allowed: bool
    level: str
    reason: str


class SourceCompliance:
    """Simple domain-level compliance gate with conservative defaults."""

    def __init__(self, policy_path: str = "") -> None:
        # Conservative, explicit policy examples.
        self.domain_rules: dict[str, tuple[bool, str, str]] = {
            "en.wikipedia.org": (True, "allow", "cc-by-sa_reference"),
            "ja.wikipedia.org": (True, "allow", "cc-by-sa_reference"),
            "commons.wikimedia.org": (True, "allow", "open_media_repository"),
            "arxiv.org": (True, "warn", "mixed_license_preprint"),
            "librivox.org": (True, "allow", "public_domain_audio"),
            "freemusicarchive.org": (True, "warn", "mixed_audio_license"),
            "pexels.com": (True, "warn", "license_terms_check_required"),
        }
        self.default_rule: tuple[bool, str, str] = (
            True,
            "warn",
            "unknown_domain_manual_policy_recommended",
        )
        self.source_type_rules: dict[str, tuple[bool, str, str]] = {}
        if policy_path:
            self.load_policy(policy_path)

    def load_policy(self, policy_path: str) -> bool:
        if not policy_path or not os.path.exists(policy_path):
            return False
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return False
            default_obj = raw.get("default", {})
            if isinstance(default_obj, dict):
                self.default_rule = (
                    bool(default_obj.get("allowed", True)),
                    str(default_obj.get("level", "warn")),
                    str(default_obj.get("reason", "unknown_domain_manual_policy_recommended")),
                )

            domains = raw.get("domains", {})
            if isinstance(domains, dict):
                loaded: dict[str, tuple[bool, str, str]] = {}
                for k, v in domains.items():
                    if not isinstance(v, dict):
                        continue
                    loaded[str(k).strip().lower()] = (
                        bool(v.get("allowed", True)),
                        str(v.get("level", "warn")),
                        str(v.get("reason", "policy_defined")),
                    )
                if loaded:
                    self.domain_rules = loaded

            by_source = raw.get("source_types", {})
            if isinstance(by_source, dict):
                loaded_source: dict[str, tuple[bool, str, str]] = {}
                for k, v in by_source.items():
                    if not isinstance(v, dict):
                        continue
                    loaded_source[str(k).strip().lower()] = (
                        bool(v.get("allowed", True)),
                        str(v.get("level", "warn")),
                        str(v.get("reason", "source_policy_defined")),
                    )
                self.source_type_rules = loaded_source
            return True
        except Exception:
            return False

    def apply_preset(self, preset: str) -> None:
        mode = (preset or "balanced").strip().lower()
        if mode == "strict":
            self.default_rule = (False, "deny", "strict_default_deny")
            self.domain_rules = {
                "en.wikipedia.org": (True, "allow", "cc-by-sa_reference"),
                "ja.wikipedia.org": (True, "allow", "cc-by-sa_reference"),
                "commons.wikimedia.org": (True, "allow", "open_media_repository"),
                "arxiv.org": (True, "warn", "mixed_license_preprint"),
            }
            self.source_type_rules = {
                "web": (True, "warn", "strict_web_warn"),
                "hot_inbox": (True, "allow", "trusted_local_ingest"),
            }
            return
        if mode == "open":
            self.default_rule = (True, "warn", "open_default_warn")
            self.source_type_rules = {
                "web": (True, "warn", "open_web_warn"),
                "hot_inbox": (True, "allow", "trusted_local_ingest"),
            }
            return
        # balanced
        self.default_rule = (True, "warn", "unknown_domain_manual_policy_recommended")
        self.source_type_rules = {
            "web": (True, "warn", "balanced_web_warn"),
            "hot_inbox": (True, "allow", "trusted_local_ingest"),
        }

    def decide(self, domain: str) -> ComplianceDecision:
        key = (domain or "").strip().lower()
        if key in self.domain_rules:
            allowed, level, reason = self.domain_rules[key]
            return ComplianceDecision(allowed=allowed, level=level, reason=reason)

        # Unknown domains are allowed in warning mode by default.
        allowed, level, reason = self.default_rule
        return ComplianceDecision(allowed=allowed, level=level, reason=reason)

    def decide_for_source(self, domain: str, source_type: str) -> ComplianceDecision:
        src = (source_type or "").strip().lower()
        if src in self.source_type_rules:
            allowed, level, reason = self.source_type_rules[src]
            if not allowed:
                return ComplianceDecision(allowed=False, level=level, reason=reason)
            domain_decision = self.decide(domain)
            if not domain_decision.allowed:
                return domain_decision
            if domain_decision.level == "allow":
                return ComplianceDecision(allowed=True, level=level, reason=reason)
            return domain_decision
        return self.decide(domain)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


def _bounded_strings(values: Iterable[str], limit: int = 16) -> Tuple[str, ...]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return tuple(cleaned[: max(1, int(limit))])


@dataclass(frozen=True)
class ProposalLineageLedgerEntry:
    record_id: str
    record_type: str
    source_ref: str
    source_hash: str
    extractor_name: str
    extractor_version: str
    parent_ids: Tuple[str, ...] = ()
    observed_anchor_ids: Tuple[str, ...] = ()
    proposal_model: str = ""
    proposal_config_hash: str = ""
    schema: str = "sara-proposal-lineage-ledger-entry-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "parent_ids": list(self.parent_ids),
            "observed_anchor_ids": list(self.observed_anchor_ids),
            "proposal_model": self.proposal_model,
            "proposal_config_hash": self.proposal_config_hash,
        }


def build_lineage_ledger_entry(payload: Dict[str, Any]) -> ProposalLineageLedgerEntry:
    return ProposalLineageLedgerEntry(
        record_id=str(payload.get("record_id", "") or ""),
        record_type=str(payload.get("record_type", "") or ""),
        source_ref=str(payload.get("source_ref", "") or ""),
        source_hash=str(payload.get("source_hash", "") or ""),
        extractor_name=str(payload.get("extractor_name", "") or ""),
        extractor_version=str(payload.get("extractor_version", "") or ""),
        parent_ids=_bounded_strings(payload.get("parent_ids", ()) or ()),
        observed_anchor_ids=_bounded_strings(payload.get("observed_anchor_ids", ()) or ()),
        proposal_model=str(payload.get("proposal_model", "") or ""),
        proposal_config_hash=str(payload.get("proposal_config_hash", "") or ""),
    )


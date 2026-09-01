from __future__ import annotations
import copy,json
from pathlib import Path
import pytest
from sara_engine.evaluation.phase30_preregistration import build_registered_manifest,compare_existing_registration,validate_preregistration
ROOT=Path(__file__).resolve().parents[1];DRAFT=ROOT/"workspace"/"evaluation"/"phase30_temporal_effective_interaction_preregistration_draft.json"
def _draft():return json.loads(DRAFT.read_text(encoding="utf-8"))
def test_phase30_complete_protocol_is_valid_and_fingerprinted():
    manifest=build_registered_manifest(_draft(),managed_path=True);assert validate_preregistration(manifest,managed_path=True)=={"valid":True,"errors":[]};assert len(manifest["protocol_fingerprint"])==64
def test_phase30_registration_is_immutable():
    manifest=build_registered_manifest(_draft(),managed_path=True);assert compare_existing_registration(manifest,manifest)==(True,"identical_registration_preserved");changed=copy.deepcopy(manifest);changed["budgets"]["max_active_edges"]+=1;assert compare_existing_registration(manifest,changed)==(False,"existing_registration_is_immutable")
@pytest.mark.parametrize(("mutation","error"),[
    (lambda d:d["arms"].pop(),"arms_do_not_match_frozen_protocol"),(lambda d:d["replicate_seeds"].pop(),"at_least_five_unique_seeds_required"),(lambda d:d["budgets"].pop("max_cache_bytes"),"missing_budgets:max_cache_bytes"),(lambda d:d["temporal_state_contract"]["fields"].remove("phase_bucket"),"temporal_state_contract_incomplete"),(lambda d:d["finite_scalar_ranges"].update({"phase":[1.0,0.0]}),"finite_scalar_ranges_invalid"),(lambda d:d["invalidation_contract"].update({"contradiction":False}),"invalidation_contract_incomplete"),(lambda d:d["execution_policy"].update({"durable_risa_mutation":True}),"execution_policy_mismatch")])
def test_phase30_protocol_drift_fails_closed(mutation,error):
    draft=_draft();mutation(draft)
    with pytest.raises(ValueError,match=error):build_registered_manifest(draft,managed_path=True)
def test_phase30_unmanaged_registration_fails_closed():
    with pytest.raises(ValueError,match="preregistration_path_not_managed"):build_registered_manifest(_draft(),managed_path=False)

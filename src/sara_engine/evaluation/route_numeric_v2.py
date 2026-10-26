"""Exact comparison for a separately registered numerical reproduction."""
from copy import deepcopy
from hashlib import sha256
import json

from sara_engine.evaluation.reactivation_codec import ReportCodec, check_tree

PROTOCOL_SHA256 = '3f818ccde2cd11d084c1d0b2904ab0eb8f384a139b93f41054db065582528c1c'
METADATA_FIELDS = ('registration_sha256', 'preserved_artifacts')


def canonical(value):
    check_tree(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def reference_frame(saved, registration_sha256, preserved_count):
    """Rebind exactly two provenance fields; never alter scientific values."""
    if type(registration_sha256) is not str or len(registration_sha256) != 64:
        raise ValueError('Expected new registration identity')
    if type(preserved_count) is not int or preserved_count < 1:
        raise ValueError('Expected positive preservation count')
    result = deepcopy(saved)
    result['registration_sha256'] = registration_sha256
    result['preserved_artifacts'] = preserved_count
    mapping = {key: {'from': saved[key], 'to': result[key]} for key in METADATA_FIELDS}
    return result, mapping


def payload(cases, summary, saved, registration_sha256, preserved_count):
    """Use newly computed cases/summary, not the historical numeric values."""
    return {'schema': saved['schema'], 'cases': cases,
            'selection_sha256': saved['selection_sha256'],
            'registration_sha256': registration_sha256, 'preserved_artifacts': preserved_count,
            **summary}


def record(value, saved, context):
    codec = ReportCodec(saved)
    normalized = codec.normalize(value)
    reference, mapping = reference_frame(saved, context['registration_sha256'], context['preserved_count'])
    return {'schema': 'sara-route-numeric-reproduction-report-v2',
            'phase': context['phase'], 'selection_sha256': PROTOCOL_SHA256,
            'registration_sha256': context['registration_sha256'],
            'execution_sha256': context['execution_sha256'], 'runtime': context['runtime'],
            'execution_kind': 'new_v2_study_execution',
            'historical_report_sha256': context['historical_report_sha256'],
            'reference_metadata_mapping': mapping, 'payload': normalized,
            'payload_canonical_sha256': sha256(codec.encode(normalized)).hexdigest(),
            'historical_values_exact_match': codec.equivalent(reference, normalized),
            'old_failed_replay_reclassified': False,
            'new_real_data_reads': 0, 'resource_measurements': 0, 'test_opened': False,
            'conditions_relaxed': False, 'promotion_authorized': False}


def validate_record(value, saved, context):
    expected = record(value['payload'], saved, context)
    if canonical(expected) != canonical(value):
        raise ValueError('New report envelope, provenance or strict types differ')
    if value['historical_values_exact_match'] is not True:
        raise ValueError('Fixed-condition historical numerical values differ')
    if value['payload']['execution_integrity_passed'] is not True:
        raise ValueError('Learning execution integrity failed')
    return value


def compare(primary, replication, saved):
    """Compare every payload field with the validated path-specific codec."""
    return ReportCodec(saved).equivalent(primary['payload'], replication['payload'])

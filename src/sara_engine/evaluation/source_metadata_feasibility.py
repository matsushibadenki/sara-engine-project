"""Conservative source-metadata capacities; no payload access or eligibility inference."""
from datetime import date
from fractions import Fraction
import json

TARGET = 70212
ORDER = ('TIHM', 'SPHERE', 'IDEAL', 'ECO')


def positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError('Positive integer required')
    return value


def calendar_envelope(start, end):
    """A source-wide calendar envelope bounds every contained identity's dates."""
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    return positive_integer(days)


def training_date_ceiling(source_date_ceiling):
    """Retain the original split: even 180 source dates give only 120 fit dates."""
    n = positive_integer(source_date_ceiling)
    return min(180, 2 * n // 3)


def regular_grid(days, cadence_seconds):
    """Exact ALL-binary regular-grid witness, never an unknown-subsequence upper bound.

    Half-open days start with a row at zero. The original greedy clock consumes
    early cutoffs even before eight history rows are available. No real source,
    event values, binary filter or learner is used by this conditional scenario.
    """
    duration = positive_integer(days) * 86400
    step = positive_integer(cadence_seconds)
    rows = (duration + step - 1) // step
    stride = (60 + step - 1) // step
    delay_rows = (300 + step - 1) // step
    first_cutoff_multiple = (7 + stride - 1) // stride
    last_cutoff_multiple = (rows - 1 - delay_rows) // stride
    arrivals = (max(0, last_cutoff_multiple - first_cutoff_multiple + 1)
                if step <= 300 and delay_rows * step <= 600 else 0)
    return {'days': days, 'cadence_seconds': step, 'raw_rows': rows,
            'qualified_arrivals_if_assumptions_hold': arrivals,
            'hypothetical_writes_per_arm': 3 * arrivals,
            'count_target_possible_in_scenario': arrivals >= TARGET,
            'within_unchanged_input_row_cap': rows <= 5000000,
            'within_unchanged_outcome_write_caps': arrivals <= 263295 and 3 * arrivals <= 789885,
            'scope': 'Conditional all-binary regular-grid witness only; not a measured source count or arbitrary-subsequence bound'}


def evidence_budget(selection, evidence):
    limits = selection['budgets']
    if [r['id'] for r in selection['candidates']] != list(ORDER) or selection['order'] != list(ORDER):
        raise ValueError('Frozen source roster or order differs')
    if len(selection['candidates']) > limits['candidate_sources']:
        raise ValueError('Candidate source budget exceeded')
    if len(evidence['search_queries']) > limits['search_queries']:
        raise ValueError('Search budget exceeded')
    pages = evidence['pages']
    if len(pages) > limits['official_metadata_pages']:
        raise ValueError('Metadata page budget exceeded')
    if len({p['id'] for p in pages}) != len(pages):
        raise ValueError('Duplicate evidence page identifier')
    for page in pages:
        if page['candidate'] not in ORDER:
            raise ValueError('Unregistered candidate evidence')
        if len(json.dumps(page, ensure_ascii=False, sort_keys=True).encode()) > limits['retained_bytes_per_page']:
            raise ValueError('Per-page retained evidence budget exceeded')
    retained = len((json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode())
    if retained > limits['retained_evidence_utf8_bytes']:
        raise ValueError('Retained evidence budget exceeded')
    for key in ('dataset_payload_bytes', 'dataset_preview_rows'):
        if evidence[key] != 0 or limits[key] != 0:
            raise ValueError('Dataset payload or preview opening prohibited')
    if evidence['sensor_message_fields_inspected'] or evidence['sealed_periods_opened']:
        raise ValueError('Sealed information opened')
    return {'search_queries': len(evidence['search_queries']), 'metadata_pages': len(pages),
            'retained_evidence_utf8_bytes': retained, 'dataset_payload_bytes': 0, 'dataset_preview_rows': 0}


def assess(source, selection):
    """Missing scoped facts stay unknown; aggregate averages cannot pass a source."""
    ceiling = source.get('source_calendar_date_ceiling')
    training = training_date_ceiling(ceiling) if ceiling is not None else None
    cap = 1440 * training if training is not None else None
    reasons = []
    if ceiling is not None and ceiling < 180:
        reasons.append('documented_duration_cannot_reach_180_recorded_dates')
    if source['native_contract'] == 'incompatible_documented_stream':
        reasons.append('documented_high_rate_stream_is_not_native_binary_events')
    elif source['native_contract'] != 'verified':
        reasons.append('native_binary_event_contract_unresolved')
    if source.get('minimum_full_bundle_bytes', 0) > selection['prospective_acquisition_ceiling_unchanged']['http_body_bytes']:
        reasons.append('documented_full_bundle_over_budget_subset_unverified')
    if source.get('verified_identity_recorded_dates') is None:
        reasons.append('per_identity_recorded_dates_unverified')
    if source.get('verified_training_upper') is None:
        reasons.append('per_identity_training_arrivals_unverified')
    elif source['verified_training_upper'] < TARGET:
        reasons.append('certified_training_upper_below_target')
    if not source.get('bounded_access_manifest_verified', False):
        reasons.append('identity_size_access_manifest_unverified')
    if cap is not None and cap < TARGET:
        reasons.append('calendar_cutoff_ceiling_below_target')
    return {'id': source['id'], 'status': ('exclude_documented_duration' if ceiling is not None and ceiling < 180
             else 'incompatible_documented_stream' if source['native_contract'] == 'incompatible_documented_stream'
             else 'unresolved_metadata'),
            'source_calendar_date_ceiling': ceiling, 'training_recorded_date_ceiling': training,
            'minute_cutoff_capacity_ceiling': cap,
            'capacity_scope': 'Conditional on published metadata envelope; not a timestamp-validated certificate',
            'ceiling_is_observed_yield': False, 'reasons': reasons,
            'dataset_acquisition_authorized': False, 'source_quality_pass': False,
            'independent_validation_passed': False, 'evidence_pages': source['evidence_pages']}


def evaluate(selection, evidence, inputs):
    if selection['fixed']['required_arrivals'] != TARGET:
        raise ValueError('Fixed target changed')
    if [s['id'] for s in inputs['sources']] != list(ORDER):
        raise ValueError('Assessment source order differs')
    work = evidence_budget(selection, evidence)
    page_ids = {p['id'] for p in evidence['pages']}
    owners = {p['id']: p['candidate'] for p in evidence['pages']}
    for source in inputs['sources']:
        if not source['evidence_pages'] or not set(source['evidence_pages']) <= page_ids:
            raise ValueError('Missing source evidence provenance')
        if any(owners[p] != source['id'] for p in source['evidence_pages']):
            raise ValueError('Evidence belongs to another source')
    return {'sources': [assess(s, selection) for s in inputs['sources']], 'work': work,
            'required_arrivals': TARGET,
            'required_per_day_at_180_train_dates': str(Fraction(TARGET, 180)),
            'minimum_valid_source_dates_for_180_fit_dates': 270,
            'regular_grid_scenarios': [regular_grid(d, c) for d in inputs['scenario_days'] for c in inputs['scenario_cadences_seconds']],
            'approved_payload_acquisitions': [], 'decision': 'no_acquisition_ready_source_within_registered_metadata_review',
            'actual_learning_writes': 0, 'casas_candidates_added': 0,
            'sensor_message_fields_inspected': False, 'independent_periods_opened': False,
            'metadata_arithmetic_is_source_or_quality_pass': False}

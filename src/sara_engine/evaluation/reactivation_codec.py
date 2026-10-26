"""Strict saved-report codec. Never imports or executes a learner."""
from copy import deepcopy
from hashlib import sha256
import json
import math

MAX_BYTES = 2097152
MAX_DEPTH = 32
MAX_NODES = 100000
ARMS = ('round_robin', 'permanent', 'periodic', 'error_triggered', 'shuffled_error')
PROTOCOL_SHA256 = 'a634fecb6c348f4ca0dfecc4fbdf54efb00d0208fbc22ea0a936a0d8f3f4e255'


def _limits(max_bytes, max_depth, max_nodes):
    for value, ceiling in ((max_bytes, MAX_BYTES), (max_depth, MAX_DEPTH), (max_nodes, MAX_NODES)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError('Invalid or enlarged codec bound')


def check_tree(value, max_bytes=MAX_BYTES, max_depth=MAX_DEPTH, max_nodes=MAX_NODES):
    """Bound JSON primitives before encoding; reject Python numeric coercions."""
    _limits(max_bytes, max_depth, max_nodes)
    nodes = string_bytes = 0

    def walk(item, depth):
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError('Codec node/depth bound exceeded')
        kind = type(item)
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError('Only string object keys are allowed')
                walk(key, depth+1)
                walk(child, depth+1)
        elif kind is list:
            for child in item:
                walk(child, depth+1)
        elif kind is str:
            string_bytes += len(item.encode('utf-8'))
            if string_bytes > max_bytes:
                raise ValueError('Codec string-byte bound exceeded')
        elif kind is int:
            if not -(2**63) <= item < 2**63:
                raise ValueError('Integer exceeds exact signed 64-bit range')
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError('Nonfinite numbers are forbidden')
        elif item is not None and kind is not bool:
            raise ValueError('Non-JSON value or numeric subclass')
    walk(value, 0)
    return nodes


def loads(raw, max_bytes=MAX_BYTES, max_depth=MAX_DEPTH, max_nodes=MAX_NODES):
    """Reject duplicate keys instead of silently dropping evidence fields."""
    _limits(max_bytes, max_depth, max_nodes)
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise ValueError('Bounded UTF-8 JSON bytes required')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid UTF-8 or excessive JSON nesting') from exc
    check_tree(value, max_bytes, max_depth, max_nodes)
    return value


def tuple_pool_path(path):
    return (len(path) == 5 and path[0] == 'cases' and type(path[1]) is int
            and path[2] == 'arms' and path[3] in ARMS and path[4] == 'initial_pool')


class ReportCodec:
    """Exact saved-layout comparison, with one path-specific tuple exception.

    The caller must bind the template to the pinned saved report. This schema
    accepts that report's exact structure, not arbitrary future report layouts.
    Equal canonical bytes mean equal supplied reports, not independent replay.
    """

    def __init__(self, saved):
        check_tree(saved)
        if type(saved) is not dict or saved.get('schema') != 'sara-route-reactivation-result-v1':
            raise ValueError('Expected the registered saved-report schema')
        self._template = deepcopy(saved)

    def normalize(self, value):
        nodes = 0

        def walk(item, template, path):
            nonlocal nodes
            nodes += 1
            if nodes > MAX_NODES or len(path) > MAX_DEPTH:
                raise ValueError('Codec layout bound exceeded')
            if type(item) is tuple and type(template) is list and tuple_pool_path(path):
                item = list(item)
            if type(item) is not type(template):
                raise ValueError('Saved-report type differs at '+str(path))
            if type(template) is dict:
                if any(type(k) is not str for k in item) or set(item) != set(template):
                    raise ValueError('Missing or extra report field')
                return {key: walk(item[key], child, path+(key,)) for key, child in template.items()}
            if type(template) is list:
                if len(item) != len(template):
                    raise ValueError('Saved-report array length differs')
                return [walk(child, reference, path+(index,))
                        for index, (child, reference) in enumerate(zip(item, template))]
            return item
        result = walk(value, self._template, ())
        check_tree(result)
        return result

    def encode(self, value):
        result = self.normalize(value)
        raw = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
        if len(raw) > MAX_BYTES:
            raise ValueError('Canonical report byte bound exceeded')
        return raw

    def equivalent(self, left, right):
        # Byte comparison keeps bool/int/float and signed zero distinct.
        return self.encode(left) == self.encode(right)


def reconstruct_pools(saved):
    """Rebuild tuple shapes from saved metadata, never from a learner replay."""
    result, paths = deepcopy(saved), []
    for index, case in enumerate(result['cases']):
        for arm, values in case['arms'].items():
            if values['initial_pool'] is not None:
                values['initial_pool'] = tuple(values['initial_pool'])
                paths.append('/cases/'+str(index)+'/arms/'+arm+'/initial_pool')
    return result, paths


def preflight(saved, failure, raw):
    """Finite, report-derived codec witnesses, not new experiment observations."""
    codec = ReportCodec(saved)
    canonical = codec.encode(saved)
    reconstructed, paths = reconstruct_pools(saved)
    if len(saved['cases']) != 20 or len(paths) != 80:
        raise ValueError('Registered saved-report roster differs')
    if failure['full_replay_verified'] is not False or failure['numeric_trace_equality'] is not None:
        raise ValueError('Original replay failure was reinterpreted')
    checks = {
        'saved_roundtrip': codec.equivalent(saved, loads(canonical)),
        'whitelisted_tuple_reconstruction': saved != reconstructed and codec.equivalent(saved, reconstructed),
        'object_key_order': codec.equivalent(saved, dict(reversed(list(saved.items())))),
    }

    def reject(name, action):
        try:
            action()
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False

    def brier(value):
        result = deepcopy(saved)
        result['cases'][0]['arms']['error_triggered']['tail_brier'] = value
        return result

    old = saved['cases'][0]['arms']['error_triggered']['tail_brier']
    checks['float_one_ulp_change'] = not codec.equivalent(saved, brier(math.nextafter(old, math.inf)))
    checks['signed_zero_distinction'] = not codec.equivalent(brier(0.0), brier(-0.0))
    variant = deepcopy(saved)
    variant['component_passed'] = int(variant['component_passed'])
    reject('bool_to_int', lambda: codec.encode(variant))
    variant = deepcopy(saved)
    variant['cases'][0]['seed'] = float(variant['cases'][0]['seed'])
    reject('int_to_float', lambda: codec.encode(variant))
    variant = deepcopy(saved)
    variant['cases'][0]['arms']['permanent']['initial_pool'].reverse()
    checks['array_order'] = not codec.equivalent(saved, variant)
    variant = deepcopy(saved)
    del variant['preserved_artifacts']
    reject('missing_field', lambda: codec.encode(variant))
    variant = deepcopy(saved)
    variant['extra'] = False
    reject('extra_field', lambda: codec.encode(variant))
    variant = deepcopy(saved)
    variant['cases'] = tuple(variant['cases'])
    reject('tuple_elsewhere', lambda: codec.encode(variant))
    reject('nan', lambda: codec.encode(brier(float('nan'))))
    reject('infinity', lambda: codec.encode(brier(float('inf'))))
    duplicate = b'{"schema":"sara-route-reactivation-result-v1",'+raw.lstrip()[1:]
    reject('duplicate_json_keys', lambda: loads(duplicate))
    variant = deepcopy(saved)
    variant[1] = 0
    reject('non_string_keys', lambda: codec.encode(variant))
    variant = deepcopy(saved)
    variant['cases'][0]['seed'] = 2**63
    reject('oversized_integer', lambda: codec.encode(variant))
    reject('input_byte_limit', lambda: loads(raw, max_bytes=16))
    reject('node_limit', lambda: loads(raw, max_nodes=10))
    reject('depth_limit', lambda: loads(raw, max_depth=2))
    return {'schema': 'sara-reactivation-codec-result-v1', 'checks': checks,
            'codec_preflight_passed': all(checks.values()),
            'canonical_bytes': len(canonical), 'canonical_sha256': sha256(canonical).hexdigest(),
            'reconstructed_tuple_paths': paths,
            'reconstruction_provenance': 'Saved initial_pool lists only; not a runtime/replay report',
            'full_replay_verified': False, 'numeric_trace_equality': None,
            'original_component_passed': saved['component_passed'],
            'source_report_sha256': sha256(raw).hexdigest(),
            'learner_executions': 0, 'new_predictions_scored': 0, 'new_measurements': 0,
            'new_acquisitions': 0, 'test_opened': False, 'conditions_relaxed': False,
            'study_restart_authorized': False, 'promotion_authorized': False}

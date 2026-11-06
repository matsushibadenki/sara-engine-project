"""Read-only consistency guard for selected roadmap gates, never a run authorizer."""
from hashlib import sha256
import re

STATUS = {'Done':'🟢', 'Next':'🟠', 'Later':'🔴', 'Pending':'⭕️'}


def check(manifest, evidence, roadmaps):
    """Return bounded documentation errors without importing models or changing files."""
    errors = []
    if manifest.get('schema') != 'sara-roadmap-gates-v1' or manifest.get('runtime_authorized') is not False:
        return ['Invalid manifest or attempted runtime authorization']
    entries = manifest['entries']
    ids = [entry['id'] for entry in entries]
    if len(set(ids)) != len(ids) or len(ids) != 4:
        return ['Selected gate roster must contain four unique entries']
    if set(roadmaps) != set(manifest['roadmaps']):
        return ['Roadmap roster differs']
    if set(evidence) != {entry['evidence'] for entry in entries}:
        return ['Evidence roster differs']
    if sum(map(len, evidence.values())) + sum(map(len, roadmaps.values())) > manifest['input_byte_limit']:
        return ['Documentation input budget exceeded']
    for entry in entries:
        if sha256(evidence[entry['evidence']]).hexdigest() != entry['evidence_sha256']:
            errors.append('Evidence changed; review gate before updating pin: ' + entry['id'])
        if entry['status'] not in STATUS:
            errors.append('Unknown status: ' + entry['id'])
            continue
        marker = '<!-- gate:' + entry['id'] + ' -->'
        for name, raw in roadmaps.items():
            text = raw.decode('utf-8')
            rows = [line for line in text.splitlines() if marker in line]
            expected = '- ' + STATUS[entry['status']] + ' [' + entry['status'] + '] '
            if text.count(marker) != 1 or len(rows) != 1 or not rows[0].startswith(expected):
                errors.append('Missing, duplicate or inconsistent gate status: ' + name + ':' + entry['id'])
    for name, raw in roadmaps.items():
        text = raw.decode('utf-8')
        for line in text.splitlines():
            if re.search(r'\[Next\]', line) and any(phrase in line for phrase in manifest['forbidden_next_phrases']):
                errors.append('Consumed work is still Next: ' + name)
    return errors

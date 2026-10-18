"""Finite-roster timestamp screening with shared, fail-closed work budgets."""
from collections import Counter
from datetime import date
import re
import struct

from sara_engine.evaluation.casas_180_source import timestamp
from sara_engine.evaluation.casas_timestamp_bound import BoundLimits, bound_lines


class CapacityError(ValueError):
    """A preregistered capacity was exhausted before the next operation."""


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.used = Counter()

    def charge(self, key, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('Nonnegative integer charge required')
        if self.used[key] + amount > self.limits[key]:
            raise CapacityError('Aggregate capacity: ' + key)
        self.used[key] += amount


def validate_selection(selection, directory):
    """Reject roster substitution, duplication, reordering and impossible reserves."""
    roster = selection['roster']
    if [r['identity'] for r in roster] != ['HH106', 'HH107', 'HH108']:
        raise ValueError('Frozen finite roster or order differs')
    limits = selection['budgets']
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError('Positive integer capacities required')
    if len(roster) != limits['households']:
        raise ValueError('Roster size differs')
    for r in roster:
        expected = next(x for x in directory['members'] if x['name'] == r['name'])
        if {k: r[k] for k in expected} != expected or r['method'] != 8:
            raise ValueError('Pinned directory entry differs')
        if r['size'] > limits['expanded_bytes_per_pass_per_household']:
            raise CapacityError('Member expanded size exceeds capacity')
        if r['compressed_size'] + 1024 > limits['network_bytes_per_household']:
            raise CapacityError('Member range exceeds capacity')
    if sum(r['compressed_size'] + 1024 for r in roster) > limits['network_bytes']:
        raise CapacityError('Roster acquisition exceeds capacity')
    if 2 * sum(r['size'] for r in roster) > limits['expanded_bytes_all_passes']:
        raise CapacityError('Two-pass roster expansion exceeds capacity')


def validate_range(headers, start, end, total, length):
    """Inspect only response range metadata; never expose private header contents."""
    matches = re.findall(rb'(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$', headers)
    if not matches or tuple(map(int, matches[-1])) != (start, end, total):
        raise ValueError('HTTP range differs')
    if length != end - start + 1:
        raise ValueError('HTTP range length differs')


def member_range(header, entry):
    if len(header) < 30:
        raise ValueError('Short ZIP local header')
    signature, version, flags, method, time, day, crc, compressed, size, names, extra = struct.unpack('<4s5H3I2H', header[:30])
    stop = 30 + names + extra
    if signature != b'PK\x03\x04' or flags & 1 or method != 8 or stop > len(header):
        raise ValueError('Unsupported ZIP local header')
    if header[30:30 + names].decode('utf-8') != entry['name']:
        raise ValueError('ZIP identity differs')
    if not flags & 8 and (crc, compressed, size) != (entry['crc32'], entry['compressed_size'], entry['size']):
        raise ValueError('ZIP local and central metadata differ')
    start = entry['offset'] + stop
    return start, start + entry['compressed_size'] - 1


def date_time_prefix(line):
    """Delimiter choice depends only on the ISO-date boundary, never on payload."""
    comma = len(line) > 10 and line[10] == ','
    prefix = line.split(',', 1) if comma else line.split(None, 1)
    if len(prefix) != 2 or len(prefix[0]) != 10:
        raise ValueError('Missing canonical date/time prefix')
    day = date.fromisoformat(prefix[0]).isoformat()
    if day != prefix[0]:
        raise ValueError('Noncanonical date')
    parts = prefix[1].split(',', 1) if comma else prefix[1].split(None, 1)
    if not parts or not parts[0]:
        raise ValueError('Missing time prefix')
    at = timestamp(day, parts[0])
    if at.tzinfo is not None:
        raise ValueError('Naive source timestamp required')
    return day, at


def timestamp_metadata(lines, budget, per_household):
    """Preserve original date splits and regression exclusions without event parsing."""
    counts = Counter()
    regressions = set()
    previous = None
    rows = 0
    for line in lines:
        if rows >= per_household:
            raise CapacityError('Household metadata timestamp capacity')
        budget.charge('metadata_timestamps_total', 1)
        day, at = date_time_prefix(line)
        if previous is not None and at < previous:
            regressions.update((previous.date().isoformat(), day))
        previous = at
        counts[day] += 1
        rows += 1
    valid = sorted(set(counts) - regressions)
    if len(valid) < 180:
        raise ValueError('Source lacks 180 recorded chronological dates')
    training = min(180, 2 * len(valid) // 3)
    development = min(60, len(valid) // 6)
    return {'rows': rows, 'first': min(counts), 'last': max(counts),
            'recorded_days': len(counts), 'valid_recorded_days': len(valid),
            'clock_regression_days': sorted(regressions),
            'recorded_date_counts': dict(sorted(counts.items())),
            'train_start': valid[0], 'development_start': valid[training],
            'test_start': valid[training + development],
            'train_recorded_days': training, 'development_recorded_days': development,
            'sealed_recorded_days': len(valid) - training - development,
            'sensor_message_fields_inspected': False,
            'scope': 'All-source timestamp census for original splits and clock exclusions only'}


def screen(lines, metadata, budget, limits):
    remaining = limits['training_timestamps_total'] - budget.used['training_timestamps_total']
    if remaining < 1:
        raise CapacityError('Aggregate training timestamp capacity')
    bound_limits = BoundLimits(min(remaining, limits['training_timestamps_per_household']), limits['pending_buckets'])
    def metered():
        charged = 0
        for line in lines:
            # Meter timestamp attempts before handing the prefix to the parser.
            day = line[:10]
            if (metadata['train_start'] <= day < metadata['development_start']
                    and day not in metadata['clock_regression_days']
                    and charged < bound_limits.opened_timestamps):
                budget.charge('training_timestamps_total', 1)
                charged += 1
            yield line
    return bound_lines(metered(), metadata, bound_limits)

"""Bounded range-member ingestion and timestamp-only qualification."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import zlib

CHUNK = 65536
MAX_LINE = 1048576


def file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def timestamp(date, time):
    """Normalize variable fractional precision for Python 3.10."""
    whole, dot, fraction = time.partition('.')
    normalized = whole + ('.' + fraction.ljust(6, '0')[:6] if dot else '')
    return datetime.fromisoformat(date + 'T' + normalized)


def member_lines(path, receipt, audit=None):
    """Stream raw DEFLATE, validating exact size, CRC and SHA at EOF."""
    if Path(path).stat().st_size != receipt['compressed_size']:
        raise ValueError('Compressed range length differs')
    if file_sha256(path) != receipt['compressed_sha256']:
        raise ValueError('Compressed source checksum differs')
    decoder = zlib.decompressobj(-15)
    digest = sha256()
    crc = 0
    size = 0
    buffer = b''
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            while block:
                output = decoder.decompress(block, MAX_LINE)
                block = decoder.unconsumed_tail
                size += len(output)
                if size > receipt['size']:
                    raise ValueError('Expanded source exceeds pinned size')
                digest.update(output)
                crc = zlib.crc32(output, crc)
                buffer += output
                parts = buffer.split(b'\n')
                buffer = parts.pop()
                if len(buffer) > MAX_LINE:
                    raise ValueError('Source line exceeds bound')
                for line in parts:
                    if len(line) > MAX_LINE:
                        raise ValueError('Source line exceeds bound')
                    yield line.rstrip(b'\r').decode('utf-8')
    if not decoder.eof or decoder.unused_data or size != receipt['size'] or crc != receipt['crc32']:
        raise ValueError('Truncated, trailing or corrupted source member')
    actual = digest.hexdigest()
    if receipt.get('uncompressed_sha256') and actual != receipt['uncompressed_sha256']:
        raise ValueError('Expanded source checksum differs')
    if buffer:
        yield buffer.rstrip(b'\r').decode('utf-8')
    if audit is not None:
        audit.update({'uncompressed_sha256':actual, 'size':size, 'crc32':crc})


def metadata(lines):
    """Read only dates, times and column counts, never sensor/message cells."""
    counts, shapes = Counter(), Counter()
    regressions = set()
    previous = None
    rows = 0
    for line in lines:
        prefix = line.split(',', 2) if ',' in line else line.split(None, 2)
        if len(prefix) < 2:
            raise ValueError('Missing timestamp metadata')
        t = timestamp(prefix[0], prefix[1])
        day = t.date().isoformat()
        if previous is not None and t < previous:
            regressions.update((previous.date().isoformat(), day))
        previous = t
        counts[day] += 1
        shapes[line.count(',') + 1 if ',' in line else len(line.split())] += 1
        rows += 1
    valid = sorted(set(counts) - regressions)
    if len(valid) < 180:
        raise ValueError('Source lacks 180 recorded chronological dates')
    train_days = min(180, 2*len(valid)//3)
    development_days = min(60, len(valid)//6)
    return {'rows':rows, 'first':min(counts), 'last':max(counts),
            'recorded_days':len(counts), 'valid_recorded_days':len(valid),
            'clock_regression_days':sorted(regressions), 'column_counts':dict(shapes),
            'recorded_date_counts':dict(sorted(counts.items())),
            'train_start':valid[0], 'development_start':valid[train_days],
            'test_start':valid[train_days + development_days],
            'train_recorded_days':train_days, 'development_recorded_days':development_days,
            'sealed_recorded_days':len(valid)-train_days-development_days,
            'selection':'HH103 is smallest unconsumed HH identity; numeric input/outcome cells unread'}

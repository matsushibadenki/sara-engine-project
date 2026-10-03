"""Parse electricity support rows only after a frozen year access check."""
from datetime import datetime
from decimal import Decimal, InvalidOperation


NUMERIC_ACCESS_YEARS = frozenset((2011, 2012, 2013))


def parse_load_token(token: str) -> Decimal | None:
    """Return an exact nonnegative finite load, or an invalid observation."""
    try:
        value = Decimal(token.strip().strip('"').replace(',', '.'))
    except InvalidOperation:
        return None
    return value if value.is_finite() and value >= 0 else None


def parse_support_row(line: str, *, client_count: int = 370):
    """Leave sealed and out-of-scope payloads opaque, even when malformed."""
    prefix, separator, payload = line.rstrip('\r\n').partition(';')
    if not separator:
        raise ValueError('Missing timestamp separator')
    timestamp = datetime.strptime(prefix.strip('"'), '%Y-%m-%d %H:%M:%S')
    # Access check precedes splitting or numeric conversion of load tokens.
    if timestamp.year not in NUMERIC_ACCESS_YEARS:
        return timestamp, None
    tokens = payload.split(';')
    if len(tokens) != client_count:
        raise ValueError('Unexpected client count')
    return timestamp, tuple(parse_load_token(token) for token in tokens)

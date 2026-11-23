from datetime import datetime, timedelta
from email.message import Message
from io import BytesIO

from scripts.eval.asos_ksea_format_audit import run
from sara_engine.evaluation.asos_ksea_delayed_contract import DEVELOPMENT_ACQUISITION_PLAN
from sara_engine.evaluation.asos_ksea_format_audit import summarize


def test_oversize_header_stops_without_reading_or_requesting_next_file():
    calls = []

    class Response(BytesIO):
        status = 200
        headers = Message()

        def read(self, size):
            raise AssertionError("Over-budget body must remain unopened")

    response = Response()
    first = DEVELOPMENT_ACQUISITION_PLAN[0]
    response.headers["Content-Length"] = str(first.max_body_bytes + 1)

    def open_url(request, timeout):
        calls.append((request.full_url, timeout))
        return response

    result = run(open_url)
    assert calls == [(first.url, 60)]
    assert result["status"] == "inconclusive"
    assert result["stage"] == "acquisition"
    assert result["acquired"] == []
    assert "counts" not in result


def test_distinguishes_absent_row_unrecognized_visibility_and_opposing_histories():
    jan = BytesIO()
    start = datetime(2024, 1, 1)
    for week, first, future in [(0, b"0.080", b"0.100"), (1, b"0.110", b"0.099")]:
        for i in range(14):
            at = start + timedelta(weeks=week, minutes=i)
            utc = at + timedelta(hours=8)
            header = f"24233KSEA SEA{at:%Y%m%d%H%M}{utc:%H%M}".encode()
            coefficient = {0: first, 12: future}.get(i, b"0.090")
            payload = b" " + coefficient + b" D 0.500 N" if i < 13 else b" [M] M"
            jan.write(header + payload.ljust(70) + b"\n")
    jan.seek(0)

    counts = summarize((jan, *(BytesIO() for _ in range(5))))
    assert counts["indexed_minutes"] == 28
    assert counts["missing_rows"] == 182 * 1440 - 28
    assert counts["unrecognized_visibility"] == 2
    assert counts["valid_minutes"] == 26
    assert counts["valid_dates"] == counts["eligible_prediction_dates"] == 2
    assert counts["coverage_180"] is False
    assert counts["negative"] == counts["positive"] == 1
    assert counts["repeated_keys"] == counts["keys_with_both_classes"] == 1
    assert counts["keys_with_opposite_outcomes_across_directions"] == 1
    assert counts["opposing_keys_with_80_per_direction"] == 0

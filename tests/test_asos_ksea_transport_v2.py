from email.message import Message
from io import BytesIO

import pytest

from scripts.eval import asos_ksea_transport_v2 as transport


class Response(BytesIO):
    status = 200

    def __init__(self, body, size, etag):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Length"] = str(size)
        self.headers["ETag"] = etag
        self.reads = []

    def read(self, size):
        self.reads.append(size)
        return super().read(size)


def test_manifest_is_fixed():
    assert len(transport._MANIFEST) == 6
    assert sum(size for size, _ in transport._MANIFEST) == 27_589_402
    assert transport._MANIFEST[0] == (4_668_708, '"473d24-6106bbb1c40c0"')


@pytest.mark.parametrize("header,value", [
    ("ETag", '"wrong"'),
    ("Content-Length", "0"),
    ("Content-Encoding", "gzip"),
    ("Transfer-Encoding", "chunked"),
])
def test_header_rejection_before_read(header, value):
    size, etag = transport._MANIFEST[0]
    response = Response(b"", size, etag)
    del response.headers[header]
    response.headers[header] = value
    result = transport.run(lambda request, timeout: response)
    assert result["stage"] == "acquisition"
    assert result["reserved_bytes"] == 0
    assert response.reads == []


def test_structure_stop_reports_lengths_before_next_month(monkeypatch):
    body = b"X" * 104 + b"\n" + b"X" * 105
    etag = transport._MANIFEST[0][1]
    monkeypatch.setattr(transport, "_MANIFEST", ((len(body), etag), *transport._MANIFEST[1:]))
    calls = []

    def open_url(request, timeout):
        calls.append((request.full_url, request.get_header("If-match"), timeout))
        return Response(body, len(body), etag)

    result = transport.run(open_url)
    assert calls == [(transport.DEVELOPMENT_ACQUISITION_PLAN[0].url, etag, 60)]
    assert result["stage"] == "structure"
    assert result["reserved_bytes"] == len(body)
    assert result["physical_lengths_excluding_lf"] == {104: 1, 105: 1}
    assert result["final_lf"] is False
    assert "counts" not in result


def test_all_six_valid_months_allow_summary(monkeypatch):
    bodies = [
        f"24233KSEA SEA2024{month:02d}0100000800".encode() + b" " * 70 + b"\n"
        for month in range(1, 7)
    ]
    manifest = tuple((len(body), f'"{i}"') for i, body in enumerate(bodies))
    monkeypatch.setattr(transport, "_MANIFEST", manifest)
    responses = iter(Response(body, *entry) for body, entry in zip(bodies, manifest))

    result = transport.run(lambda request, timeout: next(responses))
    assert result["status"] == "complete"
    assert result["reserved_bytes"] == 600
    assert result["counts"]["indexed_minutes"] == 6


def test_truncation_keeps_reservation_and_stops():
    size, etag = transport._MANIFEST[0]
    response = Response(b"x", size, etag)
    result = transport.run(lambda request, timeout: response)
    assert result["stage"] == "acquisition"
    assert result["reserved_bytes"] == size
    assert result["acquired"] == []
    assert response.reads == [size]

import io

import pytest

from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.memory.topic_evidence_pages import refresh_from_pages
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore


def transport(monkeypatch, body=b'{"page":0}', *, status=200, headers=None):
    state = {}
    class Response(io.BytesIO):
        def getheader(self, name, default=None):
            return (headers if headers is not None else {"Content-Type": "application/json"}).get(name, default)
    response = Response(body)
    response.status = status
    class Connection:
        sock = None
        def __init__(self, host, port, timeout):
            state["timeout"] = timeout
        def request(self, method, target, headers):
            state["target"] = target
        def getresponse(self):
            return response
        def close(self):
            state["closed"] = True
    monkeypatch.setattr("sara_engine.memory.evidence_http.http.client.HTTPSConnection", Connection)
    return state


def test_bounded_json_transport(monkeypatch):
    state = transport(monkeypatch)
    client = EvidenceHTTPClient("https://example.test/evidence?scope=sensor&page=9", lambda payload: payload)
    assert client(2) == {"page": 0}
    assert state == {"timeout": 5.0, "target": "/evidence?scope=sensor&page=2", "closed": True}


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'not json', b'\xff'])
def test_invalid_json_never_reaches_decoder(monkeypatch, body):
    state = transport(monkeypatch, body)
    client = EvidenceHTTPClient("https://example.test/", lambda _: pytest.fail("Invalid payload decoded"))
    with pytest.raises(ValueError):
        client(0)
    assert state["closed"]


def test_stream_size_limit_without_content_length(monkeypatch):
    state = transport(monkeypatch, b"x" * 33)
    with pytest.raises(ValueError, match="size limit"):
        EvidenceHTTPClient("https://example.test/", lambda x: x, max_bytes=32)(0)
    assert state["closed"]


@pytest.mark.parametrize("headers", [
    {"Content-Type": "text/html"},
    {"Content-Type": "application/json", "Content-Encoding": "gzip"},
    {"Content-Type": "application/json", "Content-Length": "10000000"},
    {"Content-Type": "application/json", "Content-Length": "15"},
])
def test_bad_response_contract(monkeypatch, headers):
    transport(monkeypatch, headers=headers)
    with pytest.raises(ValueError):
        EvidenceHTTPClient("https://example.test/", lambda x: x)(0)


def test_http_failure_propagates_to_refresh_revocation(monkeypatch):
    state = transport(monkeypatch, status=302)
    store = TopicEvidenceStore()
    store.publish((), expected_generation=0, now_segment=1)
    client = EvidenceHTTPClient("https://example.test/", lambda _: pytest.fail("Redirect decoded"))
    result = refresh_from_pages(store, client, scope="test", expected_generation=1, now_segment=2)
    assert result.decision == "page_unavailable" and result.generation == 2
    assert state["closed"]


def test_late_response_is_rejected(monkeypatch):
    state = transport(monkeypatch)
    ticks = iter((0.0, 6.0))
    monkeypatch.setattr("sara_engine.memory.evidence_http.time.monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        EvidenceHTTPClient("https://example.test/", lambda x: x)(0)
    assert state["closed"]


@pytest.mark.parametrize("endpoint", ["http://example.test/", "https://user:pass@example.test/", "ftp://example.test/", "https://example.test/#x"])
def test_endpoint_policy(endpoint):
    with pytest.raises(ValueError):
        EvidenceHTTPClient(endpoint, lambda x: x)

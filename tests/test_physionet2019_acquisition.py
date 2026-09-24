"""PhysioNet acquisition verifies bytes before publishing raw files."""
from hashlib import md5, sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/eval"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "physionet2019_acquire", SCRIPT_DIR / "physionet2019_acquire.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Body:
    def __init__(self, value):
        self.value = value
        self.closed = False

    def iter_chunks(self, chunk_size):
        yield self.value[:3]
        yield self.value[3:]

    def close(self):
        self.closed = True


class Client:
    def __init__(self, value, etag=None):
        self.value = value
        self.etag = etag or md5(value).hexdigest()
        self.calls = 0
        self.body = None

    def get_object(self, **kwargs):
        self.calls += 1
        self.body = Body(self.value)
        return {"Body": self.body, "ContentLength": len(self.value),
                "ETag": f'"{self.etag}"'}


def configure_paths(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    monkeypatch.setattr(MODULE, "raw_data_path", lambda *parts: str(raw.joinpath(*parts)))
    monkeypatch.setattr(MODULE, "ensure_parent_directory", lambda path: (
        Path(path).parent.mkdir(parents=True, exist_ok=True) or str(path)))
    return raw


def test_verified_body_is_atomically_published_and_resume_avoids_network(monkeypatch, tmp_path):
    raw = configure_paths(monkeypatch, tmp_path)
    value = b"patient-body"
    etag = md5(value).hexdigest()
    client = Client(value)
    kwargs = {"bucket": "source", "key": "prefix/set/p000001.psv",
              "size": len(value), "etag": etag, "prefix": "prefix/", "scratch": tmp_path}
    first = MODULE.download_one(client, **kwargs)
    assert first == ("prefix/set/p000001.psv", len(value), sha256(value).hexdigest(), False)
    assert (raw / "r2_physionet2019/set/p000001.psv").read_bytes() == value
    assert client.body.closed is True
    second = MODULE.download_one(client, **kwargs)
    assert second[-1] is True
    assert client.calls == 1


def test_changed_or_multipart_source_fails_closed(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    value = b"patient-body"
    with pytest.raises(ValueError, match="single-part"):
        MODULE.download_one(Client(value), bucket="source", key="prefix/set/p.psv",
                            size=len(value), etag=f"{md5(value).hexdigest()}-2",
                            prefix="prefix/", scratch=tmp_path)
    with pytest.raises(ValueError, match="changed"):
        MODULE.download_one(Client(value, "0" * 32), bucket="source",
                            key="prefix/set/p000002.psv", size=len(value), etag="1" * 32,
                            prefix="prefix/", scratch=tmp_path)


def test_existing_mismatched_raw_file_is_not_overwritten(monkeypatch, tmp_path):
    raw = configure_paths(monkeypatch, tmp_path)
    target = raw / "r2_physionet2019/set/p000003.psv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")
    value = b"patient-body"
    client = Client(value)
    with pytest.raises(ValueError, match="Existing raw file"):
        MODULE.download_one(client, bucket="source", key="prefix/set/p000003.psv",
                            size=len(value), etag=md5(value).hexdigest(),
                            prefix="prefix/", scratch=tmp_path)
    assert target.read_bytes() == b"wrong"
    assert client.calls == 0


def test_complete_acquisition_result_is_pinned_and_parse_free():
    protocol = ROOT / "data/processed/benchmark_fixtures/physionet2019_acquisition_protocol_v1.json"
    result_path = ROOT / "workspace/evaluation/physionet2019_acquisition_v1.json"
    assert sha256(protocol.read_bytes()).hexdigest() == MODULE.PROTOCOL_SHA256
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "3427bd697c98232bef86e70601bc6a27d53af0cab760703d8a943573446788db")
    result = json.loads(result_path.read_text())
    assert result["complete"] is True
    assert result["verified_files"] == 40336
    assert result["verified_bytes"] == 255057484
    assert result["clinical_columns_parsed"] is False
    assert result["patient_labels_interpreted"] is False
    assert result["candidate_scored"] is False

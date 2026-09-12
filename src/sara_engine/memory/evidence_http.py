"""Small bounded HTTP JSON transport for a caller-configured evidence source."""

import http.client
import json
import math
import ssl
import time
from urllib.parse import parse_qsl, urlencode, urlsplit


class EvidenceHTTPClient:
    """Fetch JSON pages; a trusted decoder supplies the EvidencePage contract.

    No redirects, credentials, decompression or implicit retries. Socket timeout
    limits inactivity; elapsed-budget checks reject late results but cannot
    interrupt every blocking resolver/header operation at an exact deadline.
    """

    def __init__(self, endpoint, decode_page, *, timeout_seconds=5.0, max_bytes=262144, ca_file=None):
        if not isinstance(endpoint, str) or len(endpoint) > 2048:
            raise ValueError("Invalid endpoint")
        url = urlsplit(endpoint)
        if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("Invalid endpoint")
        if url.scheme == "http" and url.hostname not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("HTTP is permitted only for local testing")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("Invalid timeout")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 1048576:
            raise ValueError("Invalid response limit")
        if not callable(decode_page):
            raise ValueError("A page decoder is required")
        if ca_file is not None and url.scheme != "https":
            raise ValueError("Custom trust roots require HTTPS")
        # Private publishers may use an explicit CA bundle. Verification and
        # hostname checks remain enabled; no insecure-context option is exposed.
        self._tls_context = ssl.create_default_context(cafile=ca_file) if ca_file is not None else None
        self._url = url
        self._decode = decode_page
        self._timeout = float(timeout_seconds)
        self._max_bytes = max_bytes

    def __call__(self, index):
        if type(index) is not int or not 0 <= index < 12:
            raise ValueError("Invalid page index")
        params = [(key, value) for key, value in parse_qsl(self._url.query, keep_blank_values=True) if key != "page"]
        params.append(("page", str(index)))
        target = (self._url.path or "/") + "?" + urlencode(params)
        connection_type = http.client.HTTPSConnection if self._url.scheme == "https" else http.client.HTTPConnection
        options = {"timeout": self._timeout}
        if self._tls_context is not None:
            options["context"] = self._tls_context
        connection = connection_type(self._url.hostname, self._url.port, **options)
        started = time.monotonic()
        try:
            connection.request("GET", target, headers={"Accept": "application/json", "Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("Evidence HTTP status is not 200")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError("Evidence response is not JSON")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Encoded evidence responses are unsupported")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > self._max_bytes):
                raise ValueError("Evidence response exceeds size limit")
            body = bytearray()
            while True:
                remaining = self._timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Evidence fetch exceeded elapsed budget")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(8192, self._max_bytes + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > self._max_bytes:
                    raise ValueError("Evidence response exceeds size limit")
            if time.monotonic() - started >= self._timeout:
                raise TimeoutError("Evidence fetch exceeded elapsed budget")
            if length is not None and len(body) != int(length):
                raise ValueError("Incomplete evidence response")
            def reject_constant(value):
                raise ValueError("Non-finite JSON value")
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result
            payload = json.loads(body.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
            if not isinstance(payload, dict):
                raise ValueError("Evidence response must be an object")
            return self._decode(payload)
        finally:
            connection.close()

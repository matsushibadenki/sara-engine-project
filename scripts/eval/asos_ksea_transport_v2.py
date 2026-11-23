"""Run the registered KSEA transport and per-month framing audit."""

import hashlib
import json
import signal
from http.client import HTTPException
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from sara_engine.evaluation.asos_ksea_delayed_contract import DEVELOPMENT_ACQUISITION_PLAN
from sara_engine.evaluation.asos_ksea_format_audit import summarize
from sara_engine.evaluation.asos_ksea_timestamp_preflight import iter_development_records

from scripts.eval.asos_ksea_format_audit import RejectRedirect, _deadline, _physical_lengths

_MANIFEST = (
    (4_668_708, '"473d24-6106bbb1c40c0"'),
    (4_389_485, '"42fa6d-61276d2d6d740"'),
    (4_689_387, '"478deb-61522065b9300"'),
    (4_809_958, '"4964e6-6180a66574980"'),
    (4_194_108, '"3fff3c-619ed408a1a40"'),
    (4_837_756, '"49d17c-621f14e414800"'),
)
_BODY_CEILING = 27_589_402


def run(open_url):
    acquired = []
    bodies = []
    reserved = 0
    stage = "acquisition"
    month = None
    try:
        for month, (item, (size, etag)) in enumerate(
            zip(DEVELOPMENT_ACQUISITION_PLAN, _MANIFEST, strict=True), 1
        ):
            stage = "acquisition"
            request = Request(item.url, headers={"Accept-Encoding": "identity", "If-Match": etag})
            with open_url(request, timeout=60) as response:
                headers = response.headers
                if (response.status != 200
                        or headers.get_all("ETag", []) != [etag]
                        or headers.get_all("Content-Length", []) != [str(size)]
                        or headers.get_all("Content-Encoding", []) not in ([], ["identity"])
                        or "Transfer-Encoding" in headers):
                    raise ValueError("Response does not match registered transport headers")
                if reserved + size > _BODY_CEILING:
                    raise ValueError("Aggregate body budget exceeded")
                reserved += size
                body = response.read(size)
                if len(body) != size:
                    raise ValueError("Response body length does not match registration")

            bodies.append(body)
            acquired.append({"month": month, "bytes": size, "sha256": hashlib.sha256(body).hexdigest()})
            stage = "structure"
            streams = tuple(BytesIO(body if slot == month else b"") for slot in range(1, 7))
            for _ in iter_development_records(streams):
                pass

        stage = "summary"
        return {"status": "complete", "reserved_bytes": reserved, "acquired": acquired,
                "counts": summarize(tuple(BytesIO(body) for body in bodies))}
    except (OSError, ValueError, HTTPException) as exc:
        if isinstance(exc, HTTPError):
            exc.close()
        result = {"status": "inconclusive", "stage": stage, "month": month,
                  "error": str(exc), "reserved_bytes": reserved, "acquired": acquired}
        if stage == "structure":
            result["physical_lengths_excluding_lf"] = _physical_lengths(bodies[-1])
            result["final_lf"] = bodies[-1].endswith(b"\n")
        return result


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(300)
    print(json.dumps(run(build_opener(ProxyHandler({}), RejectRedirect()).open), indent=2))

"""Run the one-shot registered development-only KSEA acquisition."""

import hashlib
import json
import signal
from collections import Counter
from http.client import HTTPException
from io import BytesIO
from urllib.request import HTTPRedirectHandler, Request, build_opener

from sara_engine.evaluation.asos_ksea_delayed_contract import DEVELOPMENT_ACQUISITION_PLAN
from sara_engine.evaluation.asos_ksea_format_audit import summarize


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _deadline(_signum, _frame):
    raise TimeoutError("Total audit deadline exceeded")


def _physical_lengths(body):
    return dict(sorted(Counter(map(len, body.removesuffix(b"\n").split(b"\n"))).items()))


def run(open_url):
    acquired = []
    bodies = []
    stage = "acquisition"
    try:
        for item in DEVELOPMENT_ACQUISITION_PLAN:
            request = Request(item.url, headers={"Accept-Encoding": "identity"})
            with open_url(request, timeout=60) as response:
                lengths = response.headers.get_all("Content-Length", [])
                encoding = response.headers.get("Content-Encoding", "identity")
                if (response.status != 200 or encoding != "identity"
                        or len(lengths) != 1 or not lengths[0].isascii()
                        or not lengths[0].isdigit()):
                    raise ValueError("Unexpected response headers")
                size = int(lengths[0])
                if not 0 < size <= item.max_body_bytes:
                    raise ValueError("Body length outside registered budget")
                body = response.read(size)
                if len(body) != size:
                    raise ValueError("Truncated response")
            bodies.append(body)
            acquired.append({"month": len(bodies), "bytes": size, "sha256": hashlib.sha256(body).hexdigest()})
            lengths = _physical_lengths(body)
            if any(length not in (99, 100) for length in lengths):
                stage = "framing"
                raise ValueError("Unexpected physical record lengths")

        stage = "audit"
        return {"status": "complete", "acquired": acquired,
                "counts": summarize(tuple(BytesIO(body) for body in bodies))}
    except (OSError, ValueError, HTTPException) as exc:
        result = {"status": "inconclusive", "stage": stage, "error": str(exc), "acquired": acquired}
        if stage != "acquisition":
            result["physical_lengths_excluding_lf"] = [
                _physical_lengths(body) for body in bodies
            ]
        return result


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(300)
    print(json.dumps(run(build_opener(RejectRedirect()).open), indent=2))

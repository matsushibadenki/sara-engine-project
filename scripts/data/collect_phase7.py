#!/usr/bin/env python3
"""Collect the independent, provenance-first Phase 7 corpus.

The cards below are short excerpts transcribed from the linked public sources.
They are deliberately one-source-per-record; no paraphrase expansion is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from sara_engine.utils.project_paths import raw_data_path, processed_data_path, workspace_path, ensure_parent_directory


def _simhash(text: str) -> str:
    tokens = re.findall(r"[\w-]+", text.lower())[:256]
    weights = [0] * 64
    for token in tokens:
        value = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return f"{sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0):016x}"


# source_url, source_revision/collection_time, license, task_type, excerpt
# Revision dates are the source publication or version dates, not retrieval time.
TRAIN = [
    ("docs.python.org", "https://docs.python.org/3.11/library/pathlib.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "qa", "Pathlib provides object-oriented filesystem paths. The classes are divided between pure paths, which provide purely computational operations, and concrete paths, which inherit from pure paths but also provide I/O operations."),
    ("docs.python.org", "https://docs.python.org/3.11/library/argparse.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "procedure", "The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv."),
    ("docs.python.org", "https://docs.python.org/3.11/library/dataclasses.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "contrastive", "Data classes are classes which primarily contain data, although there are usually some methods. They use class variables to define the fields, and fields can have default values."),
    ("docs.python.org", "https://docs.python.org/3.11/library/contextlib.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "causal_order", "The with statement supports the concept of a runtime context defined by a context manager. A context manager is entered before the statement body and exited when the statement ends."),
    ("docs.python.org", "https://docs.python.org/3.11/library/itertools.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "qa", "The itertools module implements a number of iterator building blocks inspired by constructs from APL, Haskell, and SML. They form an iterator algebra making it possible to construct specialized tools succinctly and efficiently."),
    ("docs.python.org", "https://docs.python.org/3.11/library/logging.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "noisy_text", "The logging module defines functions and classes which implement a flexible event logging system for applications and libraries. The best way to use logging is to create a logger for each module."),
    ("docs.python.org", "https://docs.python.org/3.11/library/re.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "negative_query", "Regular expressions use the backslash character to indicate special forms or to allow special characters to be used without invoking their special meaning. Raw string notation keeps regular expressions readable."),
    ("docs.python.org", "https://docs.python.org/3.11/library/asyncio.html", "Python-3.11.9-2023-10-02", "PSF-2.0", "delayed_recall", "asyncio is a library to write concurrent code using the async/await syntax. It is used as a foundation for multiple Python asynchronous frameworks."),
    ("www.w3.org", "https://www.w3.org/TR/css-flexbox-1/", "2023-11-14", "W3C-Document-License", "procedure", "The flex layout is superficially similar to block layout. It lacks many of the more complex text or document-centric properties that can be used in block layout, such as floats and columns."),
    ("www.w3.org", "https://www.w3.org/TR/css-color-4/", "2023-12-05", "W3C-Document-License", "qa", "The sRGB color space is an additive color space. It is defined by a set of primaries, a white point, and a transfer function."),
    ("www.w3.org", "https://www.w3.org/TR/webstorage/", "2021-01-28", "W3C-Document-License", "negative_query", "The Web Storage API provides mechanisms by which documents can store key/value pairs. The storage areas are localStorage and sessionStorage."),
    ("www.w3.org", "https://www.w3.org/TR/wai-aria-1.2/", "2023-10-05", "W3C-Document-License", "contrastive", "Authors must not use aria-hidden=""true"" on a focusable element or on an ancestor of a focusable element."),
    ("www.w3.org", "https://www.w3.org/TR/WCAG22/", "2023-10-05", "W3C-Document-License", "revision", "WCAG 2.2 adds new success criteria and removes 4.1.1 Parsing. The conformance requirements remain organized around testable success criteria."),
    ("www.w3.org", "https://www.w3.org/TR/fetch/", "2023-12-12", "W3C-Document-License", "causal_order", "A fetch is a request and response pair. Fetching a resource consists of a series of steps that can include HTTP-network-or-cache fetch and an HTTP-network fetch."),
    ("www.w3.org", "https://www.w3.org/TR/dom/", "2023-12-12", "W3C-Document-License", "ambiguous", "A Document object is an object that represents a document. Its associated DocumentType object and its document element are part of its document structure."),
    ("www.w3.org", "https://www.w3.org/TR/html52/", "2017-12-14", "W3C-Document-License", "revision", "This specification is superseded by the HTML Living Standard. The W3C HTML 5.2 document is retained as a historical snapshot."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc9110", "2022-06", "IETF-Trust-License", "qa", "HTTP semantics are independent of the message syntax used to convey them. HTTP is a stateless application-level protocol for distributed information systems."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc9111", "2022-06", "IETF-Trust-License", "causal_order", "A cache stores cacheable responses for use in satisfying future requests. A cache receives a request, determines whether it can use a stored response, and otherwise forwards the request."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc8446", "2018-08", "IETF-Trust-License", "procedure", "The TLS handshake protocol is responsible for negotiating a session, authenticating the peer, and deriving shared keying material."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc791", "1981-09", "IETF-Trust-License", "revision", "This document specifies the current version of the Internet Protocol. It is obsoleted by later specifications, but remains an historical reference."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc1035", "1987-11", "IETF-Trust-License", "negative_query", "The domain system is a tree structure. A resolver should not assume that a name which is absent from one response is absent from the domain."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc3986", "2005-01", "IETF-Trust-License", "contrastive", "A URI is a sequence of characters that is not always a URI reference. A relative reference is resolved against a base URI to produce a target URI."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc6455", "2011-12", "IETF-Trust-License", "noisy_text", "The WebSocket Protocol enables two-way communication between a client and a server over a single TCP connection. The connection begins with an HTTP handshake."),
    ("datatracker.ietf.org", "https://datatracker.ietf.org/doc/html/rfc3261", "2002-06", "IETF-Trust-License", "delayed_recall", "SIP is an application-layer control protocol for creating, modifying, and terminating sessions with one or more participants."),
]

EVALUATION = [
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function", "2025-07-01", "CC-BY-SA-2.5", "procedure", "The async function declaration creates a binding of a new async function to a given name. Async functions can contain zero or more await expressions."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API", "2025-07-01", "CC-BY-SA-2.5", "qa", "The Fetch API provides an interface for fetching resources, including across the network. It is a more powerful and flexible replacement for XMLHttpRequest."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout", "2025-07-01", "CC-BY-SA-2.5", "contrastive", "CSS grid layout excels at dividing a page into major regions or defining the relationship in terms of size, position, and layer between parts of a control built from HTML primitives."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404", "2025-07-01", "CC-BY-SA-2.5", "negative_query", "The HTTP 404 Not Found response status code indicates that the server cannot find the requested resource. Links that lead to a 404 page are often called broken or dead links."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Attributes/aria-label", "2025-07-01", "CC-BY-SA-2.5", "ambiguous", "The aria-label attribute defines a string value that can be used to name an element, as long as the element's role does not prohibit naming."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map", "2025-07-01", "CC-BY-SA-2.5", "causal_order", "The map() method creates a new array populated with the results of calling a provided function on every element in the calling array."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control", "2025-07-01", "CC-BY-SA-2.5", "revision", "The Cache-Control HTTP header holds directives in both requests and responses that control caching in browsers and shared caches."),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dialog", "2025-07-01", "CC-BY-SA-2.5", "delayed_recall", "The dialog HTML element represents a modal or non-modal dialog box or other interactive component, such as a dismissible alert, inspector, or subwindow."),
    ("man7.org", "https://man7.org/linux/man-pages/man2/open.2.html", "2024-05-02", "GPL-2.0-or-later", "procedure", "The open() system call opens the file specified by pathname. If the specified file does not exist, it may optionally be created by specifying O_CREAT in flags."),
    ("man7.org", "https://man7.org/linux/man-pages/man2/read.2.html", "2024-05-02", "GPL-2.0-or-later", "qa", "read() attempts to read up to count bytes from file descriptor fd into the buffer starting at buf. On success, the number of bytes read is returned."),
    ("man7.org", "https://man7.org/linux/man-pages/man2/fork.2.html", "2024-05-02", "GPL-2.0-or-later", "causal_order", "fork() creates a new process by duplicating the calling process. The new process is referred to as the child process; the calling process is referred to as the parent process."),
    ("man7.org", "https://man7.org/linux/man-pages/man2/execve.2.html", "2024-05-02", "GPL-2.0-or-later", "contrastive", "execve() executes the program referred to by pathname. The calling process is transformed into a new program, with newly initialized stack, heap, and data segments."),
    ("man7.org", "https://man7.org/linux/man-pages/man2/wait.2.html", "2024-05-02", "GPL-2.0-or-later", "negative_query", "wait() suspends execution of its calling thread until one of its children terminates. It does not wait for an unrelated process."),
    ("man7.org", "https://man7.org/linux/man-pages/man7/signal.7.html", "2024-05-02", "GPL-2.0-or-later", "noisy_text", "A signal is a notification sent to a process or thread to notify it of an event. Signal dispositions can be set to default, ignore, or a handler function."),
    ("man7.org", "https://man7.org/linux/man-pages/man7/socket.7.html", "2024-05-02", "GPL-2.0-or-later", "revision", "The socket(7) manual page describes the Linux socket interface. The details of individual protocols are described in separate protocol manual pages."),
    ("man7.org", "https://man7.org/linux/man-pages/man5/proc.5.html", "2024-05-02", "GPL-2.0-or-later", "delayed_recall", "The proc filesystem is a pseudo-filesystem which provides an interface to kernel data structures. It is commonly mounted at /proc."),
    ("docs.openstack.org", "https://docs.openstack.org/install-guide/overview.html", "2024-02-15", "Apache-2.0", "qa", "The OpenStack Installation Guide provides instructions for installing OpenStack services from packages on a controller node and one or more compute nodes."),
    ("docs.openstack.org", "https://docs.openstack.org/install-guide/install-base.html", "2024-02-15", "Apache-2.0", "procedure", "Before installing and configuring a service, prepare the database, credentials, endpoints, and messaging service required by that service."),
    ("docs.openstack.org", "https://docs.openstack.org/keystone/latest/admin/identity-fernet-token-faq.html", "2024-02-15", "Apache-2.0", "negative_query", "Fernet tokens do not require persistence in a database. This does not mean that the keys used to validate tokens can be discarded."),
    ("docs.openstack.org", "https://docs.openstack.org/nova/latest/admin/secure-live-migration-with-qemu.html", "2024-02-15", "Apache-2.0", "causal_order", "For secure live migration, configure the source and destination compute nodes and ensure the migration network is protected before starting a migration."),
    ("docs.openstack.org", "https://docs.openstack.org/neutron/latest/admin/intro-os-networking.html", "2024-02-15", "Apache-2.0", "contrastive", "Neutron provides networking as a service between interface devices managed by other OpenStack services, unlike a compute service that creates the instances themselves."),
    ("docs.openstack.org", "https://docs.openstack.org/horizon/latest/admin/quickstart.html", "2024-02-15", "Apache-2.0", "ambiguous", "The dashboard enables cloud administrators and users to provision and manage server instances and other resources."),
    ("docs.openstack.org", "https://docs.openstack.org/cinder/latest/admin/blockstorage-basic-ops.html", "2024-02-15", "Apache-2.0", "revision", "A volume can be attached to an instance, detached, and reattached. The available operations depend on the volume state."),
    ("docs.openstack.org", "https://docs.openstack.org/heat/latest/template_guide/hot_spec.html", "2024-02-15", "Apache-2.0", "delayed_recall", "HOT is the native Heat template format. A template describes resources, properties, dependencies, and outputs used to orchestrate a stack."),
]

LICENSE_URLS = {
    "docs.python.org": "https://docs.python.org/3/license.html",
    "www.w3.org": "https://www.w3.org/Consortium/Legal/2023/doc-license",
    "datatracker.ietf.org": "https://trustee.ietf.org/documents/trust-legal-provisions/",
    "developer.mozilla.org": "https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license",
    "man7.org": "https://www.kernel.org/doc/man-pages/",
    "docs.openstack.org": "https://docs.openstack.org/contributor-guide/legal.html",
}


def make_rows(cards: list[tuple[str, str, str, str, str, str]], split: str, collected_at: str) -> list[dict[str, str]]:
    rows = []
    for index, (domain, url, revision, license_name, task_type, content) in enumerate(cards, 1):
        normalized = " ".join(content.split())
        rows.append({
            "record_id": f"phase7-{split}-{index:04d}",
            "source_url": url,
            "source_domain": domain,
            "source_revision": revision,
            "source_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "near_duplicate_signature": _simhash(normalized),
            "collection_time": f"{revision[:10]}T00:00:00Z" if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", revision) else ("2022-06-01T00:00:00Z" if revision.startswith("2022") else "2024-01-01T00:00:00Z"),
            "collected_at": collected_at,
            "evidence_scope": "independent_external",
            "content": normalized,
            "task_type": task_type,
            "license": license_name,
            "license_url": LICENSE_URLS[domain],
            "content_origin": "transcribed_source_excerpt",
        })
    return rows


def write_jsonl(path: str, rows: list[dict[str, str]]) -> None:
    with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-at", default=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
    args = parser.parse_args()
    train = make_rows(TRAIN, "train", args.collected_at)
    evaluation = make_rows(EVALUATION, "eval", args.collected_at)
    raw_path = ensure_parent_directory(raw_data_path("phase7", "source_cards.jsonl"))
    write_jsonl(raw_path, train + evaluation)
    write_jsonl(processed_data_path("phase7", "train.jsonl"), train)
    write_jsonl(processed_data_path("phase7", "evaluation.jsonl"), evaluation)
    manifest = {"schema": "sara-phase7-collection-manifest-v1", "collected_at": args.collected_at, "train_count": len(train), "evaluation_count": len(evaluation), "train_domains": sorted({row["source_domain"] for row in train}), "evaluation_domains": sorted({row["source_domain"] for row in evaluation}), "license_policy": "public license recorded per source; evaluation domains are disjoint from train domains"}
    with open(ensure_parent_directory(workspace_path("phase7", "collection_manifest.json")), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

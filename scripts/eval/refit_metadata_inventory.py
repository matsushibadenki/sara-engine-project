"""Summarize REFIT motion and contact metadata without reading observations."""

import argparse
from collections import Counter
import json
import sys
import xml.etree.ElementTree as ET


NAMESPACE = "{http://www.refitsmarthomes.org}"
MAX_BYTES = 1_755_414
CHANNELS = ("Motion", "Open/Closed")


def inventory(source):
    payload = source.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Survey XML exceeds the published file size")
    # Accept UTF-8 only so the declaration check applies to the parsed text.
    xml = payload.decode("utf-8-sig")
    if "<!DOCTYPE" in xml:
        raise ValueError("DTD declarations are unsupported")
    root = ET.fromstring(xml)
    if root.tag != NAMESPACE + "RefitXML":
        raise ValueError("Expected a REFIT survey document")
    buildings = list(root.iter(NAMESPACE + "Building"))
    if not buildings:
        raise ValueError("Survey contains no buildings")
    summary = Counter()
    for building in buildings:
        counts = Counter(
            variable.get("variableType")
            for variable in building.iter(NAMESPACE + "TimeSeriesVariable")
        )
        for channel in CHANNELS:
            summary[channel] += counts[channel]
        summary["buildings_with_both"] += all(counts[channel] for channel in CHANNELS)
    return {
        "buildings": len(buildings),
        "buildings_with_both": summary["buildings_with_both"],
        "motion_variables": summary["Motion"],
        "open_closed_variables": summary["Open/Closed"],
        "coverage_verified": False,
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(inventory(sys.stdin.buffer), sort_keys=True))


if __name__ == "__main__":
    main()

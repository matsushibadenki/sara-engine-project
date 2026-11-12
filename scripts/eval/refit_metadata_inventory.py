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
        parents = {child: parent for parent in building.iter() for child in parent}
        counts = Counter()
        for variable in building.iter(NAMESPACE + "TimeSeriesVariable"):
            channel = variable.get("variableType")
            if channel not in CHANNELS:
                continue
            counts[channel] += 1
            if variable.get("hasMissingData") is None:
                summary["without_missing_data_flag"] += 1
            if channel == "Open/Closed":
                sensor = parents[variable]
                if sensor.tag == NAMESPACE + "Sensor" and parents[sensor] is building:
                    summary["building_level_contacts"] += 1
        for channel in CHANNELS:
            summary[channel] += counts[channel]
        summary["buildings_with_both"] += all(counts[channel] for channel in CHANNELS)
    return {
        "buildings": len(buildings),
        "buildings_with_both": summary["buildings_with_both"],
        "motion_variables": summary["Motion"],
        "open_closed_variables": summary["Open/Closed"],
        "building_level_contacts": summary["building_level_contacts"],
        "without_missing_data_flag": summary["without_missing_data_flag"],
        "coverage_verified": False,
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(inventory(sys.stdin.buffer), sort_keys=True))


if __name__ == "__main__":
    main()

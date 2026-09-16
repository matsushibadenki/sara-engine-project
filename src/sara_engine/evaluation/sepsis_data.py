"""Hash-pinned Sepsis Cases trace loading for independent R2 evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import gzip, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from sara_engine.utils.project_paths import raw_data_path

SOURCE_SHA256 = "709c52340306415952811b9b9c5dc6bcc8f8d47d583eba39df9a538459dc543a"
TRAIN_END = (datetime.fromisoformat("2014-09-18T22:24:34+02:00"), "LQ")
DEVELOPMENT_END = (datetime.fromisoformat("2014-11-15T16:48:51+01:00"), "RW")

@dataclass(frozen=True)
class Event:
    activity: str; lifecycle: str; timestamp: datetime

@dataclass(frozen=True)
class Trace:
    case_id: str; events: tuple[Event, ...]

def _local(tag: str) -> str: return tag.rsplit("}", 1)[-1]
def _attributes(element):
    return {child.attrib["key"]: child.attrib.get("value", "") for child in element
            if _local(child.tag) in ("string", "date", "int", "float", "boolean") and "key" in child.attrib}

def load_traces() -> list[Trace]:
    source = Path(raw_data_path("r2_sepsis", "Sepsis Cases - Event Log.xes.gz"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Sepsis source SHA-256 mismatch")
    traces = []
    with gzip.open(source, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local(element.tag) != "trace": continue
            attrs = _attributes(element); events = []
            for child in element:
                if _local(child.tag) == "event":
                    row = _attributes(child); events.append(Event(row["concept:name"], row.get("lifecycle:transition", "missing"), datetime.fromisoformat(row["time:timestamp"])))
            traces.append(Trace(attrs["concept:name"], tuple(events))); element.clear()
    traces.sort(key=lambda trace: (trace.events[0].timestamp, trace.case_id)); return traces

def split_name(trace: Trace) -> str:
    key = (trace.events[0].timestamp, trace.case_id)
    return "training" if key <= TRAIN_END else "development" if key <= DEVELOPMENT_END else "frozen_test"

__all__ = ["Event", "Trace", "load_traces", "split_name"]

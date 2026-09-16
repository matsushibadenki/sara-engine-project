"""Hash-pinned BPI 2012 trace loading shared by evaluation code."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import gzip, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from sara_engine.utils.project_paths import raw_data_path

SOURCE_SHA256="5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3"
DEV_BOUNDARY=datetime.fromisoformat("2012-01-19T10:02:23.184000+01:00")
TEST_BOUNDARY=datetime.fromisoformat("2012-02-11T10:46:32.413000+01:00")
GAP_BOUNDS_SECONDS=(0.0,60.0,300.0,1800.0,7200.0,28800.0,86400.0,259200.0)

@dataclass(frozen=True)
class Event:
    activity:str; lifecycle:str; timestamp:datetime

@dataclass(frozen=True)
class Trace:
    case_id:str; events:tuple[Event,...]

def _local(tag:str)->str:return tag.rsplit("}",1)[-1]
def _attributes(element):
    return {child.attrib["key"]:child.attrib.get("value","") for child in element
            if _local(child.tag) in ("string","date","int","float","boolean") and "key" in child.attrib}

def load_traces()->list[Trace]:
    source=Path(raw_data_path("r2_bpi2012","BPI_Challenge_2012.xes.gz"))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:raise ValueError("BPI 2012 source SHA-256 mismatch")
    traces=[]
    with gzip.open(source,"rb") as handle:
        for _,element in ET.iterparse(handle,events=("end",)):
            if _local(element.tag)!="trace":continue
            attrs=_attributes(element);events=[]
            for child in element:
                if _local(child.tag)=="event":
                    row=_attributes(child);events.append(Event(row["concept:name"],row.get("lifecycle:transition","missing"),datetime.fromisoformat(row["time:timestamp"])))
            traces.append(Trace(attrs["concept:name"],tuple(events)));element.clear()
    traces.sort(key=lambda trace:(trace.events[0].timestamp,trace.case_id));return traces

def split_name(trace:Trace)->str:
    start=trace.events[0].timestamp
    return "training" if start<DEV_BOUNDARY else "development" if start<TEST_BOUNDARY else "frozen_test"

def gap_bucket(seconds:float)->int:
    for index,upper in enumerate(GAP_BOUNDS_SECONDS):
        if seconds<=upper:return index
    return len(GAP_BOUNDS_SECONDS)

__all__=["Event","Trace","gap_bucket","load_traces","split_name"]

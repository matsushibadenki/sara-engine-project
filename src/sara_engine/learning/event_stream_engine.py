"""Serializable bounded event encoder plus normalized-hybrid learning runtime."""

from __future__ import annotations
from bisect import bisect_left
from dataclasses import asdict, dataclass
import fcntl, hashlib, json, os, stat
from pathlib import Path
from typing import Sequence

from sara_engine.learning.normalized_hybrid import BoundedNormalizedHybrid, NormalizedHybridConfig, NormalizedHybridPrediction, NormalizedHybridUpdate
from sara_engine.utils.project_paths import ensure_parent_directory, model_path

@dataclass(frozen=True)
class EventRouteConfig:
    gap_bounds_seconds: tuple[float,...]=(0.,60.,300.,1800.,7200.,28800.,86400.,259200.)
    prefix_length_bounds: tuple[int,...]=(1,2,4,8,16,32,64,128)
    include_calendar_route: bool=False
    constant_gap: bool=False
    max_routes: int=4096
    def __post_init__(self):
        if not self.gap_bounds_seconds or tuple(sorted(self.gap_bounds_seconds))!=self.gap_bounds_seconds:raise ValueError("gap bounds must be sorted")
        if not self.prefix_length_bounds or tuple(sorted(self.prefix_length_bounds))!=self.prefix_length_bounds:raise ValueError("prefix bounds must be sorted")
        if type(self.max_routes) is not int or self.max_routes<1:raise ValueError("max_routes must be positive")

class BoundedEventRouteEncoder:
    SCHEMA="sara-bounded-event-route-encoder-v1"
    def __init__(self,config:EventRouteConfig|None=None):self.config=config or EventRouteConfig();self._routes={}
    def _bucket(self,seconds):
        if self.config.constant_gap and seconds>0:return 1
        for index,upper in enumerate(self.config.gap_bounds_seconds):
            if seconds<=upper:return index
        return len(self.config.gap_bounds_seconds)
    def _route(self,key):
        if key not in self._routes:
            if len(self._routes)>=self.config.max_routes:raise ValueError("event route budget exceeded")
            self._routes[key]=len(self._routes)
        return self._routes[key]
    def encode(self,events:Sequence,index:int)->tuple[int,...]:
        if type(index) is not int or index<0 or index>=len(events):raise ValueError("event index is invalid")
        event=events[index];previous=events[index-1].activity if index else "<BOS>";previous2=events[index-2].activity if index>1 else "<BOS2>"
        gap=0. if index==0 else (event.timestamp-events[index-1].timestamp).total_seconds()
        keys=[("a",event.activity),("aa",previous,event.activity),("aaa",previous2,previous,event.activity),("al",event.activity,event.lifecycle),
            ("alg",event.activity,event.lifecycle,self._bucket(gap)),("aag",previous,event.activity,self._bucket(gap))]
        if self.config.include_calendar_route:keys.append(("hw",event.timestamp.hour//4,event.timestamp.weekday()))
        keys.append(("p",bisect_left(self.config.prefix_length_bounds,index+1)))
        return tuple(self._route(key) for key in keys)
    def state_dict(self):
        return {"schema":self.SCHEMA,"config":asdict(self.config),"routes":[[list(key),route] for key,route in sorted(self._routes.items(),key=lambda item:item[1])]}
    @classmethod
    def from_state_dict(cls,state):
        if type(state) is not dict or state.get("schema")!=cls.SCHEMA:raise ValueError("encoder schema is invalid")
        raw_config=state.get("config",{});raw_config["gap_bounds_seconds"]=tuple(raw_config.get("gap_bounds_seconds",()));raw_config["prefix_length_bounds"]=tuple(raw_config.get("prefix_length_bounds",()))
        encoder=cls(EventRouteConfig(**raw_config));rows=state.get("routes")
        if type(rows) is not list or len(rows)>encoder.config.max_routes:raise ValueError("encoder routes are invalid")
        for expected,(raw_key,route) in enumerate(rows):
            if type(raw_key) is not list or route!=expected:raise ValueError("encoder route ordering is invalid")
            encoder._routes[tuple(raw_key)]=route
        return encoder

@dataclass(frozen=True)
class CheckpointReceipt:
    path:str
    generation:int
    payload_sha256:str

class BoundedEventStreamEngine:
    SCHEMA="sara-bounded-event-stream-engine-v1"
    ARTIFACT_SCHEMA="sara-bounded-event-stream-artifact-v1"
    MAX_ARTIFACT_BYTES=32*1024*1024
    def __init__(self,labels:Sequence[str],*,route_config:EventRouteConfig|None=None,hybrid_config:NormalizedHybridConfig|None=None):
        self.encoder=BoundedEventRouteEncoder(route_config);self.hybrid=BoundedNormalizedHybrid(labels,hybrid_config)
        expected=8 if self.encoder.config.include_calendar_route else 7
        if self.hybrid.config.max_active<expected:raise ValueError("hybrid active budget cannot encode configured event routes")
    def predict(self,events:Sequence,index:int)->NormalizedHybridPrediction:
        current=events[index].activity;previous=events[index-1].activity if index else "<BOS>"
        return self.hybrid.predict(self.encoder.encode(events,index),context=(previous,current))
    def observe(self,prediction:NormalizedHybridPrediction,target:str)->NormalizedHybridUpdate:return self.hybrid.observe(prediction,target)
    def state_dict(self):return {"schema":self.SCHEMA,"encoder":self.encoder.state_dict(),"hybrid":self.hybrid.state_dict()}
    @classmethod
    def from_state_dict(cls,state):
        if type(state) is not dict or state.get("schema")!=cls.SCHEMA:raise ValueError("event stream engine schema is invalid")
        encoder=BoundedEventRouteEncoder.from_state_dict(state.get("encoder"));hybrid=BoundedNormalizedHybrid.from_state_dict(state.get("hybrid"))
        engine=cls(hybrid.labels,route_config=encoder.config,hybrid_config=hybrid.config);engine.encoder=encoder;engine.hybrid=hybrid;return engine
    @classmethod
    def _read_artifact(cls,path:Path):
        size=path.stat().st_size
        if size>cls.MAX_ARTIFACT_BYTES:raise ValueError("event stream artifact exceeds size limit")
        if stat.S_IMODE(path.stat().st_mode)&0o077:raise ValueError("event stream artifact permissions are not owner-only")
        document=json.loads(path.read_text())
        if type(document) is not dict:raise ValueError("event stream artifact must be an object")
        if document.get("schema")==cls.SCHEMA:return document,0
        if document.get("schema")!=cls.ARTIFACT_SCHEMA or type(document.get("payload_sha256")) is not str or type(document.get("payload")) is not dict:raise ValueError("event stream artifact schema is invalid")
        generation=document.get("generation")
        if type(generation) is not int or generation<1:raise ValueError("event stream artifact generation is invalid")
        canonical=json.dumps(document["payload"],sort_keys=True,separators=(",",":")).encode()
        if hashlib.sha256(canonical).hexdigest()!=document["payload_sha256"]:raise ValueError("event stream artifact checksum mismatch")
        return document["payload"],generation
    def publish(self,filename,*,expected_generation:int|None)->CheckpointReceipt:
        if type(filename) is not str or not filename or Path(filename).name!=filename:raise ValueError("filename must be a plain file name")
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation<0):raise ValueError("expected_generation is invalid")
        path=Path(ensure_parent_directory(model_path("event_stream_engine",filename)));lock_path=path.with_suffix(path.suffix+".lock")
        lock_fd=os.open(lock_path,os.O_RDWR|os.O_CREAT,0o600)
        try:
            os.fchmod(lock_fd,0o600);fcntl.flock(lock_fd,fcntl.LOCK_EX)
            current_generation=self._read_artifact(path)[1] if path.exists() else 0
            if expected_generation is not None and current_generation!=expected_generation:raise ValueError("checkpoint generation conflict")
            generation=current_generation+1;state=self.state_dict();canonical=json.dumps(state,sort_keys=True,separators=(",",":")).encode();digest=hashlib.sha256(canonical).hexdigest()
            envelope={"schema":self.ARTIFACT_SCHEMA,"generation":generation,"payload_sha256":digest,"payload":state};payload=json.dumps(envelope,sort_keys=True,separators=(",",":")).encode()
            if len(payload)>self.MAX_ARTIFACT_BYTES:raise ValueError("event stream artifact exceeds size limit")
            temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp");temporary_fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            try:
                with os.fdopen(temporary_fd,"wb") as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
                os.replace(temporary,path);os.chmod(path,0o600)
                directory_fd=os.open(path.parent,os.O_RDONLY)
                try:os.fsync(directory_fd)
                finally:os.close(directory_fd)
            finally:
                if temporary.exists():temporary.unlink()
            return CheckpointReceipt(str(path),generation,digest)
        finally:
            fcntl.flock(lock_fd,fcntl.LOCK_UN);os.close(lock_fd)
    def save(self,filename):
        return self.publish(filename,expected_generation=None).path
    @classmethod
    def load(cls,filename):
        if type(filename) is not str or Path(filename).name!=filename:raise ValueError("filename must be a plain file name")
        payload,_=cls._read_artifact(Path(model_path("event_stream_engine",filename)));return cls.from_state_dict(payload)
    @classmethod
    def load_with_generation(cls,filename):
        if type(filename) is not str or Path(filename).name!=filename:raise ValueError("filename must be a plain file name")
        payload,generation=cls._read_artifact(Path(model_path("event_stream_engine",filename)));return cls.from_state_dict(payload),generation

__all__=["BoundedEventRouteEncoder","BoundedEventStreamEngine","CheckpointReceipt","EventRouteConfig"]

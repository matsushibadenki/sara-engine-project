"""Generic bounded next-event hybrid with local updates and normalized routing."""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json, math, os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sara_engine.learning.confidence_router import ConfidenceRouteConfig, route_prediction
from sara_engine.learning.normalized_router import route_normalized
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import ensure_parent_directory, model_path

@dataclass(frozen=True)
class NormalizedHybridConfig:
    learning_rate: float = .30
    weight_cap: float = 2.0
    normalized_threshold: float = .6223091976516634
    routing_mode: str = "normalized"
    base_probability_cap: float = .75
    local_score_margin: float = .10
    minimum_base_support: int = 16
    max_routes: int = 4096
    max_active: int = 8
    max_classes: int = 24
    max_context_parts: int = 4
    max_contexts: int = 4096
    explicit_neurons: bool = True
    compact_event_units: bool = False

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or not 0 <= self.learning_rate <= 1: raise ValueError("learning_rate is invalid")
        if not math.isfinite(self.weight_cap) or not 0 < self.weight_cap <= 16: raise ValueError("weight_cap is invalid")
        if not math.isfinite(self.normalized_threshold) or self.normalized_threshold < 0: raise ValueError("normalized_threshold is invalid")
        if self.routing_mode not in ("normalized","absolute"): raise ValueError("routing_mode is invalid")
        if not math.isfinite(self.base_probability_cap) or not 0 <= self.base_probability_cap <= 1: raise ValueError("base_probability_cap is invalid")
        if not math.isfinite(self.local_score_margin) or self.local_score_margin < 0: raise ValueError("local_score_margin is invalid")
        for name in ("minimum_base_support","max_routes","max_active","max_classes","max_context_parts","max_contexts"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError(f"{name} must be positive")
        if self.max_active>self.max_routes: raise ValueError("max_active exceeds max_routes")
        if self.compact_event_units and not self.explicit_neurons: raise ValueError("compact event units require event mode")

@dataclass(frozen=True)
class NormalizedHybridPrediction:
    sequence: int
    predicted: str
    probabilities: tuple[tuple[str,float],...]
    active: tuple[int,...]
    context: tuple[str,...]
    local_predicted: str
    overridden: bool

@dataclass(frozen=True)
class NormalizedHybridUpdate:
    decision: str
    local_updates: int
    base_support: int

class BoundedNormalizedHybrid:
    SCHEMA="sara-bounded-normalized-hybrid-v1"
    def __init__(self,labels:Sequence[str],config:Optional[NormalizedHybridConfig]=None):
        self.config=config or NormalizedHybridConfig();self.labels=tuple(sorted(set(labels)))
        if not self.labels or len(self.labels)>self.config.max_classes:raise ValueError("class vocabulary is invalid")
        self._weights={};self._contexts=defaultdict(Counter);self._routes=set();self._neurons={};self._compact_units=set();self._pending=None;self._sequence=0

    def _validate_context(self,context:Iterable[str])->tuple[str,...]:
        parts=tuple(context)
        if not parts or len(parts)>self.config.max_context_parts or any(type(p) is not str or not p or len(p)>128 for p in parts):raise ValueError("context is invalid")
        if parts not in self._contexts and len(self._contexts)>=self.config.max_contexts:raise ValueError("context budget exceeded")
        return parts

    def _activate(self,active:Iterable[int])->tuple[int,...]:
        admitted=[];seen=set()
        for index,route in enumerate(active):
            if index>=self.config.max_active:raise ValueError("active route budget exceeded")
            if type(route) is not int or route<0 or route in seen:raise ValueError("active routes must be unique non-negative integers")
            seen.add(route);admitted.append(route)
        if not admitted:raise ValueError("at least one route is required")
        if len(self._routes|seen)>self.config.max_routes:raise ValueError("route budget exceeded")
        if self.config.compact_event_units:
            self._compact_units.update(admitted)
        elif self.config.explicit_neurons:
            for route in admitted:
                unit=self._neurons.get(route)
                if unit is None:unit=Neuron(route,num_branches=1);self._neurons[route]=unit
                unit.v=0.0;unit.spike=False;unit.refractory_time=0;unit.active_branches.clear();unit.add_input_to_branch(0,1.6)
                if not unit.step():raise RuntimeError("hybrid route neuron did not spike")
        return tuple(admitted)

    def predict(self,active:Iterable[int],*,context:Iterable[str])->NormalizedHybridPrediction:
        if self._pending is not None:raise ValueError("resolve the pending prediction first")
        routes=self._activate(active);key=self._validate_context(context);table=self._contexts[key]
        total=sum(table.values())+len(self.labels);base_probabilities={label:(table[label]+1.0)/total for label in self.labels}
        base_predicted=max(self.labels,key=lambda label:(base_probabilities[label],-self.labels.index(label)))
        local_scores={label:sum(self._weights.get((route,label),0.0) for route in routes) for label in self.labels}
        local_predicted=max(self.labels,key=lambda label:(local_scores[label],-self.labels.index(label)))
        if self.config.routing_mode=="normalized":
            decision=route_normalized(labels=self.labels,base_predicted=base_predicted,base_probabilities=base_probabilities,base_support=sum(table.values()),
                local_predicted=local_predicted,local_scores=local_scores,threshold=self.config.normalized_threshold,minimum_base_support=self.config.minimum_base_support)
        else:
            decision=route_prediction(labels=self.labels,base_predicted=base_predicted,base_probabilities=base_probabilities,base_support=sum(table.values()),
                local_predicted=local_predicted,local_scores=local_scores,config=ConfidenceRouteConfig(self.config.base_probability_cap,self.config.local_score_margin,self.config.minimum_base_support))
        self._sequence+=1;receipt=NormalizedHybridPrediction(self._sequence,decision.predicted,decision.probabilities,routes,key,local_predicted,decision.overridden);self._pending=receipt;return receipt

    def observe(self,prediction:NormalizedHybridPrediction,target:str)->NormalizedHybridUpdate:
        if prediction is not self._pending:raise ValueError("prediction receipt is not pending")
        if target not in self.labels:raise ValueError("unknown target")
        self._routes.update(prediction.active);updates=0
        if target!=prediction.local_predicted and self.config.learning_rate:
            delta=self.config.learning_rate/len(prediction.active)
            for route in prediction.active:
                for label,sign in ((target,1.0),(prediction.local_predicted,-1.0)):
                    key=(route,label);self._weights[key]=max(-self.config.weight_cap,min(self.config.weight_cap,self._weights.get(key,0.0)+sign*delta));updates+=1
        table=self._contexts[prediction.context];table[target]+=1;support=sum(table.values());self._pending=None
        return NormalizedHybridUpdate("correct" if target==prediction.predicted else "updated",updates,support)

    def state_dict(self)->dict:
        if self._pending is not None:raise ValueError("cannot serialize a pending prediction")
        return {"schema":self.SCHEMA,"labels":list(self.labels),"config":asdict(self.config),"sequence":self._sequence,
            "routes":sorted(self._routes),"weights":[[route,label,value] for (route,label),value in sorted(self._weights.items())],
            "contexts":[[list(context),[[label,count] for label,count in sorted(counts.items())]] for context,counts in sorted(self._contexts.items())]}

    @classmethod
    def from_state_dict(cls,state:dict)->"BoundedNormalizedHybrid":
        if type(state) is not dict or state.get("schema")!=cls.SCHEMA:raise ValueError("hybrid state schema is invalid")
        model=cls(state.get("labels",()),NormalizedHybridConfig(**state.get("config",{})))
        sequence=state.get("sequence");routes=state.get("routes");weights=state.get("weights");contexts=state.get("contexts")
        if type(sequence) is not int or sequence<0 or type(routes) is not list or type(weights) is not list or type(contexts) is not list:raise ValueError("hybrid state fields are invalid")
        if len(routes)>model.config.max_routes or len(contexts)>model.config.max_contexts:raise ValueError("hybrid state exceeds bounds")
        model._sequence=sequence;model._routes=set(routes)
        for route,label,value in weights:
            if route not in model._routes or label not in model.labels or not math.isfinite(value) or abs(value)>model.config.weight_cap:raise ValueError("hybrid weight is invalid")
            model._weights[(route,label)]=value
        for raw_context,raw_counts in contexts:
            context=model._validate_context(raw_context);counts=Counter()
            for label,count in raw_counts:
                if label not in model.labels or type(count) is not int or count<0:raise ValueError("hybrid count is invalid")
                counts[label]=count
            model._contexts[context]=counts
        if model.config.compact_event_units:model._compact_units=set(model._routes)
        elif model.config.explicit_neurons:model._neurons={route:Neuron(route,num_branches=1) for route in model._routes}
        return model

    def save(self,filename:str)->str:
        if type(filename) is not str or not filename or Path(filename).name!=filename:raise ValueError("filename must be a plain file name")
        path=Path(ensure_parent_directory(model_path("normalized_hybrid",filename)));temporary=path.with_suffix(path.suffix+".tmp")
        payload=json.dumps(self.state_dict(),sort_keys=True,separators=(",",":")).encode();
        with open(temporary,"wb") as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path);return str(path)

    @classmethod
    def load(cls,filename:str)->"BoundedNormalizedHybrid":
        if type(filename) is not str or Path(filename).name!=filename:raise ValueError("filename must be a plain file name")
        return cls.from_state_dict(json.loads(Path(model_path("normalized_hybrid",filename)).read_text()))

    def status(self,lang="en"):
        templates={"en":"Normalized hybrid: {routes} routes, {contexts} contexts","ja":"正規化ハイブリッド: {routes}経路、{contexts}文脈","zh-CN":"归一化混合模型：{routes}条路径，{contexts}个上下文"}
        return templates.get(lang,templates["en"]).format(routes=len(self._routes),contexts=len(self._contexts))

__all__=["BoundedNormalizedHybrid","NormalizedHybridConfig","NormalizedHybridPrediction","NormalizedHybridUpdate"]

"""Symbol-disjoint local transition-rule transfer benchmark."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,random
from typing import Sequence
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size

FAMILIES=("relative_direction","relative_equality","relative_distance","two_transition_composition","categorical_control")
RELATIONAL_FAMILIES=FAMILIES[:4]
@dataclass(frozen=True)
class TransitionEpisode: identity:str;family:str;symbols:tuple[int,...];label:int
@dataclass(frozen=True)
class TransitionReceipt: owner:int;predicted:int;features:tuple[int,...];work:int

def _pair(rng,symbols,predicate,wanted):
 for _ in range(100):
  a=rng.choice(symbols);b=rng.choice(symbols)
  if bool(predicate(a,b))==bool(wanted):return a,b
 raise RuntimeError("cannot generate balanced pair")

def generate_transition_episodes(*,seeds:Sequence[int],symbols:Sequence[int],count_per_family:int,split:str,namespace:str="local-transition-rule-v1")->list[TransitionEpisode]:
 rows=[]
 for seed in seeds:
  for fi,family in enumerate(FAMILIES):
   for index in range(count_per_family):
    label=index%2;rng=random.Random(seed*1000003+fi*10007+index*101)
    if family=="relative_direction": values=_pair(rng,symbols,lambda a,b:b>a,label)
    elif family=="relative_equality": values=_pair(rng,symbols,lambda a,b:a==b,label)
    elif family=="relative_distance": values=_pair(rng,symbols,lambda a,b:abs(b-a)>=3,label)
    elif family=="two_transition_composition":
     for _ in range(100):
      values=(rng.choice(symbols),rng.choice(symbols),rng.choice(symbols));r1=(values[1]>values[0])-(values[1]<values[0]);r2=(values[2]>values[1])-(values[2]<values[1])
      if bool(r1==r2 and r1!=0)==bool(label):break
     else:raise RuntimeError("cannot generate composition")
    else: values=(rng.choice(symbols),99 if label else 98)
    rows.append(TransitionEpisode(f"{namespace}:{split}:{seed}:{family}:{index}",family,tuple(values),label))
 random.Random(sum(seeds)+(1 if split=="development" else 0)).shuffle(rows);return rows

class TransitionRuleLearner:
 ARMS=("L_categorical_pair","R_relational_transition","S_relational_composition")
 def __init__(self,arm,*,intervention="none",reserve=0):
  if arm not in self.ARMS or intervention not in ("none","relation_shuffle","composition_reset"):raise ValueError("invalid arm")
  self.arm=arm;self.intervention=intervention;self.weights={};self.ids={};self.pending=None;self.reserve=bytearray(reserve)
 def _id(self,key):
  if key not in self.ids:
   if len(self.ids)>=1024:raise ValueError("feature budget")
   self.ids[key]=len(self.ids)
  return self.ids[key]
 @staticmethod
 def _relation(a,b):return ((b>a)-(b<a),0 if a==b else (1 if abs(b-a)<=2 else 2))
 def predict(self,episode):
  if self.pending is not None:raise ValueError("pending")
  values=episode.symbols;keys=[]
  if self.arm=="L_categorical_pair":
   keys=[("symbol",v) for v in values]+[("pair",values[i-1],v) for i,v in enumerate(values) if i]
  else:
   relations=[self._relation(values[i-1],v) for i,v in enumerate(values) if i]
   if self.intervention=="relation_shuffle":
    rng=random.Random(sum(ord(c) for c in episode.identity));relations=[(rng.randrange(-1,2),rng.randrange(3)) for _ in relations]
   keys=[("rel",*relation) for relation in relations]
   # Fixed control tokens preserve a non-relational sanity task in every arm.
   keys.extend(("marker",v) for v in values if v in (98,99))
   if self.arm=="S_relational_composition" and self.intervention!="composition_reset" and len(relations)>=2:keys.append(("compose",relations[-2],relations[-1]))
  features=tuple(dict.fromkeys(self._id(k) for k in keys));score=sum(self.weights.get(f,0.) for f in features);r=TransitionReceipt(id(self),int(score>0),features,len(values)+len(keys));self.pending=r;return r
 def observe(self,r,outcome):
  if r is not self.pending or r.owner!=id(self):raise ValueError("receipt")
  n=0
  if r.predicted!=outcome and r.features:
   delta=(.3 if outcome else -.3)/len(r.features)
   for f in r.features:self.weights[f]=max(-2.,min(2.,self.weights.get(f,0.)+delta));n+=1
  self.pending=None;return n
 def state_bytes(self):return _deep_size((self.weights,self.ids,self.reserve))

def run_transition_arm(arm,training,development,*,intervention="none",outcome_shuffle_seed=None,reserve=0,development_offset=0):
 learner=TransitionRuleLearner(arm,intervention=intervention,reserve=reserve);train_y=[e.label for e in training];dev_y=[e.label for e in development]
 if outcome_shuffle_seed is not None:random.Random(outcome_shuffle_seed).shuffle(train_y);random.Random(outcome_shuffle_seed+1).shuffle(dev_y)
 updates=0;work=0
 for e,y in zip(training,train_y):r=learner.predict(e);work=max(work,r.work);updates+=learner.observe(r,y)
 totals={f:0 for f in FAMILIES};correct={f:0 for f in FAMILIES};rows=[]
 for e,y in zip(development,dev_y):
  shifted=TransitionEpisode(e.identity,e.family,tuple(v+development_offset if v not in (98,99) else v for v in e.symbols),e.label);r=learner.predict(shifted);work=max(work,r.work);totals[e.family]+=1;correct[e.family]+=int(r.predicted==e.label);rows.append((e.identity,r.predicted));updates+=learner.observe(r,y)
 by={f:correct[f]/totals[f] for f in FAMILIES};return {"accuracy":sum(correct.values())/len(development),"relational_accuracy":sum(correct[f] for f in RELATIONAL_FAMILIES)/sum(totals[f] for f in RELATIONAL_FAMILIES),"accuracy_by_family":by,"trace_sha256":hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest(),"rows":rows,"updates":updates,"maximum_event_work":work,"state_bytes":learner.state_bytes(),"feature_count":len(learner.ids)}

__all__=["FAMILIES","RELATIONAL_FAMILIES","TransitionRuleLearner","generate_transition_episodes","run_transition_arm"]

CONTEXT_BY_FAMILY={family:index for index,family in enumerate(FAMILIES)}
class ContextualTransitionLearner(TransitionRuleLearner):
 ARMS=("L_contextual_categorical","R_contextual_relation","S_contextual_composition")
 def __init__(self,arm,*,intervention="none",reserve=0):
  if arm not in self.ARMS or intervention not in ("none","context_shuffle","relation_shuffle","composition_reset","context_relation_decouple"):raise ValueError("invalid contextual arm")
  self.arm=arm;self.intervention=intervention;self.weights={};self.ids={};self.pending=None;self.reserve=bytearray(reserve)
 def predict(self,episode):
  if self.pending is not None:raise ValueError("pending")
  values=episode.symbols;ctx=CONTEXT_BY_FAMILY[episode.family];rng=random.Random(sum(ord(c) for c in episode.identity)+920701)
  if self.intervention=="context_shuffle":ctx=rng.randrange(len(FAMILIES))
  relations=[self._relation(values[i-1],v) for i,v in enumerate(values) if i]
  if self.intervention=="relation_shuffle":relations=[(rng.randrange(-1,2),rng.randrange(3)) for _ in relations]
  if self.arm=="L_contextual_categorical":keys=[("ctxpair",ctx,values[i-1],v) for i,v in enumerate(values) if i]+[("marker",ctx,v) for v in values if v in (98,99)]
  else:
   keys=[]
   for relation in relations:
    keys.extend((("ctx",ctx),("rel",*relation))) if self.intervention=="context_relation_decouple" else keys.append(("ctxrel",ctx,*relation))
   keys.extend(("marker",ctx,v) for v in values if v in (98,99))
   if self.arm=="S_contextual_composition" and self.intervention!="composition_reset" and len(relations)>=2:keys.append(("ctxcompose",ctx,relations[-2],relations[-1]))
  features=tuple(dict.fromkeys(self._id(k) for k in keys));score=sum(self.weights.get(f,0.) for f in features);r=TransitionReceipt(id(self),int(score>0),features,len(values)+len(keys));self.pending=r;return r

def run_contextual_arm(arm,training,development,*,intervention="none",outcome_shuffle_seed=None,reserve=0,development_updates=True):
 learner=ContextualTransitionLearner(arm,intervention=intervention,reserve=reserve);train_y=[e.label for e in training];dev_y=[e.label for e in development]
 if outcome_shuffle_seed is not None:random.Random(outcome_shuffle_seed).shuffle(train_y);random.Random(outcome_shuffle_seed+1).shuffle(dev_y)
 updates=0;work=0
 for e,y in zip(training,train_y):r=learner.predict(e);work=max(work,r.work);updates+=learner.observe(r,y)
 totals={f:0 for f in FAMILIES};correct={f:0 for f in FAMILIES};rows=[]
 for e,y in zip(development,dev_y):r=learner.predict(e);work=max(work,r.work);totals[e.family]+=1;correct[e.family]+=int(r.predicted==e.label);rows.append((e.identity,r.predicted));updates+=learner.observe(r,y if development_updates else r.predicted)
 by={f:correct[f]/totals[f] for f in FAMILIES};return {"accuracy":sum(correct.values())/len(development),"relational_accuracy":sum(correct[f] for f in RELATIONAL_FAMILIES)/sum(totals[f] for f in RELATIONAL_FAMILIES),"accuracy_by_family":by,"trace_sha256":hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest(),"rows":rows,"updates":updates,"maximum_event_work":work,"state_bytes":learner.state_bytes(),"feature_count":len(learner.ids)}

__all__.extend(["CONTEXT_BY_FAMILY","ContextualTransitionLearner","run_contextual_arm"])

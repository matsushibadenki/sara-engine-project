"""Training-only frozen support-gated temporal readout diagnostic."""
from dataclasses import replace
from heapq import heappush, heappop
from hashlib import sha256
import json
from sara_engine.evaluation.casas_longitudinal import addresses, metrics
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts

ARMS = ('current','transition','joint','confidence','constant_gap',
        'confidence_constant','scalar_confidence','shuffled_confidence')


class ConfidenceCounts(SparseCapacityCounts):
    """Use the exact current reference until the detail has eight outcomes."""
    def __init__(self, arm, capacity=100000):
        super().__init__(capacity)
        self.arm = arm
        self.overrides = 0
        self.fallbacks = 0

    def predict_example(self, example):
        kind = 'constant_gap' if self.arm in ('constant_gap','confidence_constant') else self.arm
        receipt = super().predict(addresses(example,kind),example.cutoff,example.due)
        if self.arm in ('confidence','confidence_constant','scalar_confidence','shuffled_confidence'):
            detail = self.cells.get(receipt.addresses[-1],(0,0))
            if sum(detail) < 8:
                # Duplicate current posterior step matches the sufficient current arm.
                p = .5
                for key in (receipt.addresses[0],receipt.addresses[1],receipt.addresses[1]):
                    neg,pos = self.cells.get(key,(0,0))
                    p = (pos+8*p)/(neg+pos+8)
                receipt = replace(receipt,probability=p)
                self.pending[receipt.identity] = receipt
                self.fallbacks += 1
            else:
                self.overrides += 1
        return receipt


def phase(model,rows,training):
    queue=[];predictions=[]
    labels=[e.label for e in rows]
    if training and model.arm=='shuffled_confidence':
        offset=len(labels)//2;labels=labels[offset:]+labels[:offset]
    def flush(at):
        while queue and queue[0][0]<=at:
            due,_,receipt,label=heappop(queue)
            model.deliver(receipt,due,lambda:label,training=training)
    for i,e in enumerate(rows):
        flush(e.cutoff)
        receipt=model.predict_example(e);predictions.append(receipt.probability)
        heappush(queue,(e.due,i,receipt,labels[i]))
    if queue:flush(max(v[0] for v in queue))
    return predictions


def compare(examples):
    reports={};states={};traces={}
    for arm in ARMS:
        model=ConfidenceCounts(arm)
        phase(model,examples['train'],True)
        state=model.checkpoint()
        restored=ConfidenceCounts(arm)
        restored.cells=json.loads(json.dumps(state['cells']))
        restored.changed_writes=state['changed_writes']
        model.overrides=model.fallbacks=0
        prediction=phase(model,examples['development'],False)
        replay=phase(restored,examples['development'],False)
        if prediction!=replay or model.checkpoint()!=state:
            raise ValueError('Frozen checkpoint/prediction replay differs')
        if model.changed_writes!=3*len(examples['train']):
            raise ValueError('Actual writes not matched')
        reports[arm]={**metrics(examples['development'],prediction),
                      'changed_writes':model.changed_writes,'confirmation_writes':0,
                      'routes':len(model.cells),'peak_pending':model.peak_pending,
                      'overrides':model.overrides,'fallbacks':model.fallbacks}
        states[arm]={'arm':arm,'minimum_detail_support':8,'counts':state}
        traces[arm]=sha256(json.dumps(prediction).encode()).hexdigest()
    best=min(reports[a]['daily_brier'] for a in ('current','transition','joint'))
    ba=max(reports[a]['balanced_accuracy'] for a in ('current','transition','joint'))
    candidate=reports['confidence']
    gates={'brier_gain':best-candidate['daily_brier']>=.01,
           'balanced_gain':candidate['balanced_accuracy']-ba>=.02,
           'gap_necessity':reports['confidence_constant']['daily_brier']-candidate['daily_brier']>=.005,
           'paired_outcome':reports['shuffled_confidence']['daily_brier']-candidate['daily_brier']>=.01,
           'scalar_equivalence':traces['confidence']==traces['scalar_confidence']}
    return {'arms':reports,'traces':traces,'gates':gates,'passed':all(gates.values()),
            'original_development_opened':False,'test_opened':False,
            'scope':'Already consumed training-only component diagnostic; no fresh generalization evidence'},states

"""Sparse multimodal binding primitives."""

from .synesthetic_binding import (
    AudioEventAdapter,
    BindingAuditRecord,
    LanguageEventAdapter,
    SparseEventBundle,
    SparseMultimodalEvent,
    SparseModalityAdapter,
    SparsePluggableCorticalColumn,
    SparseSynestheticLinker,
    SparseTemporalBinder,
    SparseThalamicGate,
    TactileEventAdapter,
    ThalamicGateResult,
    VisionEventAdapter,
)
from .relation_hypothesis import (
    BoundedCrossModalHypothesisLedger,
    CrossModalHypothesisObservation,
    CrossModalHypothesisUpdate,
    CrossModalRelationHypothesis,
)

__all__ = [
    "AudioEventAdapter",
    "BindingAuditRecord",
    "LanguageEventAdapter",
    "SparseEventBundle",
    "SparseMultimodalEvent",
    "SparseModalityAdapter",
    "SparsePluggableCorticalColumn",
    "SparseSynestheticLinker",
    "SparseTemporalBinder",
    "SparseThalamicGate",
    "TactileEventAdapter",
    "ThalamicGateResult",
    "VisionEventAdapter",
    "BoundedCrossModalHypothesisLedger",
    "CrossModalHypothesisObservation",
    "CrossModalHypothesisUpdate",
    "CrossModalRelationHypothesis",
]

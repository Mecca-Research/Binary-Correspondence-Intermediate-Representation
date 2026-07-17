"""Opt-in hosted Llama training APIs with lazy framework imports."""
from __future__ import annotations

from .spec import (CorpusManifest, HostedRunReport, HostedTokenSource, HostedTrainEvent,
                   HostedTrainSpec, SequenceTokenSource)

_LAZY = {
    "HostedLlama": (".model", "HostedLlama"),
    "train_hosted": (".train", "train_hosted"),
    "export_hf_checkpoint": (".export", "export_hf_checkpoint"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module
    module, attribute = _LAZY[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = ["CorpusManifest", "HostedLlama", "HostedRunReport", "HostedTokenSource",
           "HostedTrainEvent", "HostedTrainSpec", "SequenceTokenSource",
           "export_hf_checkpoint", "train_hosted"]

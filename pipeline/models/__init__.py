from __future__ import annotations
from typing import TYPE_CHECKING
from .parsing_rules import ParsingRules
from .entities_linker import EntitiesLinker
__all__ = ["ParsingRules", "NeuralModel", "EntitiesLinker", "ValidationModel"]
if TYPE_CHECKING:
    from .neural_model import NeuralModel as NeuralModel
    from .validation_model import ValidationModel as ValidationModel
def __getattr__(name: str):
    if name == "NeuralModel":
        from .neural_model import NeuralModel as _NeuralModel
        return _NeuralModel
    if name == "ValidationModel":
        from .validation_model import ValidationModel as _ValidationModel
        return _ValidationModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

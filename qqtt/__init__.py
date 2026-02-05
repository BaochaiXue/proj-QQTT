"""
Top-level exports for qqtt.

Use lazy imports so camera-only utilities (e.g. calibration/recording) do not
pull heavyweight training dependencies such as torch at import time.
"""

from importlib import import_module
from typing import Any

__all__ = ["SpringMassSystemWarp", "InvPhyTrainerWarp", "OptimizerCMA"]


def __getattr__(name: str) -> Any:
    if name == "SpringMassSystemWarp":
        return import_module(".model", __name__).SpringMassSystemWarp
    if name in ("InvPhyTrainerWarp", "OptimizerCMA"):
        engine = import_module(".engine", __name__)
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

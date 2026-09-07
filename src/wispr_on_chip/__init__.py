"""wispr-on-chip: plan Whisper-class models onto a wafer-scale fabric with hardwired weights.

The package is a first-order engineering model, not a circuit simulator. It answers:

* how much silicon a model needs when its weights are etched into via/metal
  programmed cells next to the multipliers that use them (Taalas-style),
* how many copies of the model fit on a stitched 300 mm wafer (Cerebras-style),
* what latency, throughput and power the resulting fabric delivers, and
* how much redundancy the wafer needs to yield.

Every physical constant lives in :mod:`wispr_on_chip.process` and
:mod:`wispr_on_chip.fabric` and is meant to be overridden.
"""

from .fabric import DieTarget, TileBudget, TileConfig, WaferTarget, tile_budget
from .mapper import Plan, auto_mux, plan
from .models import MODELS, ModelSpec, get_model
from .process import PROCESSES, Process, get_process

__all__ = [
    "DieTarget",
    "MODELS",
    "ModelSpec",
    "PROCESSES",
    "Plan",
    "Process",
    "TileBudget",
    "TileConfig",
    "WaferTarget",
    "auto_mux",
    "get_model",
    "get_process",
    "plan",
    "tile_budget",
]

__version__ = "0.1.0"

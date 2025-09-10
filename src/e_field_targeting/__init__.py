"""
E-Field Targeting Package
=========================

A Python package for optimizing transcranial magnetic stimulation (TMS) coil currents
to achieve a desired electric field target in the brain.
"""

from .optimizer import EFieldModel, Mesh, OptimizerSettings, Stimulator
from .tms_optimizer import OptimizationResult, TMSOptimizer
from . import preprocessing

__version__ = "1.2.0"

__all__ = [
    "TMSOptimizer",
    "OptimizationResult",
    "OptimizerSettings",
    "Stimulator",
    "EFieldModel",
    "Mesh",
    "preprocessing",
]


"""
Approximate Bayesian Computation (ABC) for UQ PhysiCell.

This module provides Approximate Bayesian Computation (ABC) for PhysiCell model calibration
with enhanced strategies for model selection using pyABC.
"""

from .abc_context import (
    CalibrationContext,
    ModelSpec,
    run_abc_calibration,
)
from .utils import patch_pyabc_dataframe_csv_fallback

# Applied on import so every user of CalibrationContext/run_abc_calibration gets
# it for free -- see patch_pyabc_dataframe_csv_fallback's docstring.
patch_pyabc_dataframe_csv_fallback()

__all__ = [
    'CalibrationContext',
    'ModelSpec',
    'run_abc_calibration',
]
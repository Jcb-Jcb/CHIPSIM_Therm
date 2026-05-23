#!/usr/bin/env python3
# thermal_adapter.py
"""
Thermal adapter for converting CHIPSIM post-processing state into thermal-model inputs.

This module owns the boundary between CHIPSIM-native power data and the thermal
model input contract. Stage 2 follow-up work fills in power extraction, CSV/YAML
input generation, and adapter metadata production here.
"""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


class ThermalAdapterError(RuntimeError):
    """Raised when CHIPSIM data cannot be adapted for thermal modeling."""


@dataclass(frozen=True)
class ThermalAdapterInputs:
    """Inputs required to prepare thermal-model files from a CHIPSIM run."""

    formatted_results_dir: str
    time_step_us: float
    chiplet_power_over_time: Mapping[int, Sequence[float]]
    thermal_config: Mapping[str, Any]


@dataclass(frozen=True)
class ThermalAdapterResult:
    """Paths and metadata produced by the thermal adapter."""

    thermal_output_dir: str
    power_sequence_file: str
    resolved_power_config_file: Optional[str]
    max_power_w_by_chiplet: Mapping[int, float]
    zero_power_chiplets: Sequence[int]


class ThermalAdapter:
    """Prepares thermal-model inputs from CHIPSIM post-processing data."""

    def __init__(self, inputs: ThermalAdapterInputs):
        self.inputs = inputs

    @property
    def thermal_output_dir(self) -> str:
        """Return the per-run thermal output directory path."""
        return os.path.join(self.inputs.formatted_results_dir, 'thermal')

    def prepare_inputs(self) -> ThermalAdapterResult:
        """Prepare thermal-model inputs from CHIPSIM power traces."""
        raise NotImplementedError(
            "Thermal input generation will be implemented by the remaining Stage 2 actions."
        )

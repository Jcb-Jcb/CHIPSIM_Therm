#!/usr/bin/env python3
# thermal_adapter.py
"""
Thermal adapter for converting CHIPSIM post-processing state into thermal-model inputs.

This module owns the boundary between CHIPSIM-native power data and the thermal
model input contract. Stage 2 follow-up work fills in CSV/YAML input generation
and adapter metadata production here.
"""

import os
from collections.abc import Mapping as MappingABC
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


def extract_chiplet_power_over_time(metric_computer: Any) -> Mapping[int, Sequence[float]]:
    """Read chiplet total power traces from a MetricComputer after power metrics run."""
    power_traces = getattr(metric_computer, 'chiplet_total_power_over_time', None)

    if power_traces is None:
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time is not available. "
            "Call MetricComputer.compute_power_profile() before preparing thermal inputs."
        )

    if not isinstance(power_traces, MappingABC):
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time must be a mapping of "
            "chiplet_id to power trace."
        )

    if not power_traces:
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time is empty; no chiplet power "
            "traces are available for thermal input generation."
        )

    for chiplet_id, power_trace in power_traces.items():
        if power_trace is None:
            raise ThermalAdapterError(
                f"Power trace for chiplet {chiplet_id!r} is missing."
            )

    return dict(power_traces)


class ThermalAdapter:
    """Prepares thermal-model inputs from CHIPSIM post-processing data."""

    def __init__(self, inputs: ThermalAdapterInputs):
        self.inputs = inputs

    @classmethod
    def from_metric_computer(
        cls,
        formatted_results_dir: str,
        time_step_us: float,
        metric_computer: Any,
        thermal_config: Mapping[str, Any]
    ):
        """Build an adapter by reading power traces from a computed MetricComputer."""
        return cls(
            ThermalAdapterInputs(
                formatted_results_dir=formatted_results_dir,
                time_step_us=time_step_us,
                chiplet_power_over_time=extract_chiplet_power_over_time(metric_computer),
                thermal_config=thermal_config,
            )
        )

    @property
    def thermal_output_dir(self) -> str:
        """Return the per-run thermal output directory path."""
        return os.path.join(self.inputs.formatted_results_dir, 'thermal')

    def prepare_inputs(self) -> ThermalAdapterResult:
        """Prepare thermal-model inputs from CHIPSIM power traces."""
        raise NotImplementedError(
            "Thermal input generation will be implemented by the remaining Stage 2 actions."
        )

#!/usr/bin/env python3

import os
import yaml
from typing import Dict, Any

THERMAL_PATH_KEYS = (
    'geometry_file',
    'material_prop_file',
    'power_config_file',
)
THERMAL_REQUIRED_KEYS = THERMAL_PATH_KEYS + ('power_layer',)


def get_project_root() -> str:
    """Return the CHIPSIM repository root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_project_path(path_value: str, project_root: str) -> str:
    """Resolve a path relative to the CHIPSIM repository root."""
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(project_root, path_value))


def validate_and_resolve_thermal_config(
    config: Dict[str, Any],
    config_file: str,
    project_root: str
) -> None:
    """
    Validate enabled thermal config and resolve path fields in-place.

    Args:
        config: Configuration dictionary to validate
        config_file: Path to config file (for error messages)
        project_root: CHIPSIM repository root

    Raises:
        FileNotFoundError: If a required thermal path does not exist
        ValueError: If required thermal keys are missing or malformed
    """
    thermal_config = config.get('post_processing', {}).get('thermal')
    if thermal_config is None:
        return

    if not isinstance(thermal_config, dict):
        raise ValueError(
            f"Invalid thermal configuration in {config_file}.\n"
            f"Expected 'post_processing.thermal' to be a mapping."
        )

    if not thermal_config.get('enabled', False):
        return

    missing_keys = [
        key for key in THERMAL_REQUIRED_KEYS
        if key not in thermal_config or thermal_config[key] in (None, "")
    ]
    if missing_keys:
        raise ValueError(
            f"Invalid thermal configuration in {config_file}.\n"
            f"'post_processing.thermal.enabled' is true, but missing required key(s): "
            f"{', '.join(missing_keys)}."
        )

    if not isinstance(thermal_config['power_layer'], str):
        raise ValueError(
            f"Invalid thermal configuration in {config_file}.\n"
            f"'post_processing.thermal.power_layer' must be a string."
        )

    for key in THERMAL_PATH_KEYS:
        path_value = thermal_config[key]
        if not isinstance(path_value, str):
            raise ValueError(
                f"Invalid thermal configuration in {config_file}.\n"
                f"'post_processing.thermal.{key}' must be a path string."
            )

        resolved_path = resolve_project_path(path_value, project_root)
        thermal_config[key] = resolved_path

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Thermal configuration path not found in {config_file}.\n"
                f"Key: post_processing.thermal.{key}\n"
                f"Path: {resolved_path}"
            )


def validate_config_structure(config: Dict[str, Any], config_file: str) -> None:
    """
    Validate that the configuration has the expected nested structure.
    
    Args:
        config: Configuration dictionary to validate
        config_file: Path to config file (for error messages)
        
    Raises:
        ValueError: If the configuration structure is invalid
    """
    # Check for required top-level keys
    if 'simulation' not in config:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required 'simulation' key.\n"
            f"Expected structure:\n"
            f"  simulation:\n"
            f"    input_files: ...\n"
            f"    core_settings: ...\n"
            f"    ...\n"
            f"  post_processing: ..."
        )
    
    if 'post_processing' not in config:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required 'post_processing' key.\n"
            f"Expected structure:\n"
            f"  simulation: ...\n"
            f"  post_processing:\n"
            f"    warmup_period_us: ...\n"
            f"    generate_plots: ..."
        )
    
    # Check for required simulation subsections
    required_subsections = ['input_files', 'core_settings', 'hardware_parameters']
    missing_subsections = [s for s in required_subsections if s not in config['simulation']]
    
    if missing_subsections:
        raise ValueError(
            f"Invalid configuration structure in {config_file}.\n"
            f"Missing required simulation subsection(s): {', '.join(missing_subsections)}\n"
            f"Expected subsections under 'simulation':\n"
            f"  - input_files\n"
            f"  - core_settings\n"
            f"  - hardware_parameters\n"
            f"  - gem5_parameters (optional)\n"
            f"  - dsent_parameters (optional)"
        )


def load_config(config_file: str = "configs/experiments/config_1.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_file: Path to config file, defaults to "configs/experiments/config_1.yaml" 
                    Can be:
                    - Relative path from project root (e.g., "configs/experiments/config_2.yaml")
                    - Just experiment name (e.g., "config_2" - will look in configs/experiments/)
                    - Absolute path
        
    Returns:
        Dictionary containing configuration values with nested structure
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
    """
    # Handle experiment name shortcut (e.g., "config_2" -> "configs/experiments/config_2.yaml")
    if not config_file.endswith('.yaml') and not os.path.isabs(config_file):
        config_file = f"configs/experiments/{config_file}.yaml"
    
    project_root = get_project_root()

    # If relative path, make it relative to the project root
    if not os.path.isabs(config_file):
        config_file = os.path.join(project_root, config_file)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate the configuration structure
    validate_config_structure(config, config_file)
    validate_and_resolve_thermal_config(config, config_file, project_root)
    
    return config

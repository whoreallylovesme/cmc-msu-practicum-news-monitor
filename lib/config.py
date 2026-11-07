"""Load service configuration from YAML and apply explicit environment overrides."""

import os
from pathlib import Path
from typing import Optional

import yaml

from .contracts import ProjectConfig


def load_config(
    path: Path,
    *,
    env_var: Optional[str] = None,
    overrides: Optional[dict[str, str]] = None,
) -> dict:
    selected = Path(os.environ.get(env_var, str(path))) if env_var else Path(path)
    with selected.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {selected}")
    for key, variable in (overrides or {}).items():
        if variable in os.environ:
            config[key] = os.environ[variable]
    return config


def load_project_config() -> dict:
    path = Path(
        os.environ.get("MONITORING_CONFIG", Path(__file__).resolve().parents[1] / "config.yaml")
    )
    values = load_config(path)
    if os.environ.get("MONITOR_DATASET_PATH"):
        values["path_to_offline_demo_data"] = os.environ["MONITOR_DATASET_PATH"]
    config = ProjectConfig.model_validate(values).model_dump()
    snapshot = Path(config["path_to_offline_demo_data"])
    if not snapshot.is_absolute():
        snapshot = path.resolve().parent / snapshot
    config["path_to_offline_demo_data"] = str(snapshot)
    return config


def load_monitoring() -> dict:
    project = load_project_config()
    monitoring = project["monitoring"]
    if project["offline_demo"]:
        monitoring["source"] = "hn_snapshot"
    return monitoring

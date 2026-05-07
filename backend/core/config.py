import yaml
from pathlib import Path
from typing import Optional

from ..schemas.config import RunConfig

# Global variable to hold the loaded configuration
_config: Optional[RunConfig] = None


def load_config(config_path: str = "config.yaml") -> RunConfig:
    """Load configuration from a YAML file and validate it with Pydantic.

    Args:
        config_path: Relative or absolute path to the YAML configuration file.

    Returns:
        An instance of :class:`RunConfig` containing the validated configuration.
    """
    global _config
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    _config = RunConfig(**raw_data)
    return _config


def get_config() -> Optional[RunConfig]:
    """Retrieve the currently loaded configuration.

    Returns:
        The :class:`RunConfig` instance if ``load_config`` has been called,
        otherwise ``None``.
    """
    return _config

"""Загрузчик конфигурации экспериментов из params.yaml.

Все гиперпараметры, пути и random seeds живут в params.yaml на корне проекта,
чтобы DVC мог их трекать и пересобирать пайплайн при их изменении.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params(path: Path | str | None = None) -> dict[str, Any]:
    """Прочитать params.yaml. Кешируется внутри одного процесса."""
    target = Path(path) if path else PARAMS_PATH
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{target} must contain a top-level mapping, got {type(data).__name__}")
    return data


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """Удобный доступ: get('train', 'random_seed') или get('train')."""
    params = load_params()
    block = params.get(section, {})
    if key is None:
        return block
    return block.get(key, default)

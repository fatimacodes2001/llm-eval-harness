from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    golden_path: Path
    split: str = "test"
    provider: dict[str, Any]
    few_shot_k: int = Field(default=0, ge=0)
    structured_output: bool = False
    max_tokens: int = 2048
    concurrency: int = Field(default=4, ge=1)
    use_cache: bool = True
    cache_dir: Path = Path(".cache")
    reports_dir: Path = Path("reports")
    limit: int | None = None
    bootstrap_samples: int = 1000
    seed: int = 0
    judge: dict[str, Any] | None = None

    def hash(self) -> str:
        blob = self.model_dump_json(exclude={"limit", "use_cache", "concurrency"})
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: Path) -> RunConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)

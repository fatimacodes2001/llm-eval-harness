from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from eval_harness.schema import GoldenExample


def load_golden(path: Path) -> list[GoldenExample]:
    rows: list[GoldenExample] = []
    errors: list[str] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(GoldenExample.model_validate_json(line))
            except ValidationError as e:
                errors.append(f"line {lineno}: {e.errors()[0]['msg']}")
    if errors:
        raise ValueError("golden set invalid:\n" + "\n".join(errors))
    ids = [r.id for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate example ids: {sorted(dupes)}")
    return rows


def write_golden(path: Path, rows: list[GoldenExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump(mode="json", by_alias=True), sort_keys=True) + "\n")

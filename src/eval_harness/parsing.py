"""Turn raw model text into a validated ``HazardReport`` or a parse error."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from eval_harness.schema import HazardReport

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_text(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def parse_report(text: str) -> tuple[HazardReport | None, str | None]:
    if not text.strip():
        return None, "empty response"
    try:
        data = json.loads(extract_json_text(text))
    except json.JSONDecodeError as e:
        return None, f"json: {e.msg} at pos {e.pos}"
    try:
        return HazardReport.model_validate(data), None
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        return None, f"schema: {loc}: {first['msg']}"

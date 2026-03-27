import re
from typing import Any
from partial_json_parser import loads as partial_json_loads  # type: ignore[import-untyped]


def parse_llm_json_stream(input_str: str) -> dict[str, Any] | None:
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", input_str.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    brace_idx = cleaned.find("{")
    if brace_idx == -1:
        return {"answer": input_str} if len(input_str) > 10 else {}

    try:
        json_object = partial_json_loads(cleaned[brace_idx:])
        if not isinstance(json_object, dict):
            return None
        return json_object  # type: ignore[return-value]
    except Exception:
        return None

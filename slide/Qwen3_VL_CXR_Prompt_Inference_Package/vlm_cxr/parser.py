from __future__ import annotations

import json
from dataclasses import asdict, dataclass


PREDICTION_LABELS = ("NORMAL", "PNEUMONIA", "UNCERTAIN")


@dataclass(frozen=True)
class ParseResult:
    predicted_label: str
    parse_ok: bool
    parse_error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_model_output(raw_text: str) -> ParseResult:
    """Parse the frozen one-key schema; every failure becomes UNCERTAIN."""
    try:
        value = json.loads(raw_text.strip())
    except (json.JSONDecodeError, TypeError) as error:
        return ParseResult("UNCERTAIN", False, f"invalid_json:{error.msg if hasattr(error, 'msg') else error}")
    if not isinstance(value, dict):
        return ParseResult("UNCERTAIN", False, "not_an_object")
    if set(value) != {"label"}:
        return ParseResult("UNCERTAIN", False, "schema_keys_must_equal_label")
    label = value["label"]
    if not isinstance(label, str) or label not in PREDICTION_LABELS:
        return ParseResult("UNCERTAIN", False, "invalid_label")
    return ParseResult(label, True, None)

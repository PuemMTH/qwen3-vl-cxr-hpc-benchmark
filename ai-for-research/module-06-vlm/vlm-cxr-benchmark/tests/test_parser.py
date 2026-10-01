from vlm_cxr.parser import parse_model_output


def test_strict_valid_schema() -> None:
    result = parse_model_output('{"label":"NORMAL"}')
    assert result.parse_ok is True
    assert result.predicted_label == "NORMAL"


def test_every_schema_failure_becomes_uncertain() -> None:
    invalid = [
        "NORMAL",
        "```json\n{\"label\":\"NORMAL\"}\n```",
        '{"label":"normal"}',
        '{"label":"NORMAL","explanation":"x"}',
        '{"decision":"NORMAL"}',
        "[]",
        "",
    ]
    for raw in invalid:
        result = parse_model_output(raw)
        assert result.parse_ok is False
        assert result.predicted_label == "UNCERTAIN"

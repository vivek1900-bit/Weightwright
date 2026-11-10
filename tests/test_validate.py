"""Tests for weightwright.validate and `wright validate` CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tests.conftest import E, Q, W, ToyBackend
from weightwright.cli import main
from weightwright.store import SafetensorsStore
from weightwright.validate import (
     format_report,
    report_to_dict,
    validate_spec,
    validate_spec_file,
)


def _codes(report) -> set[str]:
    return {f.code for f in report.findings}


def test_good_spec_ok(make_spec):
    spec = make_spec()
    report = validate_spec(spec, check_backend=False, check_env=False)
    assert report.ok
    assert report.errors == []
    codes = _codes(report)
    assert "MODEL_SUMMARY" in codes
    assert "MODEL_SIZE" in codes
    assert "EVAL_SUMMARY" in codes
    assert "NOT_CHECKED" in codes


def test_validate_spec_file_missing(tmp_path):
    report = validate_spec_file(tmp_path / "does_not_exist.yaml")
    assert not report.ok
    assert _codes(report) == {"SPEC_LOAD"}


def test_model_dir_missing_and_quantized_and_unreadable(tmp_path, make_spec):
    spec = make_spec()
    spec.model_path = str(tmp_path / "no_such_model")
    report = validate_spec(spec, check_backend=False, check_env=False)
    assert "MODEL_DIR_MISSING" in _codes(report)

    q_dir = tmp_path / "quantized_model"
    q_dir.mkdir()
    (q_dir / "model-q4.gguf").write_bytes(b"GGUF")
    spec.model_path = str(q_dir)
    report_q = validate_spec(spec, check_backend=False, check_env=False)
    assert "MODEL_QUANTIZED" in _codes(report_q)

    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()
    spec.model_path = str(empty_dir)
    report_empty = validate_spec(spec, check_backend=False, check_env=False)
    assert "MODEL_NO_SAFETENSORS" in _codes(report_empty)

    bad_dir = tmp_path / "bad_safetensors"
    bad_dir.mkdir()
    (bad_dir / "corrupt.safetensors").write_bytes(b"not a valid safetensors file")
    spec.model_path = str(bad_dir)
    report_bad = validate_spec(spec, check_backend=False, check_env=False)
    assert "MODEL_UNREADABLE" in _codes(report_bad)


def test_eval_errors_and_warnings(tmp_path, make_spec):
    spec = make_spec()
    spec.eval_file = str(tmp_path / "missing_eval.jsonl")
    report = validate_spec(spec, check_backend=False, check_env=False)
    assert "EVAL_FILE_MISSING" in _codes(report)

    bad_eval = tmp_path / "bad_eval.jsonl"
    bad_eval.write_text("{not valid json\n")
    spec.eval_file = str(bad_eval)
    report_bad = validate_spec(spec, check_backend=False, check_env=False)
    assert "EVAL_BAD" in _codes(report_bad)

    unknown_eval = tmp_path / "unknown_check.jsonl"
    unknown_eval.write_text(
        json.dumps(
            {
                "prompt": "p1",
                "tag": "target",
                "check": {"all": [{"type": "contains", "value": "ok"}, {"type": "magic_check"}]},
            }
        )
        + "\n"
        + json.dumps(
            {
                "prompt": "p1",
                "tag": "regression",
                "check": {"type": "regex", "pattern": "[unclosed"},
            }
        )
        + "\n"
    )
    spec.eval_file = str(unknown_eval)
    report_chk = validate_spec(spec, check_backend=False, check_env=False)
    codes = _codes(report_chk)
    assert "EVAL_UNKNOWN_CHECK" in codes
    assert "EVAL_DUPLICATE_PROMPTS" in codes
    unknown_msgs = [f.message for f in report_chk.errors if f.code == "EVAL_UNKNOWN_CHECK"]
    assert len(unknown_msgs) == 2

    no_target = tmp_path / "no_target.jsonl"
    no_target.write_text(
        json.dumps({"prompt": "r1", "tag": "regression", "check": {"type": "finished"}}) + "\n"
    )
    spec.eval_file = str(no_target)
    report_nt = validate_spec(spec, check_backend=False, check_env=False)
    assert "EVAL_NO_TARGET" in _codes(report_nt)


def test_no_regression_warning_and_strict_cli(tmp_path, make_spec):
    spec = make_spec()
    eval_no_reg = tmp_path / "only_target.jsonl"
    eval_no_reg.write_text(
        json.dumps({"prompt": "t1", "tag": "target", "check": {"type": "finished"}}) + "\n"
        + json.dumps({"prompt": "t2", "tag": "target", "check": {"type": "no_loop"}}) + "\n"
    )
    spec.eval_file = str(eval_no_reg)
    report = validate_spec(spec, check_backend=False, check_env=False)
    assert report.ok
    assert "EVAL_NO_REGRESSION" in _codes(report)
    assert "EVAL_FEW_TARGET" in _codes(report)

    spec_file = tmp_path / "spec.yaml"
    text = spec_file.read_text()
    spec_file.write_text(text.replace(Path(make_spec().eval_file).name, eval_no_reg.name))

    rc_normal = main(
        ["validate", str(spec_file), "--no-backend-checks", "--no-env-checks"]
    )
    assert rc_normal == 0

    rc_strict = main(
        ["validate", str(spec_file), "--strict", "--no-backend-checks", "--no-env-checks"]
    )
    assert rc_strict == 1


def test_budget_and_gate_cases(make_spec):
    spec = make_spec()
    spec.max_touched_fraction = 0.0
    assert "BUDGET_FRACTION" in _codes(validate_spec(spec, check_backend=False, check_env=False))

    spec.max_touched_fraction = 2.0
    assert "BUDGET_FRACTION" in _codes(validate_spec(spec, check_backend=False, check_env=False))

    spec.max_touched_fraction = 1e-9
    assert "BUDGET_ZERO" in _codes(validate_spec(spec, check_backend=False, check_env=False))

    spec.max_touched_fraction = 0.5
    spec.allow = ["*nonexistent_layer*"]
    report_allow = validate_spec(spec, check_backend=False, check_env=False)
    assert "BUDGET_ALLOW_NO_MATCH" in _codes(report_allow)
    assert "BUDGET_LARGE" in _codes(report_allow)

    spec2 = make_spec()
    spec2.min_gain = 0.0
    assert "GATE_VALUES" in _codes(validate_spec(spec2, check_backend=False, check_env=False))


def test_scripted_ops_cases(make_spec):
    spec = make_spec()
    spec.scripted_ops = [
        "not_a_dict",
        {"op": "scale", "tensor": "model.unknown.weight", "factor": 0.5},
        {"op": "explode", "tensor": W},
        {"op": "scale", "tensor": W, "index": "[0:1, 0:1, 0:1]", "factor": 0.5},
        {"op": "scale", "tensor": E, "factor": 0.5},
        {
            "op": "project_out",
            "tensor": W,
            "direction": {"source": "inline", "values": [1.0, 2.0, 3.0]},
        },
        {"op": "scale", "tensor": W},
    ]
    report = validate_spec(spec, check_backend=False, check_env=False)
    codes = _codes(report)
    assert "OPS_NOT_DICT" in codes
    assert "OPS_UNKNOWN_TENSOR" in codes
    assert "OPS_UNKNOWN_KIND" in codes
    assert "OPS_BAD_INDEX" in codes
    assert "OPS_DENIED" in codes
    assert "OPS_DIRECTION_DIM" in codes
    assert "OPS_MISSING_FIELD" in codes

    spec_big = make_spec()
    spec_big.max_touched_fraction = 0.01
    spec_big.scripted_ops = [{"op": "scale", "tensor": Q, "factor": 0.9}]
    report_big = validate_spec(spec_big, check_backend=False, check_env=False)
    assert "OPS_TOO_BIG" in _codes(report_big)


def test_backend_checks(model_dir, make_spec, monkeypatch):
    spec = make_spec()

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: True if name == "transformers" else None,
    )
    report = validate_spec(spec, check_backend=True, check_env=False)
    assert "BACKEND_NO_TOKENIZER" in _codes(report)

    (Path(model_dir) / "tokenizer.json").write_text("{}")
    report_fixed = validate_spec(spec, check_backend=True, check_env=False)
    assert "BACKEND_NO_TOKENIZER" not in _codes(report_fixed)
    assert report_fixed.ok

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "transformers" else True,
    )
    report_no_tf = validate_spec(spec, check_backend=True, check_env=False)
    assert "BACKEND_NO_TRANSFORMERS" in _codes(report_no_tf)

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)
    spec.device = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report_no_cuda = validate_spec(spec, check_backend=True, check_env=False)
    assert "BACKEND_CUDA_UNAVAILABLE" in _codes(report_no_cuda)

    (Path(model_dir) / "config.json").unlink()
    report_no_cfg = validate_spec(spec, check_backend=True, check_env=False)
    assert "BACKEND_NO_CONFIG" in _codes(report_no_cfg)

    spec.backend = "vllm"
    spec.model_dtype = "int8"
    spec.device = "tpu"
    report_bad_fields = validate_spec(spec, check_backend=False, check_env=False)
    assert {"BACKEND_UNKNOWN", "DTYPE_INVALID", "DEVICE_INVALID"}.issubset(
        _codes(report_bad_fields)
    )


def test_env_checks_never_leak_secret(make_spec, monkeypatch):
    spec = make_spec()
    spec.editor = {"provider": "anthropic", "model": "claude-3-5-sonnet", "api_key_env": "TEST_WRIGHT_KEY"}
    monkeypatch.delenv("TEST_WRIGHT_KEY", raising=False)

    report_missing = validate_spec(spec, check_backend=False, check_env=True)
    assert "EDITOR_KEY_MISSING" in _codes(report_missing)
    assert "SCRIPTED_OPS_IGNORED" in _codes(report_missing)

    monkeypatch.setenv("TEST_WRIGHT_KEY", "SECRET-VALUE")
    report_set = validate_spec(spec, check_backend=False, check_env=True)
    assert "EDITOR_KEY_MISSING" not in _codes(report_set)

    formatted = format_report(report_set)
    as_json = json.dumps(report_to_dict(report_set))
    assert "SECRET-VALUE" not in formatted
    assert "SECRET-VALUE" not in as_json


def test_editor_provider_and_model_errors(make_spec):
    spec = make_spec()
    spec.editor = {"provider": "mystery_llm"}
    assert "EDITOR_PROVIDER" in _codes(validate_spec(spec, check_backend=False, check_env=False))

    spec.editor = {"provider": "openai_compat"}
    assert "EDITOR_MODEL" in _codes(validate_spec(spec, check_backend=False, check_env=False))


def test_validate_never_loads_weights(make_spec, monkeypatch):
    def _forbid_get(self, name):
        raise AssertionError(f"SafetensorsStore.get({name!r}) must never be called by validate")

    monkeypatch.setattr(SafetensorsStore, "get", _forbid_get)
    spec = make_spec()
    report = validate_spec(spec, check_backend=False, check_env=False)
    assert report.ok


def test_cli_validate_json_and_run_gate(tmp_path, model_dir, make_spec, monkeypatch, capsys):
    make_spec()
    spec_path = tmp_path / "spec.yaml"
    (Path(model_dir) / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)

    rc = main(["validate", str(spec_path), "--json", "--no-env-checks"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["errors"] == 0
    assert isinstance(payload["findings"], list)

    (Path(model_dir) / "tokenizer.json").unlink()
    rc_err = main(["validate", str(spec_path), "--json", "--no-env-checks"])
    out_err = capsys.readouterr().out
    payload_err = json.loads(out_err)
    assert rc_err == 1
    assert payload_err["ok"] is False
    assert payload_err["errors"] >= 1

    import weightwright.runner as runner_mod

    if hasattr(runner_mod, "make_backend"):
        monkeypatch.setattr(
            runner_mod,
            "make_backend",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("make_backend must not be called")),
        )
    monkeypatch.setattr(
        runner_mod,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run must not be called when validate fails")),
    )
    rc_run_fail = main(["run", str(spec_path)])
    assert rc_run_fail == 1
    capsys.readouterr()

    called = {"ran": False}

    def _fake_run(s):
        called["ran"] = True

        class _DummyRes:
            status = "accepted"
            run_dir = tmp_path / "runs"
            patch_dir = None
            reasons = []

        return _DummyRes()

    monkeypatch.setattr(runner_mod, "run", _fake_run)
    rc_skip = main(["run", str(spec_path), "--skip-validate"])
    assert rc_skip == 0
    assert called["ran"] is True

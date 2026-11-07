import json

import pytest

from tests.conftest import E, Q, ToyBackend, W
from weightwright.cli import main
from weightwright.patch import load_patch
from weightwright.runner import extract_json, run
from weightwright.store import SafetensorsStore

FIX = {"op": "scale", "tensor": W, "factor": 0.3, "index": "[0, 0]"}


def test_extract_json_handles_prose_and_braces_in_strings():
    assert extract_json('Sure! {"action": "x", "t": "a } b"} done') == {"action": "x", "t": "a } b"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_scripted_run_accepts_good_edit(make_spec, model_dir):
    spec = make_spec([FIX])
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), say=lambda *_: None)
    assert res.status == "accepted"
    assert res.base["target"] == 0.0 and res.final["target"] == 1.0
    assert res.final["regression"] == res.base["regression"] == 1.0
    patch = load_patch(res.patch_dir)
    assert patch.touched == 1 and patch.meta["baseline_scores"]["target"] == 0.0
    assert (res.run_dir / "log.jsonl").exists() and (res.run_dir / "checkpoints" / "eval_01.json").exists()


def test_gate_rejects_regression_and_saves_rejected_patch(make_spec, model_dir):
    spec = make_spec([FIX, {"op": "zero", "tensor": Q, "index": "[0, 0]"}])  # fixes loop but breaks "capital"
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), say=lambda *_: None)
    assert res.status == "rejected" and res.patch_dir is None
    assert (res.run_dir / "rejected_patch" / "patch.json").exists()
    assert not (res.run_dir / "patch").exists()


def test_gate_rejects_insufficient_gain(make_spec, model_dir):
    spec = make_spec([{"op": "scale", "tensor": W, "factor": 0.9, "index": "[0, 0]"}])  # still > 0.5
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), say=lambda *_: None)
    assert res.status == "rejected"


class FakeLLM:
    """Plays a messy editor: junk reply, bad tool use, a denied tensor, then a correct fix."""

    def __init__(self):
        self.script = [
            "I think I should look around first.",  # not JSON
            json.dumps({"action": "list_tensors", "pattern": "*layers*"}),
            json.dumps({"action": "stats", "tensor": W, "svd_k": 2}),
            json.dumps({"action": "show_failures", "n": 1}),
            json.dumps({"action": "propose_edit", "op": {"op": "zero", "tensor": E}}),  # denied
            json.dumps({"action": "propose_edit", "op": {"op": "scale", "tensor": "nope", "factor": 2}}),  # unknown
            "Here is my edit:\n" + json.dumps({"action": "propose_edit", "op": FIX}),
            json.dumps({"action": "evaluate"}),
            json.dumps({"action": "submit"}),
        ]
        self.seen = []

    def chat(self, system, messages):
        self.seen.append(messages[-1]["content"])
        return self.script.pop(0)


def test_llm_style_editor_flow(make_spec, model_dir):
    spec = make_spec(provider="openai_compat", editor={"provider": "openai_compat", "model": "x", "max_turns": 15, "max_evals": 5})
    llm = FakeLLM()
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), editor=llm, say=lambda *_: None)
    assert res.status == "accepted"
    joined = "\n".join(llm.seen)
    assert "not a single valid JSON" in joined          # junk reply was reported back
    assert "deny list" in joined                         # budget refused the embedding edit
    assert "unknown tensor" in joined                    # bad tensor name reported, no crash
    assert "top_singular_values" in joined               # stats tool works
    assert "hit token limit" in joined                   # failures tool shows diagnosis info


def test_eval_budget_and_turn_limit(make_spec, model_dir):
    class Looper:
        def chat(self, system, messages):
            return json.dumps({"action": "list_tensors"})
    spec = make_spec(provider="openai_compat", editor={"provider": "openai_compat", "model": "x", "max_turns": 4, "max_evals": 1})
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), editor=Looper(), say=lambda *_: None)
    assert res.status == "rejected" and "turn limit" in res.reasons[0]


def test_cli_inspect_apply_revert(model_dir, make_spec, tmp_path, capsys):
    spec = make_spec([FIX])
    res = run(spec, backend=ToyBackend(SafetensorsStore(model_dir)), say=lambda *_: None)
    assert main(["inspect", str(model_dir)]) == 0
    assert "model.layers.0.mlp.w" in capsys.readouterr().out
    out_dir, back = tmp_path / "patched", tmp_path / "back"
    assert main(["apply", str(res.patch_dir), "--model", str(model_dir), "--out", str(out_dir)]) == 0
    assert main(["revert", str(res.patch_dir), "--model", str(out_dir), "--out", str(back)]) == 0
    assert main(["apply", str(res.patch_dir), "--model", str(out_dir), "--out", str(tmp_path / "x")]) == 1  # refuses
    assert main(["info", str(res.patch_dir)]) == 0
    assert main(["inspect", str(tmp_path / "missing")]) == 1


def test_cli_init(tmp_path):
    assert main(["init", str(tmp_path / "proj")]) == 0
    assert (tmp_path / "proj" / "specs" / "example.yaml").exists()


def test_relative_paths_resolve_from_working_directory(tmp_path, monkeypatch):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "s.yaml").write_text(
        "model: {path: ./models/m}\ngoal: g\neval: {file: ./evals/e.jsonl}\n"
        "editor: {provider: scripted}\nscripted_ops: [{op: zero, tensor: t}]\n")
    monkeypatch.chdir(tmp_path)
    from weightwright.spec import load_spec
    spec = load_spec("specs/s.yaml")
    assert spec.resolve(spec.model_path) == str(tmp_path / "models" / "m")
    assert spec.resolve(spec.eval_file) == str(tmp_path / "evals" / "e.jsonl")


def test_cli_eval_show_failures(make_spec, model_dir, capsys, monkeypatch):
    spec = make_spec([FIX])
    import yaml, weightwright.runner as runner
    monkeypatch.setattr(runner, "make_backend", lambda s: ToyBackend(SafetensorsStore(model_dir)))
    spec_path = spec.resolve("spec.yaml") if False else None
    import pathlib
    path = pathlib.Path(spec.eval_file).parent / "spec.yaml"
    assert main(["eval", str(path), "--show-failures"]) == 0
    out = capsys.readouterr().out
    assert '"target": 0.0' in out and "failing item(s)" in out and "hit token limit: True" in out

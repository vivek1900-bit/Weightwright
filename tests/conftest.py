import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from weightwright.backends import _Revertible
from weightwright.evalset import Generation
from weightwright.store import SafetensorsStore

W = "model.layers.0.mlp.w"
Q = "model.layers.1.self_attn.q"
E = "model.embed_tokens.weight"


@pytest.fixture
def model_dir(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {
        W: torch.ones(8, 8),  # w[0,0] > 0.5 makes the toy model "loop"
        Q: (torch.rand(16, 8, generator=g) + 0.25).to(torch.bfloat16),  # bf16 on purpose
        E: torch.randn(10, 8, generator=g),
    }
    d = tmp_path / "model"
    d.mkdir()
    save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"toy": True}))
    return d


class ToyBackend(_Revertible):
    """Behaviour is a pure function of two weights, so tests need no real LLM."""

    def __init__(self, store: SafetensorsStore):
        super().__init__()
        self.t = {n: store.get(n).clone() for n in store.names()}

    def _param(self, name):
        return self.t[name]

    def generate(self, prompts, max_new_tokens):
        looping = self.t[W][0, 0].item() > 0.5
        q_ok = self.t[Q][0, 0].item() != 0
        out = []
        for p in prompts:
            if "capital" in p:
                out.append(Generation("Paris" if q_ok else "???"))
            elif looping:
                out.append(Generation("abc abc abc abc abc abc abc abc abc abc " * 3, hit_limit=True))
            else:
                out.append(Generation("The answer is 4"))
        return out


EVAL_LINES = [
    {"prompt": "math one", "tag": "target", "check": {"all": [{"type": "finished"}, {"type": "no_loop"}, {"type": "last_number", "value": 4}]}},
    {"prompt": "math two", "tag": "target", "check": {"all": [{"type": "finished"}, {"type": "no_loop"}, {"type": "last_number", "value": 4}]}},
    {"prompt": "capital of France?", "tag": "regression", "check": {"type": "contains", "value": "paris"}},
    {"prompt": "capital of Italy?", "tag": "regression", "check": {"type": "contains", "any": ["paris", "rome"]}},
]


@pytest.fixture
def make_spec(tmp_path, model_dir):
    def _make(scripted_ops=None, provider="scripted", **over):
        (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in EVAL_LINES))
        raw = {
            "name": "t",
            "model": {"path": str(model_dir)},
            "goal": "stop looping",
            "eval": {"file": str(tmp_path / "eval.jsonl"), "min_gain": 0.5},
            "editor": {"provider": provider, "max_turns": 12, "max_evals": 5},
            "budget": {"max_touched_fraction": 0.5, "deny": ["*embed_tokens*"]},
            "output": {"dir": str(tmp_path / "runs")},
            "scripted_ops": scripted_ops or [],
        }
        for k, v in over.items():
            raw[k].update(v) if isinstance(raw.get(k), dict) else raw.__setitem__(k, v)
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump(raw))
        from weightwright.spec import load_spec
        return load_spec(p)
    return _make

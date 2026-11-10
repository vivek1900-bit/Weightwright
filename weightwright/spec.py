"""Spec file: the one YAML file that drives a run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Spec:
    name: str
    model_path: str
    goal: str
    eval_file: str
    model_dtype: str = "bfloat16"
    device: str = "auto"
    backend: str = "hf"
    max_new_tokens: int = 256
    chat: bool = True
    chat_template_kwargs: dict = field(default_factory=dict)
    target_tag: str = "target"
    regression_tag: str = "regression"
    min_gain: float = 0.10
    max_regression_drop: float = 0.02
    editor: dict = field(default_factory=dict)
    max_turns: int = 40
    max_evals: int = 20
    max_touched_fraction: float = 0.0005
    allow: list = field(default_factory=list)
    deny: list = field(default_factory=lambda: ["*embed_tokens*", "*lm_head*", "*norm*"])
    out_dir: str = "runs"
    scripted_ops: list = field(default_factory=list)
    base_dir: Path = Path(".")

    def resolve(self, p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else (self.base_dir / pp))


REQUIRED = ("model", "goal", "eval")


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a YAML mapping")
    for k in REQUIRED:
        if k not in raw:
            raise ValueError(f"{path}: missing required section '{k}'")
    m, e, ed, b, o = raw["model"], raw["eval"], raw.get("editor", {}), raw.get("budget", {}), raw.get("output", {})
    if "path" not in m:
        raise ValueError(f"{path}: model.path is required")
    if "file" not in e:
        raise ValueError(f"{path}: eval.file is required")
    spec = Spec(
        name=raw.get("name", path.stem),
        model_path=m["path"],
        model_dtype=m.get("dtype", "bfloat16"),
        device=m.get("device", "auto"),
        goal=str(raw["goal"]).strip(),
        eval_file=e["file"],
        backend=e.get("backend", "hf"),
        max_new_tokens=int(e.get("max_new_tokens", 256)),
        chat=bool(e.get("chat", True)),
        chat_template_kwargs=e.get("chat_template_kwargs", {}) or {},
        target_tag=e.get("target_tag", "target"),
        regression_tag=e.get("regression_tag", "regression"),
        min_gain=float(e.get("min_gain", 0.10)),
        max_regression_drop=float(e.get("max_regression_drop", 0.02)),
        editor=ed,
        max_turns=int(ed.get("max_turns", 40)),
        max_evals=int(ed.get("max_evals", 20)),
        max_touched_fraction=float(b.get("max_touched_fraction", 0.0005)),
        allow=b.get("allow", []) or [],
        deny=b.get("deny", ["*embed_tokens*", "*lm_head*", "*norm*"]) or [],
        out_dir=o.get("dir", "runs"),
        scripted_ops=raw.get("scripted_ops", []) or [],
        base_dir=Path.cwd(),
    )
    if spec.editor.get("provider", "scripted") == "scripted" and not spec.scripted_ops:
        raise ValueError(f"{path}: provider 'scripted' needs a non-empty scripted_ops list")
    return spec

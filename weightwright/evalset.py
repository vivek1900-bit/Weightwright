"""Eval sets: JSONL files of prompts with simple, deterministic checks, grouped by tag.

Each line: {"prompt": "...", "tag": "target" | "regression" | any, "check": {"type": ..., ...}}
Check types: contains, regex, exact, last_number, no_loop, finished, json_valid
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_LOOP = re.compile(r"(.{10,}?)\1{3,}", re.DOTALL)
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass
class Generation:
    text: str
    hit_limit: bool = False  # True if generation stopped because it ran out of tokens


@dataclass
class EvalItem:
    prompt: str
    tag: str
    check: dict


def load_evalset(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
            items.append(EvalItem(row["prompt"], row.get("tag", "target"), row["check"]))
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"{path}:{n}: bad eval line ({e})") from e
    if not items:
        raise ValueError(f"{path}: eval set is empty")
    return items


def has_loop(text: str) -> bool:
    return bool(_LOOP.search(text[-3000:]))


def passes(check: dict, gen: Generation) -> bool:
    kind = check.get("type")
    text = gen.text
    ci = check.get("ignore_case", True)
    hay = text.lower() if ci else text
    if kind == "contains":
        needles = check["any"] if "any" in check else [check["value"]]
        needles = [n.lower() if ci else n for n in needles]
        return any(n in hay for n in needles)
    if kind == "regex":
        return re.search(check["pattern"], text, re.DOTALL) is not None
    if kind == "exact":
        a, b = text.strip(), str(check["value"]).strip()
        return a.lower() == b.lower() if ci else a == b
    if kind == "last_number":
        nums = _NUM.findall(text)
        if not nums:
            return False
        got = float(nums[-1].replace(",", ""))
        return abs(got - float(check["value"])) < 1e-6
    if kind == "no_loop":
        return not has_loop(text)
    if kind == "finished":
        return not gen.hit_limit
    if kind == "json_valid":
        try:
            json.loads(text.strip())
            return True
        except json.JSONDecodeError:
            return False
    raise ValueError(f"unknown check type {kind!r}")


def check_all(item: EvalItem, gen: Generation) -> bool:
    """A check may be a single dict or {"all": [dict, ...]}."""
    c = item.check
    if "all" in c:
        return all(passes(x, gen) for x in c["all"])
    return passes(c, gen)


@dataclass
class EvalResult:
    scores: dict[str, float]  # tag -> pass rate (0..1)
    failures: list[dict]  # prompt/output/tag of failed items

    def get(self, tag: str) -> float:
        return self.scores.get(tag, float("nan"))


def score(items: list[EvalItem], gens: list[Generation]) -> EvalResult:
    totals: dict[str, int] = {}
    ok: dict[str, int] = {}
    failures: list[dict] = []
    for it, g in zip(items, gens):
        totals[it.tag] = totals.get(it.tag, 0) + 1
        if check_all(it, g):
            ok[it.tag] = ok.get(it.tag, 0) + 1
        else:
            failures.append({"tag": it.tag, "prompt": it.prompt, "output": g.text[-600:], "hit_limit": g.hit_limit})
    return EvalResult({t: ok.get(t, 0) / n for t, n in totals.items()}, failures)

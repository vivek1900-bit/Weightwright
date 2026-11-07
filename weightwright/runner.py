"""The run loop: baseline eval, editor proposes edits, every edit is evaluated, a gate decides."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .backends import Backend, HFBackend
from .budget import Budget, BudgetError
from .editors import make_editor
from .evalset import EvalResult, load_evalset, score
from .ops import EditSet
from .patch import Patch, make_meta, save_patch
from .prompts import OPENING, SYSTEM_PROMPT
from .slicing import parse_index
from .spec import Spec
from .store import SafetensorsStore


@dataclass
class RunResult:
    status: str  # "accepted" | "rejected"
    run_dir: Path
    base: dict[str, float]
    final: dict[str, float]
    patch_dir: Path | None
    reasons: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Pull the first balanced {...} object out of a model reply."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object")


def gate(base: EvalResult, new: EvalResult, spec: Spec) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    gain = new.get(spec.target_tag) - base.get(spec.target_tag)
    if not gain >= spec.min_gain - 1e-9:
        reasons.append(f"target gain {gain:+.1%} is below the required {spec.min_gain:+.0%}")
    if spec.regression_tag in base.scores:
        drop = base.get(spec.regression_tag) - new.get(spec.regression_tag)
        if drop > spec.max_regression_drop + 1e-9:
            reasons.append(f"regression score dropped {drop:.1%}, more than the allowed {spec.max_regression_drop:.0%}")
    return (not reasons), reasons


def make_backend(spec: Spec) -> Backend:
    if spec.backend == "hf":
        return HFBackend(spec.resolve(spec.model_path), spec.model_dtype, spec.device,
                         spec.chat, spec.chat_template_kwargs)
    raise ValueError(f"unknown eval backend {spec.backend!r}")


class Session:
    def __init__(self, spec: Spec, store: SafetensorsStore, backend: Backend, editor, items, run_dir: Path, say=print):
        self.spec, self.store, self.backend, self.editor, self.items = spec, store, backend, editor, items
        self.say = say
        self.run_dir = run_dir
        self.budget = Budget(store.total_params(), spec.max_touched_fraction, spec.allow, spec.deny)
        self.editset = EditSet()
        self.evals_used = 0
        self.evaluated_ops = -1  # number of ops that the latest eval reflects
        self.last: EvalResult | None = None
        self.accepted = False
        self.finished = False
        self.reasons: list[str] = []

    # ---- eval --------------------------------------------------------
    def _eval(self, edits) -> EvalResult:
        if edits:
            self.backend.apply(edits)
        else:
            self.backend.revert()
        gens = self.backend.generate([i.prompt for i in self.items], self.spec.max_new_tokens)
        return score(self.items, gens)

    def baseline(self) -> EvalResult:
        self.base = self._eval([])
        self.last = self.base
        return self.base

    # ---- tools -------------------------------------------------------
    def handle(self, a: dict) -> str:
        act = a.get("action")
        fn = getattr(self, f"_do_{act}", None)
        if fn is None:
            return f"ERROR: unknown action {act!r}"
        try:
            return fn(a)
        except (KeyError, ValueError, BudgetError, TypeError, IndexError) as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _do_list_tensors(self, a):
        names = self.store.names(a.get("pattern"))
        lim = int(a.get("limit", 100))
        lines = [f"{n}  {list(self.store.info(n).shape)}  {self.store.info(n).dtype}" for n in names[:lim]]
        more = f"\n... {len(names) - lim} more, narrow the pattern" if len(names) > lim else ""
        return "\n".join(lines) + more if lines else "no tensors match"

    def _region(self, a) -> torch.Tensor:
        t = self.store.get(a["tensor"]).float()
        return t[parse_index(a.get("index"), t.ndim)]

    def _do_stats(self, a):
        r = self._region(a)
        out = {"shape": list(r.shape), "mean": round(r.mean().item(), 6), "std": round(r.std().item(), 6) if r.numel() > 1 else 0.0,
               "absmax": round(r.abs().max().item(), 6), "zero_fraction": round((r == 0).float().mean().item(), 6),
               "frobenius_norm": round(r.norm().item(), 4)}
        k = int(a.get("svd_k", 0))
        if k and r.ndim == 2:
            if r.numel() <= 4_000_000:
                s = torch.linalg.svdvals(r)[:k]
            else:
                torch.manual_seed(0)
                s = torch.svd_lowrank(r, q=k + 4, niter=3)[1][:k]
            out["top_singular_values"] = [round(x, 4) for x in s.tolist()]
        return json.dumps(out)

    def _do_read_slice(self, a):
        r = self._region(a)
        if r.numel() > 256:
            raise ValueError(f"slice has {r.numel()} values; read at most 256")
        return json.dumps([round(x, 5) for x in r.reshape(-1).tolist()]) + f"  shape={list(r.shape)}"

    def _do_show_failures(self, a):
        n = int(a.get("n", 5))
        fails = self.last.failures[:n] if self.last else []
        if not fails:
            return "no failing items"
        return "\n---\n".join(f"[{f['tag']}] prompt: {f['prompt']}\noutput (end): {f['output']}\nhit token limit: {f['hit_limit']}" for f in fails)

    def _do_propose_edit(self, a):
        op = a["op"]
        trial = EditSet(self.editset.ops + [op])
        edits = trial.build(self.store, self.budget)  # raises on budget / bad op
        self.editset = trial
        touched = sum(e.count for e in edits)
        return (f"staged. {len(self.editset.ops)} op(s) staged, {touched:,} weights would change "
                f"({self.budget.remaining(touched):,} left in budget). Call evaluate to test.")

    def _do_clear_staged(self, a):
        self.editset.clear()
        self.evaluated_ops = -1
        return "staged edits cleared"

    def _evaluate_staged(self) -> tuple[EvalResult, list]:
        edits = self.editset.build(self.store, self.budget)
        res = self._eval(edits)
        self.last = res
        self.evals_used += 1
        self.evaluated_ops = len(self.editset.ops)
        ck = self.run_dir / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / f"eval_{self.evals_used:02d}.json").write_text(json.dumps(
            {"ops": self.editset.ops, "scores": res.scores, "touched": sum(e.count for e in edits)}, indent=2))
        return res, edits

    def _summary(self, res: EvalResult, edits) -> str:
        sp = self.spec
        t, r = res.get(sp.target_tag), res.get(sp.regression_tag)
        bt, br = self.base.get(sp.target_tag), self.base.get(sp.regression_tag)
        return (f"target {t:.1%} (baseline {bt:.1%}, {t - bt:+.1%}); regression {r:.1%} (baseline {br:.1%}, {r - br:+.1%}); "
                f"{sum(e.count for e in edits):,} weights touched; evals used {self.evals_used}/{sp.max_evals}")

    def _do_evaluate(self, a):
        if not self.editset.ops:
            return "nothing staged; propose_edit first"
        if self.evals_used >= self.spec.max_evals:
            return "no evaluations left; submit or abort"
        res, edits = self._evaluate_staged()
        return self._summary(res, edits)

    def _do_submit(self, a):
        if not self.editset.ops:
            return "nothing staged; propose_edit first"
        if self.evaluated_ops != len(self.editset.ops):
            if self.evals_used >= self.spec.max_evals:
                return "staged edits were never evaluated and no evaluations are left; abort"
            res, edits = self._evaluate_staged()
        else:
            edits = self.editset.build(self.store, self.budget)
            res = self.last
        ok, reasons = gate(self.base, res, self.spec)
        patch = Patch(edits, make_meta(self.store, edits, self.editset.ops, scores=res.scores,
                                       baseline_scores=self.base.scores, goal=self.spec.goal))
        if ok:
            save_patch(self.run_dir / "patch", patch)
            self.accepted, self.finished = True, True
            return "ACCEPTED. " + self._summary(res, edits)
        save_patch(self.run_dir / "rejected_patch", patch)
        self.reasons = reasons
        return "REJECTED by the gate: " + "; ".join(reasons) + ". You may revise and submit again."

    def _do_abort(self, a):
        self.finished = True
        self.reasons = [f"editor aborted: {a.get('reason', 'no reason given')}"]
        return "aborted"

    # ---- main loop ---------------------------------------------------
    def loop(self) -> None:
        sp = self.spec
        system = SYSTEM_PROMPT.format(goal=sp.goal, max_touched=self.budget.max_touched, fraction=sp.max_touched_fraction,
                                      deny=sp.deny or "none", allow=sp.allow or "all not denied", max_evals=sp.max_evals,
                                      max_turns=sp.max_turns, target_tag=sp.target_tag, min_gain=sp.min_gain,
                                      regression_tag=sp.regression_tag, max_drop=sp.max_regression_drop)
        reg = self.base.scores.get(sp.regression_tag)
        opening = OPENING.format(target_tag=sp.target_tag, t=self.base.get(sp.target_tag), regression_tag=sp.regression_tag,
                                 r=f"{reg:.1%}" if reg is not None else "none (no regression items!)",
                                 n_tensors=len(self.store.names()), total=self.store.total_params())
        messages = [{"role": "user", "content": opening}]
        log = (self.run_dir / "log.jsonl").open("a")
        for turn in range(1, sp.max_turns + 1):
            raw = self.editor.chat(system, messages)
            messages.append({"role": "assistant", "content": raw})
            try:
                action = extract_json(raw)
                result = self.handle(action)
            except (ValueError, json.JSONDecodeError) as e:
                action, result = {"action": "?"}, f"ERROR: reply was not a single valid JSON object ({e})"
            self.say(f"[turn {turn}] {action.get('action')}: {result.splitlines()[0][:160]}")
            log.write(json.dumps({"turn": turn, "action": action, "result": result[:2000]}) + "\n")
            log.flush()
            if self.finished:
                break
            messages.append({"role": "user", "content": result})
        else:
            self.reasons = self.reasons or ["turn limit reached without an accepted submit"]
        log.close()
        self.backend.revert()


def run(spec: Spec, backend: Backend | None = None, editor=None, say=print) -> RunResult:
    store = SafetensorsStore(spec.resolve(spec.model_path))
    items = load_evalset(spec.resolve(spec.eval_file))
    backend = backend or make_backend(spec)
    editor = editor or make_editor(spec.editor, spec.scripted_ops)
    run_dir = Path(spec.resolve(spec.out_dir)) / f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    s = Session(spec, store, backend, editor, items, run_dir, say)
    base = s.baseline()
    say(f"baseline: {json.dumps({k: round(v, 3) for k, v in base.scores.items()})}")
    if spec.regression_tag not in base.scores:
        say(f"WARNING: no eval items tagged '{spec.regression_tag}'; the gate cannot detect regressions")
    (run_dir / "baseline.json").write_text(json.dumps(base.scores, indent=2))
    s.loop()
    final = s.last.scores if s.last else base.scores
    status = "accepted" if s.accepted else "rejected"
    summary = {"status": status, "baseline": base.scores, "final": final, "reasons": s.reasons}
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return RunResult(status, run_dir, base.scores, final, run_dir / "patch" if s.accepted else None, s.reasons)

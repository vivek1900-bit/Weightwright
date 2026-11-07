"""wright: command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .patch import PatchMismatch, load_patch, materialize
from .store import SafetensorsStore


def _human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.1f}T"


def cmd_init(a) -> int:
    from .templates import EXAMPLE_EVAL, EXAMPLE_SPEC
    root = Path(a.dir)
    (root / "specs").mkdir(parents=True, exist_ok=True)
    (root / "evals").mkdir(parents=True, exist_ok=True)
    for p, text in ((root / "specs" / "example.yaml", EXAMPLE_SPEC), (root / "evals" / "example.jsonl", EXAMPLE_EVAL)):
        if p.exists() and not a.force:
            print(f"skip (exists): {p}")
        else:
            p.write_text(text)
            print(f"wrote {p}")
    return 0


def cmd_inspect(a) -> int:
    store = SafetensorsStore(a.model)
    names = store.names(a.pattern)
    print(f"{len(store.names())} tensors, {_human(store.total_params())} parameters, shards: {', '.join(store.shards())}")
    for n in names[: a.limit]:
        i = store.info(n)
        print(f"{n:70s} {str(list(i.shape)):22s} {i.dtype}")
    if len(names) > a.limit:
        print(f"... {len(names) - a.limit} more (use --pattern or --limit)")
    return 0


def cmd_run(a) -> int:
    from .runner import run
    from .spec import load_spec
    res = run(load_spec(a.spec))
    print(f"\nstatus: {res.status}\nrun folder: {res.run_dir}")
    if res.patch_dir:
        print(f"patch saved: {res.patch_dir}\napply it with: wright apply {res.patch_dir} --model <model dir> --out <new dir>")
    for r in res.reasons:
        print(f"reason: {r}")
    return 0 if res.status == "accepted" else 2


def cmd_eval(a) -> int:
    from .evalset import load_evalset, score
    from .runner import make_backend
    from .spec import load_spec
    spec = load_spec(a.spec)
    backend = make_backend(spec)
    if a.patch:
        backend.apply(load_patch(a.patch).edits)
    items = load_evalset(spec.resolve(spec.eval_file))
    res = score(items, backend.generate([i.prompt for i in items], spec.max_new_tokens))
    print(json.dumps({k: round(v, 4) for k, v in res.scores.items()}, indent=2))
    if a.show_failures:
        print(f"\n{len(res.failures)} failing item(s):")
        for f in res.failures:
            print(f"\n[{f['tag']}] {f['prompt']}\n  hit token limit: {f['hit_limit']}\n  output (end): {f['output']!r}")
    return 0


def _apply(a, revert: bool) -> int:
    store = SafetensorsStore(a.model)
    patch = load_patch(a.patch)
    try:
        out = materialize(store, patch, a.out, revert=revert, force=a.force)
    except PatchMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{'reverted' if revert else 'patched'} model written to {out} ({patch.touched:,} weights changed)")
    return 0


def cmd_info(a) -> int:
    meta = load_patch(a.patch).meta
    print(json.dumps({k: v for k, v in meta.items() if k != "ops"}, indent=2))
    print(f"ops: {len(meta.get('ops', []))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wright", description="Edit LLM weights as reversible, evaluated patches.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write an example spec and eval file")
    s.add_argument("dir", nargs="?", default=".")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("inspect", help="list tensors in a model folder (no weights are loaded)")
    s.add_argument("model")
    s.add_argument("--pattern")
    s.add_argument("--limit", type=int, default=60)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("run", help="run the full edit loop from a spec file")
    s.add_argument("spec")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval", help="score the model (optionally with a patch) on the spec's eval set")
    s.add_argument("spec")
    s.add_argument("--patch")
    s.add_argument("--show-failures", action="store_true", help="print each failing prompt and the model's output")
    s.set_defaults(fn=cmd_eval)

    for name, rev in (("apply", False), ("revert", True)):
        s = sub.add_parser(name, help=f"{name} a patch and write a new model folder (source untouched)")
        s.add_argument("patch")
        s.add_argument("--model", required=True, help="source model folder")
        s.add_argument("--out", required=True, help="output folder")
        s.add_argument("--force", action="store_true", help="skip the expected-state check")
        s.set_defaults(fn=(lambda a, r=rev: _apply(a, r)))

    s = sub.add_parser("info", help="show a patch's metadata")
    s.add_argument("patch")
    s.set_defaults(fn=cmd_info)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

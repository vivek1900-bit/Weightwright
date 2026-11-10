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


def _print_run_findings(report: "ValidationReport") -> None:
    # Quoted so cli.py does not need a new runtime import for the annotation.
    """Print run validation errors and warnings to stderr."""
    for finding in report.findings:
        if finding.severity not in {"error", "warn"}:
            continue
        print(
            f"[{finding.severity.upper()}] {finding.code}: {finding.message}",
            file=sys.stderr,
        )
        if finding.hint:
            print(f"    hint: {finding.hint}", file=sys.stderr)


def cmd_validate(a) -> int:
    from .validate import format_report, report_to_dict, validate_spec_file

    report = validate_spec_file(
        a.spec,
        check_backend=not a.no_backend_checks,
        check_env=not a.no_env_checks,
    )
    if a.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_report(report))
    if report.errors:
        return 1
    if a.strict and report.warnings:
        return 1
    return 0


def cmd_verify(a) -> int:
    from .patch import load_patch, verify_patch
    from .store import SafetensorsStore

    patch = load_patch(a.patch)
    store = SafetensorsStore(a.model)
    report = verify_patch(store, patch)
    if a.expect == "auto":
        ok = report.state in {"base", "patched"}
    else:
        ok = report.state == a.expect

    if a.json:
        print(
            json.dumps(
                {
                    "state": report.state,
                    "expected": a.expect,
                    "ok": ok,
                    "tensors": [
                        {
                            "name": verdict.name,
                            "status": verdict.status,
                            "detail": verdict.detail,
                            "full_check": verdict.full_check,
                        }
                        for verdict in report.verdicts
                    ],
                },
                indent=2,
            )
        )
    else:
        for verdict in report.verdicts:
            print(f"{verdict.name}  {verdict.status}  {verdict.detail}")
        print(f"state: {report.state}")
    return 0 if ok else 2


def cmd_run(a) -> int:
    from .spec import load_spec
    from .validate import validate_spec

    spec = load_spec(a.spec)
    if not getattr(a, "skip_validate", False):
        report = validate_spec(spec)
        _print_run_findings(report)
        if report.errors:
            print(
                (
                    "validation failed: "
                    f"{len(report.errors)} error(s), "
                    f"{len(report.warnings)} warning(s); model was not loaded"
                ),
                file=sys.stderr,
            )
            return 1

    from .runner import run

    res = run(spec)
    print(f"\nstatus: {res.status}\nrun folder: {res.run_dir}")
    if res.patch_dir:
        print(
            f"patch saved: {res.patch_dir}\n"
            f"apply it with: wright apply {res.patch_dir} "
            "--model <model dir> --out <new dir>"
        )
    for reason in res.reasons:
        print(f"reason: {reason}")
    return 0 if res.status == "accepted" else 2


def cmd_info(a) -> int:
    meta = load_patch(a.patch).meta
    print(
        json.dumps(
            {
                key: value
                for key, value in meta.items()
                if key not in {"ops", "tensor_checks"}
            },
            indent=2,
        )
    )
    print(f"ops: {len(meta.get('ops', []))}")
    version = meta.get("patch_version", 1)
    checks = meta.get("tensor_checks")
    if version == 2 and isinstance(checks, list):
        print(f"checksums: yes ({len(checks)} tensors, full SHA-256)")
    else:
        print(
            f"checksums: no (patch_version {version}: "
            "element-level check only)"
        )
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
    s.add_argument("--skip-validate", action="store_true", help="skip spec, model and eval validation before running")
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

    s = sub.add_parser(
        "validate",
        help="check a spec, model folder and eval set without loading weights",
    )
    s.add_argument("spec")
    s.add_argument("--json", action="store_true", help="print the report as JSON")
    s.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when warnings are present",
    )
    s.add_argument(
        "--no-backend-checks",
        action="store_true",
        help="skip transformers, model metadata and CUDA checks",
    )
    s.add_argument(
        "--no-env-checks",
        action="store_true",
        help="skip editor API key environment checks",
    )
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser(
        "verify",
        help="verify a patch against a model without writing",
    )
    s.add_argument("patch")
    s.add_argument("--model", required=True, help="model folder to verify against")
    s.add_argument(
        "--expect",
        choices=("base", "patched", "auto"),
        default="auto",
        help="required patch state (default: auto accepts base or patched)",
    )
    s.add_argument("--json", action="store_true", help="print the report as JSON")
    s.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

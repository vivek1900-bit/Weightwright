"""Patches: save, load, verify, materialize and exactly revert sparse weight edits.

On disk a patch is a folder:
  patch.json        metadata, ops, scores, touched counts, tensor_checks (v2)
  edits.safetensors flat indices plus old and new values for every touched tensor
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .ops import TensorEdit
from .store import SafetensorsStore

PATCH_VERSION = 2
HASH_CHUNK_BYTES = 64 * 1024 * 1024

STATUS_MEANINGS: dict[str, str] = {
    "base": "tensor has unpatched base weights",
    "patched": "tensor already has patched weights",
    "mismatch": "tensor weights differ from both base and patched states",
    "missing": "tensor is missing from the model",
    "shape_mismatch": "tensor shape does not match the patch",
    "dtype_mismatch": "tensor dtype does not match the patch",
}


def tensor_sha256(t: torch.Tensor) -> str:
    """Compute SHA-256 over dtype, shape, and raw row-major bytes in native (little-endian) byte order."""
    hasher = hashlib.sha256()
    header = f"{t.dtype}|{tuple(t.shape)}|".encode()
    hasher.update(header)
    raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if raw.numel() > 0:
        mv = memoryview(raw.numpy())
        for offset in range(0, len(mv), HASH_CHUNK_BYTES):
            hasher.update(mv[offset : offset + HASH_CHUNK_BYTES])
    return hasher.hexdigest()


@dataclass
class Patch:
    edits: list[TensorEdit]
    meta: dict = field(default_factory=dict)

    @property
    def touched(self) -> int:
        return sum(e.count for e in self.edits)


@dataclass
class TensorVerdict:
    """Verification outcome for a single edited tensor."""

    name: str
    status: str
    detail: str
    full_check: bool


@dataclass
class VerifyReport:
    """Aggregate verification report across all edits in a patch."""

    verdicts: list[TensorVerdict]

    @property
    def state(self) -> str:
        if not self.verdicts:
            return "empty"
        statuses = {v.status for v in self.verdicts}
        if not statuses.issubset({"base", "patched"}):
            return "mismatch"
        if statuses == {"base"}:
            return "base"
        if statuses == {"patched"}:
            return "patched"
        return "mixed"

    def explain(self, expected: str) -> str:
        """Return multi-line text explaining every tensor not in the expected state."""
        lines = [f"patch verification failed: model state is '{self.state}', expected '{expected}'"]
        for v in self.verdicts:
            if v.status != expected:
                meaning = STATUS_MEANINGS.get(v.status, v.status)
                lines.append(f"  {v.name}: {v.status} ({meaning}) - {v.detail}")
        return "\n".join(lines)


def make_meta(store: SafetensorsStore, edits: list[TensorEdit], ops: list[dict], **extra) -> dict:
    """Build patch metadata including full-tensor before/after SHA-256 checksums."""
    touched = sum(e.count for e in edits)
    total = store.total_params()
    tensor_checks: list[dict] = []
    for e in edits:
        orig = store.get(e.tensor)
        sha_before = tensor_sha256(orig)
        cloned = orig.clone().contiguous()
        cloned.view(-1)[e.idx] = e.new
        sha_after = tensor_sha256(cloned)
        tensor_checks.append(
            {
                "name": e.tensor,
                "shape": list(orig.shape),
                "dtype": str(orig.dtype),
                "changed": e.count,
                "sha256_before": sha_before,
                "sha256_after": sha_after,
            }
        )
    meta = {
        "patch_version": PATCH_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": str(store.dir),
        "tensors": [e.tensor for e in edits],
        "params_total": total,
        "params_touched": touched,
        "fraction_touched": touched / total if total else 0.0,
        "ops": ops,
        "tensor_checks": tensor_checks,
    }
    meta.update(extra)
    return meta


def save_patch(path: str | Path, patch: Patch) -> None:
    """Save patch metadata and sparse tensor edits to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for i, e in enumerate(patch.edits):
        tensors[f"{i}.idx"] = e.idx.contiguous()
        tensors[f"{i}.old"] = e.old.contiguous()
        tensors[f"{i}.new"] = e.new.contiguous()
    save_file(tensors, str(path / "edits.safetensors"), metadata={"format": "pt"})
    meta = dict(patch.meta)
    meta["tensors"] = [e.tensor for e in patch.edits]
    (path / "patch.json").write_text(json.dumps(meta, indent=2))


def load_patch(path: str | Path) -> Patch:
    """Load a version 1 or version 2 patch from disk."""
    path = Path(path)
    meta = json.loads((path / "patch.json").read_text())
    version = meta.get("patch_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported patch_version: {version}")
    data = load_file(str(path / "edits.safetensors"))
    edits = [
        TensorEdit(tensor=name, idx=data[f"{i}.idx"], old=data[f"{i}.old"], new=data[f"{i}.new"])
        for i, name in enumerate(meta["tensors"])
    ]
    return Patch(edits=edits, meta=meta)


class PatchMismatch(RuntimeError):
    pass


def _has_v2_checks(patch: Patch) -> bool:
    checks = patch.meta.get("tensor_checks")
    return isinstance(checks, list) and len(checks) == len(patch.edits)


def verify_patch(store: SafetensorsStore, patch: Patch) -> VerifyReport:
    """Verify each edited tensor in store against patch checksums or v1 element values."""
    verdicts: list[TensorVerdict] = []
    has_checks = _has_v2_checks(patch)
    checks: list[dict] = patch.meta.get("tensor_checks", []) if has_checks else []

    for i, e in enumerate(patch.edits):
        try:
            store.info(e.tensor)
        except KeyError:
            verdicts.append(
                TensorVerdict(
                    name=e.tensor,
                    status="missing",
                    detail=f"tensor '{e.tensor}' not found in model store",
                    full_check=False,
                )
            )
            continue

        t = store.get(e.tensor)

        if has_checks:
            chk = checks[i]
            expected_shape = tuple(chk["shape"])
            expected_dtype = str(chk["dtype"])
            if tuple(t.shape) != expected_shape:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="shape_mismatch",
                        detail=f"shape {list(t.shape)} != expected {list(expected_shape)}",
                        full_check=False,
                    )
                )
                continue
            if str(t.dtype) != expected_dtype:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="dtype_mismatch",
                        detail=f"dtype {t.dtype} != expected {expected_dtype}",
                        full_check=False,
                    )
                )
                continue

            digest = tensor_sha256(t)
            sha_before = chk["sha256_before"]
            sha_after = chk["sha256_after"]
            if digest == sha_before:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="base",
                        detail=f"sha256 {digest[:12]} matches before ({sha_before[:12]})",
                        full_check=True,
                    )
                )
            elif digest == sha_after:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="patched",
                        detail=f"sha256 {digest[:12]} matches after ({sha_after[:12]})",
                        full_check=True,
                    )
                )
            else:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="mismatch",
                        detail=(
                            f"sha256 {digest[:12]} matches neither "
                            f"before ({sha_before[:12]}) nor after ({sha_after[:12]})"
                        ),
                        full_check=True,
                    )
                )
        else:
            flat = t.contiguous().view(-1)
            if flat.dtype != e.old.dtype or (e.idx.numel() > 0 and int(e.idx.max()) >= flat.numel()):
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="mismatch",
                        detail="element-level check only (patch has no checksums): shape or dtype incompatible",
                        full_check=False,
                    )
                )
                continue

            cur = flat[e.idx]
            if torch.equal(cur, e.old):
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="base",
                        detail="matches old values; element-level check only (patch has no checksums)",
                        full_check=False,
                    )
                )
            elif torch.equal(cur, e.new):
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="patched",
                        detail="matches new values; element-level check only (patch has no checksums)",
                        full_check=False,
                    )
                )
            else:
                verdicts.append(
                    TensorVerdict(
                        name=e.tensor,
                        status="mismatch",
                        detail="touched elements differ; element-level check only (patch has no checksums)",
                        full_check=False,
                    )
                )

    return VerifyReport(verdicts=verdicts)


def materialize(
    store: SafetensorsStore,
    patch: Patch,
    out_dir: str | Path,
    revert: bool = False,
    link_untouched: bool = True,
    force: bool = False,
) -> Path:
    """Write a full model folder with the patch applied (or reverted). The source is never modified.

    Only shards that contain touched tensors are rewritten, and a shard is held in RAM while
    it is rewritten, so peak memory is about one shard.
    """
    out = Path(out_dir)
    if out.resolve() == store.dir.resolve():
        raise ValueError("output directory must differ from the source model directory")

    expected_state = "patched" if revert else "base"
    if not force:
        report = verify_patch(store, patch)
        if report.state not in (expected_state, "empty"):
            raise PatchMismatch(report.explain(expected_state))

    out.mkdir(parents=True, exist_ok=True)

    has_checks = _has_v2_checks(patch)
    checks_by_idx: dict[int, dict] = {}
    if has_checks:
        for i, chk in enumerate(patch.meta["tensor_checks"]):
            checks_by_idx[i] = chk

    by_shard: dict[str, list[tuple[int, TensorEdit]]] = {}
    for i, e in enumerate(patch.edits):
        by_shard.setdefault(store.info(e.tensor).shard, []).append((i, e))

    for src in store.dir.iterdir():
        if src.is_dir():
            continue
        dst = out / src.name
        if src.suffix == ".safetensors" and src.name in by_shard:
            tensors = load_file(str(src))
            for edit_idx, e in by_shard[src.name]:
                t = tensors[e.tensor].contiguous()
                flat = t.view(-1)
                expect = e.new if revert else e.old
                target = e.old if revert else e.new
                if not force and not torch.equal(flat[e.idx], expect):
                    what = "applied" if revert else "unpatched"
                    raise PatchMismatch(
                        f"{e.tensor}: weights are not in the expected {what} state; "
                        "wrong base model or patch already used (use force=True to override)"
                    )
                flat[e.idx] = target
                tensors[e.tensor] = t
                if has_checks and not force:
                    chk = checks_by_idx[edit_idx]
                    expected_hash = chk["sha256_before"] if revert else chk["sha256_after"]
                    actual_hash = tensor_sha256(t)
                    if actual_hash != expected_hash:
                        raise PatchMismatch(
                            f"{e.tensor}: output was not produced correctly "
                            f"(post-edit sha256 {actual_hash[:12]} != expected {expected_hash[:12]}; "
                            "this indicates an internal bug)"
                        )
            save_file(tensors, str(dst), metadata={"format": "pt"})
        elif src.suffix == ".safetensors" and link_untouched:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
    return out

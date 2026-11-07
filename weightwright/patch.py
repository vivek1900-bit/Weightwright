"""Patches: save, load, materialize and exactly revert sparse weight edits.

On disk a patch is a folder:
  patch.json        metadata, ops, scores, touched counts
  edits.safetensors flat indices plus old and new values for every touched tensor
"""
from __future__ import annotations

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

PATCH_VERSION = 1


@dataclass
class Patch:
    edits: list[TensorEdit]
    meta: dict = field(default_factory=dict)

    @property
    def touched(self) -> int:
        return sum(e.count for e in self.edits)


def make_meta(store: SafetensorsStore, edits: list[TensorEdit], ops: list[dict], **extra) -> dict:
    touched = sum(e.count for e in edits)
    total = store.total_params()
    meta = {
        "patch_version": PATCH_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": str(store.dir),
        "tensors": [e.tensor for e in edits],
        "params_total": total,
        "params_touched": touched,
        "fraction_touched": touched / total if total else 0.0,
        "ops": ops,
    }
    meta.update(extra)
    return meta


def save_patch(path: str | Path, patch: Patch) -> None:
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
    path = Path(path)
    meta = json.loads((path / "patch.json").read_text())
    data = load_file(str(path / "edits.safetensors"))
    edits = [
        TensorEdit(tensor=name, idx=data[f"{i}.idx"], old=data[f"{i}.old"], new=data[f"{i}.new"])
        for i, name in enumerate(meta["tensors"])
    ]
    return Patch(edits=edits, meta=meta)


class PatchMismatch(RuntimeError):
    pass


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
    out.mkdir(parents=True, exist_ok=True)

    by_shard: dict[str, list[TensorEdit]] = {}
    for e in patch.edits:
        by_shard.setdefault(store.info(e.tensor).shard, []).append(e)

    for src in store.dir.iterdir():
        if src.is_dir():
            continue
        dst = out / src.name
        if src.suffix == ".safetensors" and src.name in by_shard:
            tensors = load_file(str(src))
            for e in by_shard[src.name]:
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

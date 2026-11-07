"""Edit operations and the EditSet that turns a list of ops into exact, sparse tensor edits.

Every op works on a float32 copy of the chosen region, then the result is cast back to
the tensor's storage dtype. The final edit is the *diff* between original and result, so
the patch only stores weights that really changed, and can be reverted exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .budget import Budget
from .slicing import parse_index
from .store import SafetensorsStore

OP_KINDS = ("scale", "zero", "set", "project_out", "add_delta")


@dataclass
class TensorEdit:
    tensor: str
    idx: torch.Tensor  # flat int64 indices into the tensor
    old: torch.Tensor  # 1-D, storage dtype
    new: torch.Tensor  # 1-D, storage dtype

    @property
    def count(self) -> int:
        return int(self.idx.numel())


def _load_vector(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(p)).float()
    obj = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        obj = next(iter(obj.values()))
    return obj.float()


def _singular_vector(store: SafetensorsStore, spec: dict, cache: dict) -> torch.Tensor:
    name = spec["tensor"]
    mat = (cache[name] if name in cache else store.get(name)).float()
    if mat.ndim != 2:
        raise ValueError(f"singular_vector needs a 2-D tensor, {name} has {mat.ndim} dims")
    which = int(spec.get("which", 0))
    side = spec.get("side", "left")
    if mat.numel() <= 4_000_000:
        u, _s, vh = torch.linalg.svd(mat, full_matrices=False)
        vec = u[:, which] if side == "left" else vh[which, :]
    else:
        g = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        u, _s, v = torch.svd_lowrank(mat, q=which + 4, niter=3)
        vec = u[:, which] if side == "left" else v[:, which]
    return vec


def _direction(store: SafetensorsStore, spec: dict, cache: dict) -> torch.Tensor:
    src = spec.get("source")
    if src == "inline":
        d = torch.tensor(spec["values"], dtype=torch.float32)
    elif src == "file":
        d = _load_vector(spec["path"])
    elif src == "singular_vector":
        d = _singular_vector(store, spec, cache)
    else:
        raise ValueError(f"unknown direction source {src!r}")
    n = d.norm()
    if n == 0:
        raise ValueError("direction has zero length")
    return d / n


def _apply_one(store: SafetensorsStore, op: dict, cache: dict) -> None:
    kind = op.get("op")
    if kind not in OP_KINDS:
        raise ValueError(f"unknown op {kind!r}; allowed: {', '.join(OP_KINDS)}")
    name = op["tensor"]
    if name not in cache:
        cache[name] = store.get(name).clone()
    cur = cache[name]
    dtype = cur.dtype
    work = cur.float()
    idx = parse_index(op.get("index"), work.ndim)
    region = work[idx]

    if kind == "scale":
        new_region = region * float(op["factor"])
    elif kind == "zero":
        new_region = torch.zeros_like(region)
    elif kind == "set":
        new_region = torch.full_like(region, float(op["value"]))
    elif kind == "project_out":
        if region.ndim != 2:
            raise ValueError("project_out needs a 2-D region")
        d = _direction(store, op["direction"], cache)
        axis = op.get("axis", "out")
        if axis == "out":
            if d.numel() != region.shape[0]:
                raise ValueError(f"direction length {d.numel()} != rows {region.shape[0]}")
            new_region = region - torch.outer(d, d @ region)
        elif axis == "in":
            if d.numel() != region.shape[1]:
                raise ValueError(f"direction length {d.numel()} != cols {region.shape[1]}")
            new_region = region - torch.outer(region @ d, d)
        else:
            raise ValueError("axis must be 'out' or 'in'")
    else:  # add_delta
        delta = _load_vector(op["delta_file"])
        if tuple(delta.shape) != tuple(region.shape):
            raise ValueError(f"delta shape {tuple(delta.shape)} != region {tuple(region.shape)}")
        new_region = region + float(op.get("scale", 1.0)) * delta

    work[idx] = new_region
    cache[name] = work.to(dtype)


class EditSet:
    """An ordered list of ops. build() turns it into sparse TensorEdits against the store."""

    def __init__(self, ops: list[dict] | None = None):
        self.ops: list[dict] = list(ops or [])

    def add(self, op: dict) -> None:
        self.ops.append(op)

    def clear(self) -> None:
        self.ops.clear()

    def build(self, store: SafetensorsStore, budget: Budget | None = None) -> list[TensorEdit]:
        cache: dict[str, torch.Tensor] = {}
        for op in self.ops:
            if budget is not None:
                budget.check_name(op["tensor"])
            _apply_one(store, op, cache)
        edits: list[TensorEdit] = []
        for name, final in cache.items():
            orig = store.get(name).contiguous()
            final = final.contiguous()
            changed = (orig.view(-1) != final.view(-1)).nonzero().squeeze(1)
            if changed.numel() == 0:
                continue
            edits.append(
                TensorEdit(
                    tensor=name,
                    idx=changed.to(torch.int64),
                    old=orig.view(-1)[changed].clone(),
                    new=final.view(-1)[changed].clone(),
                )
            )
        if budget is not None:
            budget.check_total(sum(e.count for e in edits))
        return edits

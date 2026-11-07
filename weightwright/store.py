"""Lazy, read-only access to a model stored as safetensors files.

Nothing is loaded until asked for, so a 7B model can be inspected on a laptop
with little RAM. Writing happens only when a patch is materialized (patch.py).
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str  # safetensors dtype string, e.g. "BF16"
    shard: str  # file name that holds it

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class SafetensorsStore:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.dir}")
        self._info: dict[str, TensorInfo] = {}
        self._build_index()

    def _build_index(self) -> None:
        shards = sorted(p.name for p in self.dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors files in {self.dir}")
        for shard in shards:
            with safe_open(str(self.dir / shard), framework="pt") as f:
                for name in f.keys():
                    sl = f.get_slice(name)
                    self._info[name] = TensorInfo(
                        name=name,
                        shape=tuple(sl.get_shape()),
                        dtype=sl.get_dtype(),
                        shard=shard,
                    )

    def names(self, pattern: str | None = None) -> list[str]:
        names = sorted(self._info)
        if pattern:
            names = [n for n in names if fnmatch.fnmatch(n, pattern)]
        return names

    def info(self, name: str) -> TensorInfo:
        if name not in self._info:
            raise KeyError(f"unknown tensor: {name}")
        return self._info[name]

    def total_params(self) -> int:
        return sum(i.numel for i in self._info.values())

    def shards(self) -> list[str]:
        return sorted({i.shard for i in self._info.values()})

    def get(self, name: str) -> torch.Tensor:
        info = self.info(name)
        with safe_open(str(self.dir / info.shard), framework="pt") as f:
            return f.get_tensor(name)

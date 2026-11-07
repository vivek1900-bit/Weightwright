"""Edit budget: which tensors may be touched and how much of the model may change."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field


class BudgetError(ValueError):
    pass


@dataclass
class Budget:
    total_params: int
    max_touched_fraction: float = 0.001
    allow: list[str] = field(default_factory=list)  # empty = everything not denied
    deny: list[str] = field(default_factory=list)

    @property
    def max_touched(self) -> int:
        return int(self.total_params * self.max_touched_fraction)

    def check_name(self, name: str) -> None:
        if any(fnmatch.fnmatch(name, p) for p in self.deny):
            raise BudgetError(f"tensor {name!r} is on the deny list")
        if self.allow and not any(fnmatch.fnmatch(name, p) for p in self.allow):
            raise BudgetError(f"tensor {name!r} is not on the allow list")

    def check_total(self, touched: int) -> None:
        if touched > self.max_touched:
            raise BudgetError(
                f"edit touches {touched:,} weights but the budget is {self.max_touched:,} "
                f"({self.max_touched_fraction:.4%} of {self.total_params:,})"
            )

    def remaining(self, touched: int) -> int:
        return max(0, self.max_touched - touched)

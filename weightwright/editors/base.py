from __future__ import annotations

from typing import Protocol


class Editor(Protocol):
    """Anything that can answer one chat turn. The runner handles tools, budgets and the gate."""

    def chat(self, system: str, messages: list[dict]) -> str: ...

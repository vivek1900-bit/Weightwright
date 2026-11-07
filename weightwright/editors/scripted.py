"""No-AI editor: replays the ops listed in the spec, then evaluates and submits.

Use it to test the whole pipeline (edit, eval, gate, patch, rollback) without any model API.
"""
from __future__ import annotations

import json


class ScriptedEditor:
    def __init__(self, ops: list[dict]):
        self._queue = [{"action": "propose_edit", "op": op} for op in ops]
        self._queue += [{"action": "evaluate"}, {"action": "submit"}]

    def chat(self, system: str, messages: list[dict]) -> str:
        if not self._queue:
            return json.dumps({"action": "abort", "reason": "script finished"})
        return json.dumps(self._queue.pop(0))

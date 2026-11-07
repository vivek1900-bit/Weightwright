from __future__ import annotations

from .base import Editor
from .http_editors import AnthropicEditor, OpenAICompatEditor
from .scripted import ScriptedEditor


def make_editor(cfg: dict, scripted_ops: list) -> Editor:
    provider = cfg.get("provider", "scripted")
    if provider == "scripted":
        return ScriptedEditor(scripted_ops)
    if provider == "openai_compat":
        return OpenAICompatEditor(cfg)
    if provider == "anthropic":
        return AnthropicEditor(cfg)
    raise ValueError(f"unknown editor provider {provider!r} (scripted | openai_compat | anthropic)")

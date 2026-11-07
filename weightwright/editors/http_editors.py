"""Editors that call a model over HTTP, using only the standard library.

openai_compat works with any OpenAI-style /chat/completions server: Ollama, vLLM, llama.cpp
server, LM Studio, or a model on a Kaggle notebook exposed through a tunnel.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"editor API error {e.code}: {e.read().decode(errors='replace')[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach editor API at {url}: {e.reason}") from e


class OpenAICompatEditor:
    def __init__(self, cfg: dict):
        self.base = cfg.get("base_url", "http://localhost:11434/v1").rstrip("/")
        self.model = cfg["model"]
        key_env = cfg.get("api_key_env")
        self.key = os.environ.get(key_env, "") if key_env else cfg.get("api_key", "")
        self.temperature = float(cfg.get("temperature", 0.2))
        self.timeout = int(cfg.get("timeout", 600))

    def chat(self, system: str, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": system}] + messages}
        data = _post(f"{self.base}/chat/completions", headers, payload, self.timeout)
        return data["choices"][0]["message"]["content"] or ""


class AnthropicEditor:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]
        self.key = os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        if not self.key:
            raise RuntimeError("set the API key environment variable named by editor.api_key_env "
                               "(default ANTHROPIC_API_KEY)")
        self.max_tokens = int(cfg.get("max_tokens", 2000))
        self.timeout = int(cfg.get("timeout", 600))

    def chat(self, system: str, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json", "x-api-key": self.key,
                   "anthropic-version": "2023-06-01"}
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   "messages": messages}
        data = _post("https://api.anthropic.com/v1/messages", headers, payload, self.timeout)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

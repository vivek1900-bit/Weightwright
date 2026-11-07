"""Eval backends: run the (patched) model on prompts.

A backend keeps one copy of the model in memory. apply() writes edited weights into it,
revert() restores the originals, so trying an edit costs one eval, not a model reload.
"""
from __future__ import annotations

from typing import Protocol

import torch

from .evalset import Generation
from .ops import TensorEdit


class Backend(Protocol):
    def apply(self, edits: list[TensorEdit]) -> None: ...
    def revert(self) -> None: ...
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generation]: ...


class _Revertible:
    """Shared apply/revert bookkeeping over a name -> tensor lookup."""

    def _param(self, name: str) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def __init__(self) -> None:
        self._applied: list[TensorEdit] = []

    @torch.no_grad()
    def apply(self, edits: list[TensorEdit]) -> None:
        self.revert()
        for e in edits:
            flat = self._param(e.tensor).view(-1)
            flat[e.idx.to(flat.device)] = e.new.to(device=flat.device, dtype=flat.dtype)
        self._applied = list(edits)

    @torch.no_grad()
    def revert(self) -> None:
        for e in self._applied:
            flat = self._param(e.tensor).view(-1)
            flat[e.idx.to(flat.device)] = e.old.to(device=flat.device, dtype=flat.dtype)
        self._applied = []


class HFBackend(_Revertible):
    """Hugging Face transformers model, greedy decoding.

    Works for models that fit your RAM or VRAM. For a 6 GB GPU, develop with a model of about
    1.5B parameters or less, then move up.
    """

    def __init__(self, model_dir: str, dtype: str = "bfloat16", device: str = "auto",
                 chat: bool = True, chat_template_kwargs: dict | None = None, batch_size: int = 4):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("HFBackend needs transformers: pip install 'weightwright[hf]'") from e
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
        kw = {"dtype": td} if ver >= (4, 56) else {"torch_dtype": td}  # newer versions renamed the argument
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(device).eval()
        self.device = device
        self.chat = chat
        self.chat_kwargs = chat_template_kwargs or {}
        self.batch_size = batch_size
        self._params = dict(self.model.named_parameters())
        self._params.update(dict(self.model.named_buffers()))

    def _param(self, name: str) -> torch.Tensor:
        if name not in self._params:
            raise KeyError(f"{name} not found in loaded model (tied or renamed weight?)")
        return self._params[name].data

    def _render(self, prompt: str) -> str:
        if self.chat and getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **self.chat_kwargs)
        return prompt

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generation]:
        out: list[Generation] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = [self._render(p) for p in prompts[i:i + self.batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                ids = row.tolist()
                eos = self.tok.eos_token_id
                finished = (eos in ids) if eos is not None else False
                text = self.tok.decode(row, skip_special_tokens=True)
                out.append(Generation(text=text, hit_limit=not finished and len(ids) >= max_new_tokens))
        return out

# weightwright

Edit a language model's weights directly, save the change as a small reversible patch, and let an evaluation gate decide whether the edit is kept. One command runs the whole loop.

> **Status: v0.1, early alpha.** The pipeline (tensor store, edits, patches, revert, eval gate, editor loop) is covered by automated tests that use a toy model. Baseline scoring (`wright eval`) has been run on a real model (Qwen2.5-0.5B on a laptop). **Not yet shown:** the full edit loop (`wright run`) with a real editor model or API, and any real improvement in a model's score. Treat the first real edit run as a test of the tool, not of your model.

## Why this exists

Recent work showed an LLM can inspect a small model's weights and patch them. MIT researchers reported using GPT-6 Astra to raise Qwen3.5-0.8B's MATH-500 score from 7.9% to 48.1% by editing about 0.016% of its weights (blog post: "Sparks of Weight-Space Programming", Takehi et al., 2026). Most of that gain came from removing a repetition problem, not from adding new knowledge. weightwright packages the surrounding engineering: safe edits, exact rollback, and a regression gate.

## Related work (please read before using)

This is not the first tool in the area. weightwright's focus is the one-command workflow plus the gate.

| Tool | What it does |
|---|---|
| EasyEdit | Knowledge-editing toolkit (ROME, MEMIT and others) |
| mergekit | Merging models (linear, SLERP, task arithmetic, TIES, DARE) |
| weightspace (PyPI) | LLM agent that reads and edits weights, based on the MIT paper |
| Soup | One-command LLM fine-tuning, including adapter arithmetic |

## Install

```bash
pip install -e .            # core: torch, safetensors, pyyaml
pip install -e ".[hf]"      # adds transformers for the Hugging Face eval backend
pip install -e ".[dev]" && pytest
```

## Quick start

```bash
wright init .                          # writes specs/example.yaml and evals/example.jsonl
wright inspect ./models/small-model    # list tensors without loading weights
wright run specs/example.yaml          # baseline eval, editor loop, gate, patch
wright apply runs/<run>/patch --model ./models/small-model --out ./models/small-model-patched
wright revert runs/<run>/patch --model ./models/small-model-patched --out ./models/small-model-restored
```

Relative paths in a spec file (model, eval file, output folder) are resolved from the folder you run `wright` in, not from the spec file's folder.

## How a run works

1. Load the spec (YAML) and the eval set (JSONL). Measure baseline scores.
2. The editor model sees the goal, the limits, and a small JSON tool set: list tensors, read stats and slices, show failing prompts, stage an edit, evaluate, submit.
3. Every edit is applied in memory, evaluated, and rolled back. The model is loaded once.
4. The gate accepts only if the `target` score rises by `min_gain` and the `regression` score drops by no more than `max_regression_drop`.
5. Accepted edits are saved as a patch: flat indices plus the old and new values of every changed weight. Applying and reverting are exact, and wrong-base or double-apply mistakes are refused.

Edit ops: `scale`, `zero`, `set`, `project_out` (remove one direction, for example a top singular vector), `add_delta` (groundwork for v2).

## Safety rails

- Budget: maximum fraction of weights touched, plus allow and deny lists. Embeddings, the output head and norms are denied by default.
- Source models are never modified. `apply` and `revert` write a new folder.
- The editor protocol is plain JSON, so weak models fail gracefully (bad replies are reported back, with a turn limit).

## Editors

`openai_compat` (Ollama, vLLM, llama.cpp server, LM Studio, or any model you expose through a tunnel), `anthropic`, and `scripted` (no AI, replays ops from the spec; use it to test the pipeline). How well a given model performs as the editor is untested. Make it swappable and measure.

## Hardware notes

Inspection and patch files are lazy and cheap. Evaluation needs the model in memory: on a 6 GB GPU, develop with a model around 1.5B parameters or less. Editing quantized (GGUF or Q5) weights is not supported. Edit the full-precision safetensors, then quantize the result. `apply` holds one shard in RAM at a time.

## Known limits

- Only the Hugging Face backend exists. Weights that are tied or renamed in transformers need care.
- Greedy decoding, deterministic checks only. No LLM-as-judge.
- Sandbox resume after interruption is not implemented (checkpoints are written for inspection).
- Edits are validated by your eval set. A small eval set can be passed by luck, so use many items and keep a separate held-out set.

See `ROADMAP.md` for v2 (merging and compatibility scoring).

Copyright 2026 G. Vivek. Licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).

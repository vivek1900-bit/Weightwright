# Example: yes-no-only

A small, measurable behaviour to try the tool on: when a prompt says "answer yes or no only",
Qwen2.5-0.5B-Instruct often adds a long explanation after the answer.

## Files

| File | What it is |
|---|---|
| `spec.yaml` | Run spec for the training set. |
| `train.jsonl` | 24 `target` prompts ("Answer yes or no only: ...") and 20 `regression` prompts that must not get worse. |
| `heldout_spec.yaml` | Spec that points at the held-out set. Use it with `wright eval` only, never with `wright run`. |
| `heldout.jsonl` | 24 different `target` questions with three different phrasings. The editor must never see these. |

The `target` check is format only: the whole reply must be "yes" or "no" (optionally with a full stop).
It does not check whether the answer is true, and many of this model's answers are also factually wrong.

## Baseline measured on Qwen2.5-0.5B-Instruct (greedy decoding, `max_new_tokens: 300`)

| Set | Target pass rate |
|---|---|
| `train.jsonl` | 0.417 (10 of 24) |
| `heldout.jsonl` | 0.458 (11 of 24) |

The two sets agree, so the behaviour is consistent and there is room for an edit to show a gain.
The regression set should score close to 1.0 on the baseline; run it to see your own number.

## Run it (from the repository root)

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/small-model
wright validate examples/yes-no-only/spec.yaml
wright eval examples/yes-no-only/spec.yaml --show-failures
wright eval examples/yes-no-only/heldout_spec.yaml --show-failures

# after an accepted run, check the patch on the held-out set:
wright eval examples/yes-no-only/heldout_spec.yaml --patch runs/<run folder>/patch
```

## What this example does not show

- No edit has been tried on this behaviour yet. Whether changing a few weights can fix it is untested.
- A simple prompt change may fix it more cheaply. Compare against that before claiming anything.
- A gain on `train.jsonl` alone proves little. Check the held-out set and the regression score.

Before running `wright run`, set `editor.model` and `editor.base_url` in `spec.yaml` for your own
editor, or switch to `provider: scripted` and supply ops.

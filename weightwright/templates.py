EXAMPLE_SPEC = """\
# weightwright spec: one file drives one run. Start with a SMALL model (about 0.5B-1.5B)
# on a 6 GB GPU laptop, then move up once the pipeline works.
name: fix-repetition-demo

model:
  path: ./models/small-model        # local folder with *.safetensors + config/tokenizer
  dtype: bfloat16                   # bfloat16 | float16 | float32
  device: auto                      # auto | cpu | cuda

goal: |
  On short maths questions the model repeats the same text until it runs out of tokens.
  Make it stop repeating and finish its answer, without changing other behaviour.

eval:
  file: ./evals/example.jsonl       # lines tagged "target" must improve, "regression" must not drop
  backend: hf
  max_new_tokens: 256
  chat: true
  chat_template_kwargs: {}          # e.g. {enable_thinking: false} for Qwen-style templates
  target_tag: target
  regression_tag: regression
  min_gain: 0.10                    # target pass-rate must rise by at least 10 points
  max_regression_drop: 0.02         # regression pass-rate may drop by at most 2 points

editor:
  provider: openai_compat           # openai_compat | anthropic | scripted
  model: your-editor-model-name
  base_url: http://localhost:11434/v1   # any OpenAI-style server (Ollama, vLLM, a Kaggle tunnel URL)
  api_key_env: EDITOR_API_KEY       # name of the environment variable that holds the key
  max_turns: 40
  max_evals: 20

budget:
  max_touched_fraction: 0.0005      # at most 0.05% of all weights may change
  allow: ["model.layers.*.mlp.*", "model.layers.*.self_attn.*"]
  deny: ["*embed_tokens*", "*lm_head*", "*norm*"]

output:
  dir: ./runs

# Used only when editor.provider is "scripted" (no AI, replays these ops):
scripted_ops: []
"""

EXAMPLE_EVAL = """\
# One JSON object per line. tag: target | regression. Check types:
# contains, regex, exact, last_number, no_loop, finished, json_valid, or {"all": [..]}
{"prompt": "What is 17 + 25? Answer briefly.", "tag": "target", "check": {"all": [{"type": "finished"}, {"type": "no_loop"}, {"type": "last_number", "value": 42}]}}
{"prompt": "What is 9 * 8? Answer briefly.", "tag": "target", "check": {"all": [{"type": "finished"}, {"type": "no_loop"}, {"type": "last_number", "value": 72}]}}
{"prompt": "What is 100 - 37? Answer briefly.", "tag": "target", "check": {"all": [{"type": "finished"}, {"type": "no_loop"}, {"type": "last_number", "value": 63}]}}
{"prompt": "What is the capital of France? One word.", "tag": "regression", "check": {"type": "contains", "value": "paris"}}
{"prompt": "Reply with valid JSON only: an object with key \\"ok\\" set to true.", "tag": "regression", "check": {"type": "json_valid"}}
{"prompt": "Name the chemical symbol for water's two elements, hydrogen and oxygen.", "tag": "regression", "check": {"type": "contains", "any": ["h", "o"]}}
"""

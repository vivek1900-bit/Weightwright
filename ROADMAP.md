# Roadmap

## v1: weight editing (this release)
- [x] Lazy safetensors store, sparse exact patches, apply and revert
- [x] Edit ops, budget (fraction, allow, deny), eval gate
- [x] Editor loop with scripted, OpenAI-compatible and Anthropic providers
- [ ] First end-to-end run on a real small model; record results in `docs/`
- [ ] Reproduce a repetition-loop fix and compare against `weightspace`
- [ ] Resume an interrupted run from checkpoints
- [ ] Activation-based diagnostics for the editor (hidden-state measurements)

## v2: merging and compatibility score
Goal: pick two models, get a prediction of how well they will merge, then merge and test.

Design (from the literature, not yet validated here):
1. **Hard checks (pass or fail):** architecture, tensor names and shapes, tokenizer and vocabulary, shared base model.
2. **Soft signals:** hidden-state similarity on a small probe set, task-vector similarity.
3. **Optional slow mode:** a quick test merge scored on a few examples.
4. **Output:** three separate results (structural, lineage, behavioral), a recommended method, and stated uncertainty.

Evidence to respect:
- Published predictors reach only moderate correlation with real merge success (about 0.44 to 0.63 for Zhou et al. 2026; 0.70 for COMPASS on image models). Label the score "predicted compatibility".
- Parameter-conflict metrics did not predict collapse in Cao et al. (2026); hidden-state similarity did.
- A 2025 systematic study found plain task arithmetic the most reliable method on LLMs. Start there.
- Calibrate the score on real merge runs, never on hand-picked weights.

## v3: ideas
- Add a skill from another model (task vector or adapter) with the same gate.
- Guided search: describe a capability, let the tool propose edits or merges, test, accept or reject.

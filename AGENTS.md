# Repository rules

- Install and test in an isolated environment: `pip install -e '.[test,bridge]'`; run `pytest -q` and `python -m build`.
- Keep the vLLM endpoint plugin separate from the optional HTTP bridge. Never call bridge tests proof of vLLM plugin deployment.
- Count classification transport tokens, generated tokens, input tokens and engine requests separately.
- Conditional candidate probabilities are not calibrated correctness probabilities. Schema validity is not semantic correctness.
- Do not restart existing model services during ordinary development. Preserve existing experiments and model revisions.
- English README is primary; keep a Chinese edition. No credentials, model weights or caches in Git.

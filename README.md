# vllm-jev-decison

**Classification-only typed decisions for compatible vLLM language models.**

[中文](README.zh-CN.md) · [Installation & usage](docs/GUIDE.md) · [Examples](examples) · [Validation](results/README.md)

Supply text, a question and a finite JSON Schema. The plugin scores candidate
labels, selects typed values, and constructs the result in code. **No free-form
answer generation, mixed inference, or generative fallback is available.**
Unsupported schemas are rejected before model inference.

![Classification architecture](assets/en/architecture.svg)

This is an independent Jev-like interface, not a reproduction of Jev's model or
a promise of equivalent accuracy or speed. It requires no model-specific head,
weight training or vLLM source patch.

## Install

Target API: **vLLM 0.29.0**, Python 3.11+. Install in the server environment:

```bash
git clone https://github.com/siliconkernel/vllm-jev-decison.git
cd vllm-jev-decison
pip install '.[vllm]'
vllm-jev-decison doctor --model YOUR_MODEL
export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}jev-decison"
vllm serve YOUR_MODEL --logprobs-mode raw_logprobs
```

If that vLLM version is already installed, use `pip install .`. The plugin needs
`max_logprobs` at 16 or higher; vLLM's default of 20 already satisfies this, so
raise it only if your deployment lowered it. `raw_logprobs` is also the vLLM
default and is passed explicitly so the requirement survives a default change.
`doctor --model` additionally checks that the model encodes the candidate labels
A-P as distinct single tokens, which the backend requires; it exits non-zero when
they do not, so a mismatch surfaces before deployment rather than as a disabled
endpoint. Preserve other required plugins in the allowlist. Configure `VLLM_API_KEY` or vLLM `--api-key`
for authentication. The package is installed from this repository, not a claimed
PyPI release. [Full installation and troubleshooting guide](docs/GUIDE.md).

## Use

```bash
curl http://localhost:8000/plugins/jev-decison/infer \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

Add `-H "Authorization: Bearer $VLLM_API_KEY"` when authentication is configured.
The [example request](examples/request.json) classifies a customer message into
an enum category and a boolean urgency flag:

```json
{
  "state": "My card was charged twice. Please refund me urgently.",
  "question": "Categorize this request and identify urgency.",
  "schema": {
    "type": "object",
    "properties": {
      "category": {"enum": ["billing", "technical", "other"]},
      "urgent": {"type": "boolean"}
    },
    "required": ["category", "urgent"],
    "additionalProperties": false
  },
  "mode": "classify",
  "min_confidence": 0.8
}
```

`mode` is optional and only accepts `classify`. `auto`, `generate`, output-token
budgets and nonfinite fields are rejected before any model call. Nonfinite schemas
return HTTP 422; a rejected `mode` value is a request-field error, returned as
HTTP 400 by the native plugin and HTTP 422 by the bridge's plain FastAPI stack.
No fallback is run.

The response contains `accepted`, `value`, per-field `decisions`, and `usage`.
If any classified field misses the threshold, `accepted=false` and `value=null`.
Selected values in `decisions` are diagnostic proposals, not accepted results.
`GET /plugins/jev-decison/capabilities` reports the active backend and limits.
The [Python client](examples/client.py) uses `JEV_DECISON_URL` and optional
`JEV_DECISON_API_KEY`.

## Supported output schemas

| Schema | Behavior |
| --- | --- |
| Enum with at most 16 values | Score candidate labels and select a value |
| Boolean | Classify false / true |
| Bounded integer interval, at most 16 values | Classify the numeric value |
| Constant or null | Construct directly, without a model call |
| Closed, fully required object of supported fields | Classify fields independently; supports nesting |
| Free text, open numbers, optional fields, arbitrary arrays or unsupported joint constraints | Reject before inference |

Explicit finite `enum`/`const` values may themselves be objects or arrays; the
plugin selects the entire declared value. It does not generate their contents.
Every completed result is validated against the root schema.

Limits: 32 fields, 16 candidates per field, 64,000 input characters, 32 KB schema
and 20 JSON nesting levels. That depth counts every JSON level, not just objects:
each nested object costs two levels, so roughly 9 levels of nested objects fit.
Repeated `enum` values are collapsed, since two labels sharing one value would
split its probability between them. Draft 2020-12 is used; references are
rejected in v0.1.
`format` is an annotation, not an enforced format checker.

## How classification works

Candidate values map to tokenizer-checked single-token labels A–P. The engine
computes next-token logprobs, the adapter reads the requested candidate scores,
and argmax selects a value. The model never serializes the output object's keys
or selected values; Python builds them from the declared schema and candidate table.

The portable backend still uses the ordinary vLLM sampler with `max_tokens=1`.
Its one output token is counted as a **classification transport token**, even
though its text is ignored. `generated_tokens` is always zero for this API;
that does not mean zero output tokens or a model with a new classification head.
Multiple fields create multiple engine requests, with at most eight active per
API process by default. `JEV_DECISON_CONCURRENCY` (1-256) and `JEV_DECISON_TIMEOUT`
(1-3600 seconds) change that; the gate is per API process, so one wide request can
delay others. `capabilities` reports the values actually in effect. vLLM may batch them. This is not one forward for all questions,
not a sampler-bypass patch, and not retained-session KV fusion.

![Confidence and validity](assets/en/confidence.svg)

Conditional candidate probabilities are not calibrated correctness estimates.
`candidate_mass` separately reports raw vocabulary probability assigned to the
labels. Label order, prompts, model training and distribution shift affect results.
Constants have no model confidence score. Schema-valid does not mean task-correct.

Candidate order is measured, not just disclaimed. Across 9 cases and 46 orderings
on Qwen3-4B, 8 cases held the same value under every permutation, and one flipped
to a wrong value **at confidence 0.997**. Selected positions were evenly spread
(16/15/15), so this is not a simple first-label preference: reordering only moves
decisions the model was already close on. **`min_confidence` does not defend
against it** — the flipped answer clears any usable threshold. If a decision
matters, test your own labels under more than one order.
[Measurement](results/README.md), [script](benchmarks/label_order_bias.py).

## Integration and evidence

![Deployment paths](assets/en/deployment.svg)

The native plugin uses vLLM's official
[EndpointPlugin interface](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/)
and existing EngineClient. It does not load another model. The optional bridge
provides the same classification-only API over an existing vLLM server:

```bash
pip install '.[bridge]'
vllm-jev-decison bridge --upstream http://127.0.0.1:8000 --model /model --port 18186
```

The bridge backend is explicitly `http_bridge`, not native plugin deployment.
It requires `/tokenize` and requested raw token logprobs, not structured-generation
support. See the [guide](docs/GUIDE.md) for separate bridge/upstream authentication.
Both paths are verified against a live model, and return identical decisions and
token accounting for the same cases; the bridge runs were served by an upstream
with no plugin loaded.

Intended models are vLLM-supported autoregressive text models with compatible
single-token labels and raw requested logprobs. Instruct models are the primary
target; pooling-only models, multimodal input, LoRA routing and arbitrary tokenizer
combinations are not validated. “Compatible models” does not mean “any model.”

Local tests cover finite planning, rejected generation modes, authentication,
abstention, accounting, cancellation and a fake-engine adapter contract. **Native
GPU plugin startup is verified**: the plugin loads into a real vLLM API server on
NVIDIA GB10, serves both routes, reconciles its token accounting against the engine's
own `/metrics` counters, and releases engine work when a client disconnects. Those
runs were recorded on a vLLM build newer than 0.29.0 that carries the same
`EndpointPlugin` interface. Historical bridge smoke records include the earlier mixed
prototype, which is no longer supported; see [validation records](results/README.md).
No universal speedup, production readiness or semantic-correctness guarantee is claimed.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,bridge]'
pytest -q
python -m build
python docs/render_diagrams.py
```

[Native smoke](benchmarks/smoke_native.py) runs against a loaded plugin and refuses
to record evidence for the bridge; [bridge smoke](benchmarks/smoke_http.py) covers the
other path; [label-order bias](benchmarks/label_order_bias.py) measures how much
candidate order moves a decision. All write to a new output directory. Preserve failed records.
MIT licensed; vLLM retains its own license.

# vllm-decision

**Turn a compatible vLLM-served language model into a typed decision API.**

[中文](README.zh-CN.md)

[Installation & usage guide](docs/GUIDE.md) · [Examples](examples) · [Validation records](results/README.md)

Supply input text, a question and a JSON Schema. Finite fields use next-token
candidate scoring; free-form fields use schema-constrained generation. Python
constructs and validates the final value. No model-specific classification head,
weight training, or vLLM source patch is required.

This is an independent implementation of a Jev-like **interface**, not a Jev
model, SDK, implementation claim, or guarantee of equivalent quality or speed.
It does not make every model a reliable classifier.

![Architecture / 架构](assets/en/architecture.svg)

## Install and enable the plugin

Target API: **vLLM 0.29.0**. Install inside the same environment as the API server:

```bash
git clone https://github.com/siliconkernel/vllm-decision.git
cd vllm-decision
pip install '.[vllm]'
vllm-decision doctor

# Include any other plugins your deployment needs in this allowlist.
export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}decision"
vllm serve YOUR_MODEL \
  --logprobs-mode raw_logprobs \
  --max-logprobs 16
```

In an existing vLLM 0.29.0 environment, `pip install .` is sufficient. Set
`VLLM_API_KEY` or use vLLM's `--api-key` for authenticated access. Plugin routes
independently enforce the configured keys, including their non-`/v1` prefix.
The plugin registers through the official
[EndpointPlugin interface](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/).
It reuses the running `EngineClient`; it does not load another model.

## Call the API

```bash
curl http://localhost:8000/plugins/decision/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "My card was charged twice. Please refund me urgently.",
    "question": "Categorize this customer request and identify urgency.",
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
  }'
```

When authentication is enabled, add `-H "Authorization: Bearer $VLLM_API_KEY"`.
See [examples/request.json](examples/request.json) and
[examples/client.py](examples/client.py) for reusable examples.

The response includes:

- `accepted` and `value`: the validated result, or `null` if any classified field
  falls below the requested confidence threshold.
- `decisions`: per-field selected value, candidate probabilities, raw logprobs,
  candidate mass and mode. Rejected field values are diagnostic proposals, not
  accepted results. Generated fields and deterministic constants have no confidence estimate.
- `usage`: input tokens, classification transport tokens, generated tokens and
  engine request count, summed across fields.
- `backend`: `vllm_endpoint_plugin` or explicitly `http_bridge`.

`GET /plugins/decision/capabilities` reports the active backend and limits.
Schemas and generated values are validated before a successful result is returned.
A truncated generation is an error, never a dispatchable partial value.

## Three execution modes

| Mode | Finite fields | Other fields |
| --- | --- | --- |
| `classify` | Candidate scoring | Reject request |
| `auto` | Candidate scoring | JSON Schema constrained generation |
| `generate` | Constrained generation | Constrained generation |

Finite domains include enums of at most 16 values, booleans, null, constants,
and integer intervals of at most 16 values. Simple closed objects with every
property required are decomposed into independent fields. Nested objects work.
Optional properties, arrays, conditional schemas and other joint constraints use
whole-subtree generation in `auto` mode. The full root schema is validated again.

Limits: 32 fields, 16 candidates per field, 64,000 input characters, 32 KB schema,
20 nesting levels and 4,096 generated tokens per generated field. JSON Schema
Draft 2020-12 is used; references are rejected in v0.1. `format` is an annotation,
not an enforced format checker. Structured-generation support also depends on
vLLM's grammar backend. Unknown or unsupported grammar features can fail explicitly.

## What classification actually does

A prompt maps candidate values onto tokenizer-checked, single-token labels A–P.
The engine computes next-token log probabilities once. The adapter reads the
requested label scores, selects argmax, and constructs the original typed value.
No generated class-name text or JSON punctuation needs parsing.

This portable backend still uses vLLM's **ordinary sampler** with `max_tokens=1`.
That output token is counted as a classification transport token, even though its
text is ignored. It is not the earlier sampler-bypass / retained-KV patch from
ClassWeave. Multiple fields create multiple engine requests, with at most eight
active field requests per API process. vLLM can batch them; this is **not** one
forward pass answering every question. Normal prefix caching can apply when
configured, but retained-session KV fusion is not implemented here.

Candidate probabilities are normalized **only over the listed labels**. A 0.95
score does not mean 95% task correctness. `candidate_mass` reports how much raw
vocabulary probability the labels captured; high conditional probability can
coexist with tiny mass. Label wording/order, tokenizer, model training and prompt
shift affect results. The threshold applies only to classified fields; it neither
calibrates nor rejects uncertain free-form generation.

![Confidence / 概率与结构边界](assets/en/confidence.svg)

## Model compatibility and validation status

The intended scope is vLLM-supported autoregressive text-generation models whose
tokenizers expose compatible single-token labels and whose engines return raw
requested logprobs. Instruct models are the primary target. Base models can use a
plain-text prompt when no chat template exists, with no quality guarantee.
Pooling-only models, multimodal inputs, LoRA routing, speculative decoding
interactions and arbitrary tokenizer/model combinations are not validated.

The native adapter checks tokenizer labels and `raw_logprobs` at startup. Its
interface is checked against vLLM 0.29.0 source. Local tests cover planning, API
authentication, abstention, accounting, cancellation, and a **fake-engine adapter
contract**. This does not substitute for launching the plugin with a real GPU
model. Live-model bridge results, when present, are recorded separately in
[results](results/README.md); they are not native plugin deployment evidence.

Fewer output tokens can reduce decode work, but repeated prefill, field fan-out,
scheduling and fallback generation may erase that benefit. No universal speedup,
calibrated confidence, production readiness or semantic-correctness claim is made.

![Deployment / 部署方式](assets/en/deployment.svg)

## Optional bridge for an existing engine

For testing without restarting an existing vLLM server:

```bash
pip install '.[bridge]'
vllm-decision bridge --upstream http://127.0.0.1:8000 --model /model --port 18186
```

Use the same `/plugins/decision/infer` route on port 18186. This is an HTTP bridge,
**not** the native plugin. The upstream needs `/tokenize`, requested token
logprobs and structured outputs, and must use `raw_logprobs` (the bridge cannot
verify that startup setting). Configure `DECISION_UPSTREAM_API_KEY` and
`DECISION_API_KEY` independently. The bridge connects directly, ignoring HTTP
proxy environment variables. It binds to loopback by default.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test,bridge]'
pytest -q
python -m build
```

Run the bounded live-model bridge smoke into a new directory:

```bash
python benchmarks/smoke_http.py --url http://127.0.0.1:8000 \
  --model /model --out results/my-new-run
```

MIT licensed. vLLM retains its own license. Upstream API references:
[plugin contract](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/plugins/endpoint_plugins/interface.py),
[EngineClient](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/engine/protocol.py),
[SamplingParams](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/sampling_params.py).

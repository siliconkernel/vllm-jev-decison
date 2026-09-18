# Validation records

## Native endpoint plugin

These runs load the plugin **inside a real vLLM API server process** through the
`vllm.endpoint_plugins` entry point. The server log records `Loaded endpoint plugin
jev-decison` and both routes, and `capabilities` reports backend
`vllm_endpoint_plugin`. The smoke script refuses to write a native record if the
backend reports `http_bridge`.

Environment: NVIDIA GB10 (aarch64, unified memory), container image
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, vLLM
`0.1.dev20051+g487ecf187` — a source build newer than 0.29.0 that carries the same
`EndpointPlugin` protocol, transformers 5.15.1. Served with
`--logprobs-mode raw_logprobs --max-logprobs 16 --enforce-eager`, on a shared
machine alongside unrelated services that were not restarted.

| Run | Model | Outcome | Meaning |
| --- | --- | --- | --- |
| `native-qwen3-0.6b-20260918-a` | Qwen3-0.6B | 5/8 | Mechanism cases pass; all four semantic cases fail. |
| `native-qwen3-0.6b-20260918-b` | Qwen3-0.6B | 5/8 | Same, re-run under the final case definitions. |
| `native-qwen3-4b-20260918-a` | Qwen3-4B-Instruct-2507 | 7/8 | One semantic failure traced to an ambiguous question, not to the plugin. |
| `native-qwen3-4b-20260918-b` | Qwen3-4B-Instruct-2507 | 8/8 | Same cases with the ambiguous field given a `description`. |

The four mechanism cases — one constant-bearing schema and four rejections — pass on
both models. The semantic cases depend entirely on the served model: a 0.6B model
fails all of them while a 4B instruct model passes. **This separation is the point.
It shows the mechanism works independently of whether the answers are right, and it
is not evidence that any particular model classifies well.**

### Token accounting reconciles against the engine

Every run compares its own reported usage with vLLM's `/metrics` counters over the
same interval. All four runs reconcile exactly:

| Run | Reported input / engine prompt | Reported classification / engine generation | `generated_tokens` |
| --- | --- | --- | ---: |
| `0.6b-a` | 1501 / 1501 | 7 / 7 | 0 |
| `0.6b-b` | 1554 / 1554 | 7 / 7 | 0 |
| `4b-a` | 1473 / 1473 | 7 / 7 | 0 |
| `4b-b` | 1526 / 1526 | 7 / 7 | 0 |

The engine really does emit one output token per nonconstant field. Reporting
`generated_tokens: 0` hides nothing: those tokens are counted as classification
transport tokens, and the engine's own counter agrees with that number.

### Separately confirmed on the same deployment

- **Scores are read from requested logprobs, not from the sampled token.** For the
  prompt `The capital of France is` with `logprob_token_ids=[32, 33]` (`A`, `B`),
  vLLM sampled token `12095` (` Paris`) and still returned logprobs for 32 and 33.
  The sampled token's identity is irrelevant to selection.
- **Disconnect cancels engine work.** A 32-field request completed with 32 engine
  successes. The same request, abandoned by the client after 0.3 s, produced only 12
  engine successes; running requests returned to zero and the server stayed healthy.
- **The concurrency gate does not drop fields.** A 12-field schema, above the
  eight-request gate, returned all 12 decisions with 12 engine requests.
- **Constants never reach the engine**, and schema `description` annotations do
  reach the model: adding one to an ambiguous boolean field changed its answer from
  wrong to right, which is how run `4b-a` became run `4b-b`.

### Two defects this exercise found

- `apply_chat_template(tokenize=True)` returns a `BatchEncoding` under transformers 5,
  not a token list. Every native request failed with HTTP 502 until `return_dict=False`
  was passed explicitly. A regression test now covers it.
- A rejected `mode` value returns **HTTP 400** under the native server, not the 422
  the documentation claimed; vLLM maps request validation errors to 400 while the
  bridge's plain FastAPI stack returns 422. The rejection still happens before any
  model call. The documentation was corrected rather than the status code.

### What these runs do not establish

Single GPU, single node, two small Qwen3 models, eight cases, no load. Nothing here
speaks to throughput, latency, calibration, multi-model compatibility, tensor
parallelism, long context or production readiness. The vLLM build is newer than the
0.29.0 stated as the target API; the `EndpointPlugin` protocol matched exactly, but
0.29.0 itself was not the binary under test. Semantic pass rates are development-set
observations on cases chosen by the author, not a held-out evaluation.

## HTTP bridge (historical)

**Historical scope:** runs a–c below predate the classification-only change. Their
mixed/generative paths have been removed from the current API. Records are preserved
byte-for-byte for provenance, not as a description of current capabilities.
The current smoke script checks classification and rejects nonfinite/generative requests.

These are bounded development smoke runs against the already-running DeepSeek
`/model` endpoint, reached through an ephemeral SSH tunnel. No engine restart,
new weights, or vLLM plugin loading occurred. Backend: **HTTP bridge**.

| Run | Outcome | Meaning |
| --- | --- | --- |
| `deepseek-http-smoke-20260918-a` | Failed before model calls | Environment SOCKS proxy initialization required an unavailable optional library; bridge now makes direct connections. |
| `deepseek-http-smoke-20260918-b` | HTTP/schema success 4/4; exact classification 0/2 | Initial field prompt produced incorrect urgency decisions. The generated explanation was also unhelpful despite being schema-valid. |
| `deepseek-http-smoke-20260918-c` | HTTP/schema success 4/4; exact classification 2/2 | Revised per-field task prompt with explicit yes/no mapping. This is development-set tuning, not held-out quality evaluation. |

The latest run used:

| Case | Input tokens | Classification tokens | Generated tokens | Engine requests |
| --- | ---: | ---: | ---: | ---: |
| Boolean | 115 | 1 | 0 | 1 |
| Two classified fields | 383 | 2 | 0 | 2 |
| Classification + free text | 382 | 1 | 22 | 2 |
| Whole-schema generation | 127 | 0 | 8 | 1 |

The first two cases have exact semantic expectations. The last two check accepted,
schema-valid responses; they are not general semantic quality oracles. All input
and output records are preserved. Token usage excludes tokenizer HTTP requests
and infrastructure overhead; wall times include preparation and shared-server
queue effects. This is not a native plugin benchmark or a speed comparison.

Local unit and adapter-contract tests use fake model outputs and cover planning,
authentication, timeout/disconnect cancellation, token accounting, threshold
abstention and malformed/truncated results. Native GPU plugin startup, multi-model
compatibility, calibration and production load tests remain unverified.

# Validation records

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

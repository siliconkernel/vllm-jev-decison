"""Measure how input cost grows with field count; each field carries the whole schema."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx

COUNTS = [1, 2, 4, 8, 16, 32]
SENTENCE = ('A customer reports that the checkout page fails intermittently. '
            'They were charged once but received no confirmation email. ')


def metrics(client):
    totals = {}
    for line in client.get('/metrics').text.splitlines():
        for name in ('prefix_cache_queries_total', 'prefix_cache_hits_total'):
            if line.startswith('vllm:' + name + '{'):
                totals[name] = totals.get(name, 0.0) + float(line.rsplit(' ', 1)[1])
    return totals


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    headers = {'Authorization': 'Bearer ' + args.api_key} if args.api_key else {}
    rows = []
    with httpx.Client(base_url=args.url.rstrip('/'), headers=headers, timeout=300, trust_env=False) as client:
        backend = client.get('/plugins/jev-decison/capabilities').raise_for_status().json()['backend']
        for count in COUNTS:
            # A unique prefix per case, so the cache rate reflects sharing between
            # fields of one request rather than a replay of the previous case.
            state = f'Ticket {uuid4().hex}. ' + SENTENCE * args.repeat
            properties = {f'f{i}': {'type': 'boolean', 'description': f'Flag number {i} about the report.'}
                          for i in range(count)}
            body = {'state': state, 'question': 'Answer each flag about this report.',
                    'schema': {'type': 'object', 'properties': properties, 'required': list(properties),
                               'additionalProperties': False}, 'min_confidence': 0}
            before = metrics(client)
            usage = client.post('/plugins/jev-decison/infer', json=body).raise_for_status().json()['usage']
            after = metrics(client)
            queries = after.get('prefix_cache_queries_total', 0) - before.get('prefix_cache_queries_total', 0)
            hits = after.get('prefix_cache_hits_total', 0) - before.get('prefix_cache_hits_total', 0)
            rows.append({'fields': count, 'state_characters': len(state), 'input_tokens': usage['input_tokens'],
                         'tokens_per_field': usage['input_tokens'] // count,
                         'engine_requests': usage['engine_requests'],
                         'prefix_cache_hit_rate': round(hits / queries, 3) if queries else None})
            print(f"fields={count:>3} input_tokens={usage['input_tokens']:>6} "
                  f"per_field={usage['input_tokens'] // count:>5} "
                  f"cache_hit={rows[-1]['prefix_cache_hit_rate']}", flush=True)

    first, last = rows[0], rows[-1]
    summary = {'backend': backend, 'model': args.model, 'state_characters': first['state_characters'],
               'field_growth': last['fields'] // first['fields'],
               'token_growth': round(last['input_tokens'] / first['input_tokens'], 1),
               'tokens_per_field_growth': round(last['tokens_per_field'] / first['tokens_per_field'], 1),
               'cause': 'Every nonroot field repeats the whole output schema for context, and that schema '
                        'grows with the field count, so total input scales roughly quadratically.',
               'prefix_cache_note': 'Each case uses a unique state, so the hit rate measures sharing '
                                    'between the fields of one request, not a replay of an earlier request.',
               'scope': f'One model, one state length ({first["state_characters"]} characters). Absolute numbers depend on '
                        'the tokenizer, the state and the schema; the shape of the growth does not.'}
    print('\n' + json.dumps(summary, indent=2))
    (args.out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    files = sorted(p for p in args.out.iterdir() if p.name != 'SHA256SUMS')
    (args.out / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files))
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--api-key', default=None)
    p.add_argument('--model', default='/model')
    p.add_argument('--repeat', type=int, default=8, help='How many times to repeat the sample sentence')
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

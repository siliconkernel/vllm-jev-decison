"""Live-model smoke through the HTTP bridge; does not claim plugin-loader validation."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
from vllm_jev_decison.cli import bridge_app

CASES = [
    {'name': 'boolean', 'request': {'state': 'The customer explicitly asks for an urgent response.',
        'question': 'Does the customer explicitly request urgency?', 'schema': {'type': 'boolean'}, 'mode': 'classify'}, 'expected': True},
    {'name': 'typed_fields', 'request': {'state': 'My credit card was charged twice. Please refund the duplicate charge urgently.',
        'question': 'Categorize the customer request and whether it is urgent.', 'schema': {'type': 'object',
        'properties': {'category': {'enum': ['billing', 'technical', 'other']}, 'urgent': {'type': 'boolean'}},
        'required': ['category', 'urgent'], 'additionalProperties': False}, 'mode': 'classify'},
        'expected': {'category': 'billing', 'urgent': True}},
    {'name': 'hybrid', 'request': {'state': 'The customer was charged twice and requests a refund.',
        'question': 'Categorize the request and explain it briefly.', 'schema': {'type': 'object',
        'properties': {'category': {'enum': ['billing', 'technical', 'other']}, 'explanation': {'type': 'string', 'maxLength': 180}},
        'required': ['category', 'explanation'], 'additionalProperties': False}, 'mode': 'auto'}},
    {'name': 'generation', 'request': {'state': 'The user says hello.', 'question': 'Respond with a short greeting.',
        'schema': {'type': 'object', 'properties': {'reply': {'type': 'string'}}, 'required': ['reply'], 'additionalProperties': False}, 'mode': 'generate'}},
]


async def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    app = bridge_app(args.url, args.model)
    rows = []
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=15, trust_env=False) as client:
            version = await client.get('/version')
            models = await client.get('/v1/models')
        (args.out / 'environment.json').write_text(json.dumps({'version': version.json(), 'models': models.json(),
            'validation_scope': 'HTTP bridge and live model; native plugin loader was not exercised'}, indent=2))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://smoke', timeout=200) as client:
            for case in CASES:
                start = time.perf_counter()
                response = await client.post('/plugins/jev-decison/infer', json=case['request'])
                row = {'case': case, 'status': response.status_code, 'response': response.json(), 'wall_seconds': time.perf_counter() - start}
                row['passed'] = response.status_code == 200 and row['response'].get('accepted', False)
                if 'expected' in case:
                    row['passed'] &= row['response'].get('value') == case['expected']
                rows.append(row)
                with (args.out / 'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(case['name'], row['status'], 'pass', row['passed'], row['response'].get('usage'), flush=True)
    finally:
        await app.state.decision_service.backend.close()
    (args.out / 'summary.json').write_text(json.dumps({'passed': sum(r['passed'] for r in rows), 'total': len(rows),
        'scope': 'Classify, typed fields, hybrid, and generation HTTP-bridge smoke. Only first two cases have exact semantic oracles. No speed, calibration, universal-model or native-plugin deployment claim.'}, indent=2))
    files = sorted(args.out.iterdir())
    (args.out / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files))
    return all(row['passed'] for row in rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--model', default='/model')
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(0 if asyncio.run(run(p.parse_args())) else 1)

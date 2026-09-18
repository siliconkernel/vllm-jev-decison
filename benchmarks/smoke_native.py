"""Live smoke against a loaded vLLM endpoint plugin; refuses to run on the HTTP bridge."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx

CASES = [
    {'name': 'boolean', 'request': {'state': 'The customer explicitly asks for an urgent response.',
        'question': 'Does the customer explicitly request urgency?', 'schema': {'type': 'boolean'}}, 'expected': True},
    {'name': 'typed_fields', 'request': {'state': 'My credit card was charged twice. Please refund the duplicate charge urgently.',
        'question': 'Categorize the customer request and whether it is urgent.', 'schema': {'type': 'object',
        'properties': {'category': {'enum': ['billing', 'technical', 'other']}, 'urgent': {'type': 'boolean'}},
        'required': ['category', 'urgent'], 'additionalProperties': False}},
        'expected': {'category': 'billing', 'urgent': True}},
    {'name': 'nested_and_integer', 'request': {'state': 'The login service returned HTTP 500 for three hours. Customers cannot sign in. It is still broken.',
        'question': 'Assess this incident. Severity 5 is a total outage, 1 is cosmetic.', 'schema': {'type': 'object',
        'properties': {'triage': {'type': 'object', 'properties': {'area': {'enum': ['billing', 'technical', 'other']},
        'severity': {'type': 'integer', 'minimum': 1, 'maximum': 5}}, 'required': ['area', 'severity'], 'additionalProperties': False},
        'resolved': {'type': 'boolean', 'description': 'True only if the incident is already fixed.'}},
        'required': ['triage', 'resolved'], 'additionalProperties': False}},
        'expected': {'triage': {'area': 'technical', 'severity': 5}, 'resolved': False}},
    {'name': 'constant_skips_engine', 'request': {'state': 'Any text at all.', 'question': 'Fill the record.',
        'schema': {'type': 'object', 'properties': {'kind': {'const': 'report'}, 'urgent': {'type': 'boolean'}},
        'required': ['kind', 'urgent'], 'additionalProperties': False}}, 'constant_paths': ['/kind']},
    {'name': 'reject_free_text', 'request': {'state': 'hello', 'schema': {'type': 'string'}}, 'reject': True},
    {'name': 'reject_open_object', 'request': {'state': 'hello', 'schema': {'type': 'object',
        'properties': {'a': {'type': 'boolean'}}, 'required': ['a']}}, 'reject': True},
    {'name': 'reject_schema_ref', 'request': {'state': 'hello', 'schema': {'$ref': '#/x'}}, 'reject': True},
    {'name': 'reject_generation_mode', 'request': {'state': 'hello', 'schema': {'type': 'boolean'}, 'mode': 'generate'}, 'reject': True},
]


def metric(text, name):
    """Sum a vLLM counter across engine and model labels."""
    prefix = 'vllm:' + name + '{'
    return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines() if line.startswith(prefix))


def judge(case, status, body):
    """Rejections must not reach the model; accepted runs must be schema valid and complete."""
    if case.get('reject'):
        return status in (400, 422)
    if status != 200 or not body.get('accepted'):
        return False
    constants = {d['path'] for d in body['decisions'] if d['mode'] == 'constant'}
    if constants != set(case.get('constant_paths', [])):
        return False
    return 'expected' not in case or body['value'] == case['expected']


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    headers = {'Authorization': 'Bearer ' + args.api_key} if args.api_key else {}
    rows = []
    with httpx.Client(base_url=args.url.rstrip('/'), headers=headers, timeout=200, trust_env=False) as client:
        capabilities = client.get('/plugins/jev-decison/capabilities').raise_for_status().json()
        if capabilities['backend'] != 'vllm_endpoint_plugin':
            raise SystemExit(f"Refusing to record native evidence for backend {capabilities['backend']}")
        version = client.get('/version').json()
        before = client.get('/metrics').text
        for case in CASES:
            start = time.perf_counter()
            response = client.post('/plugins/jev-decison/infer', json=case['request'])
            body = response.json()
            row = {'case': case, 'status': response.status_code, 'response': body,
                   'wall_seconds': time.perf_counter() - start}
            row['passed'] = judge(case, response.status_code, body)
            rows.append(row)
            print(case['name'], row['status'], 'pass', row['passed'], body.get('usage'), flush=True)
        after = client.get('/metrics').text
    reported = {key: sum(r['response'].get('usage', {}).get(key, 0) for r in rows)
                for key in ('input_tokens', 'classification_tokens', 'generated_tokens', 'engine_requests')}
    engine = {'prompt_tokens': metric(after, 'prompt_tokens_total') - metric(before, 'prompt_tokens_total'),
              'generation_tokens': metric(after, 'generation_tokens_total') - metric(before, 'generation_tokens_total')}
    accounting = {'reported': reported, 'engine_metrics': engine,
                  'input_tokens_match': reported['input_tokens'] == engine['prompt_tokens'],
                  'classification_tokens_match': reported['classification_tokens'] == engine['generation_tokens'],
                  'note': 'Engine generation tokens must equal reported classification transport tokens; generated_tokens stays zero.'}
    print('accounting', json.dumps(accounting), flush=True)
    (args.out / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (args.out / 'environment.json').write_text(json.dumps({'version': version, 'capabilities': capabilities,
        'model': args.model, 'validation_scope': 'Native vLLM endpoint plugin loaded in the server process; no HTTP bridge involved'}, indent=2))
    (args.out / 'accounting.json').write_text(json.dumps(accounting, indent=2))
    passed = all(row['passed'] for row in rows) and accounting['input_tokens_match'] and accounting['classification_tokens_match']
    (args.out / 'summary.json').write_text(json.dumps({'passed': sum(r['passed'] for r in rows), 'total': len(rows),
        'accounting_reconciled': accounting['input_tokens_match'] and accounting['classification_tokens_match'],
        'scope': 'Native endpoint-plugin classification, rejection and accounting smoke. Semantic cases depend on the served model; no speed, calibration or universal-model claim.'}, indent=2))
    files = sorted(p for p in args.out.iterdir() if p.name != 'SHA256SUMS')
    (args.out / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--api-key', default=None)
    p.add_argument('--model', default='/model')
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

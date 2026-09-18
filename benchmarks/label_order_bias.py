"""Measure how much candidate order changes a decision; order is a documented risk."""
import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random

import httpx

CASES = [
    {'state': 'My credit card was charged twice for the same order.',
     'question': 'Which team should handle this?', 'values': ['billing', 'technical', 'other'], 'expected': 'billing'},
    {'state': 'The login page returns HTTP 500 and nobody can sign in.',
     'question': 'Which team should handle this?', 'values': ['billing', 'technical', 'other'], 'expected': 'technical'},
    {'state': 'Just wanted to say hello and ask about the weather.',
     'question': 'Which team should handle this?', 'values': ['billing', 'technical', 'other'], 'expected': 'other'},
    {'state': 'This product is wonderful and the support team was very kind.',
     'question': 'What is the sentiment of this message?', 'values': ['positive', 'negative', 'neutral'], 'expected': 'positive'},
    {'state': 'Terrible experience. It broke on the first day and nobody replied.',
     'question': 'What is the sentiment of this message?', 'values': ['positive', 'negative', 'neutral'], 'expected': 'negative'},
    {'state': 'The package was delivered on Tuesday.',
     'question': 'What is the sentiment of this message?', 'values': ['positive', 'negative', 'neutral'], 'expected': 'neutral'},
    {'state': '这个功能完全不能用，我要求退款。',
     'question': 'Which language is this message written in?', 'values': ['english', 'chinese', 'japanese'], 'expected': 'chinese'},
    {'state': 'Please refund me immediately, this is urgent.',
     'question': 'Is the customer asking for something urgently?', 'values': ['yes', 'no'], 'expected': 'yes'},
    {'state': 'No rush at all, whenever you have time next month.',
     'question': 'Is the customer asking for something urgently?', 'values': ['yes', 'no'], 'expected': 'no'},
]
MAX_ORDERS = 6


def orders(values, rng):
    """All permutations when cheap, otherwise a fixed-seed sample of them."""
    every = list(permutations(values))
    return every if len(every) <= MAX_ORDERS else rng.sample(every, MAX_ORDERS)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(0)
    headers = {'Authorization': 'Bearer ' + args.api_key} if args.api_key else {}
    rows = []
    with httpx.Client(base_url=args.url.rstrip('/'), headers=headers, timeout=200, trust_env=False) as client:
        backend = client.get('/plugins/jev-decison/capabilities').raise_for_status().json()['backend']
        for index, case in enumerate(CASES):
            for order in orders(case['values'], rng):
                body = {'state': case['state'], 'question': case['question'],
                        'schema': {'enum': list(order)}, 'mode': 'classify'}
                decision = client.post('/plugins/jev-decison/infer', json=body).raise_for_status().json()['decisions'][0]
                rows.append({'case': index, 'order': list(order), 'value': decision['value'],
                             'position': order.index(decision['value']), 'confidence': decision['confidence'],
                             'expected': case['expected'], 'correct': decision['value'] == case['expected']})
                print(f"case={index} order={'/'.join(order)} -> {decision['value']} "
                      f"(pos {order.index(decision['value'])}, conf {decision['confidence']:.3f})", flush=True)

    per_case = []
    for index, case in enumerate(CASES):
        picks = [r for r in rows if r['case'] == index]
        chosen = {r['value'] for r in picks}
        confidences = [r['confidence'] for r in picks]
        per_case.append({'case': index, 'question': case['question'], 'expected': case['expected'],
                         'stable': len(chosen) == 1, 'values_chosen': sorted(chosen),
                         'correct_orders': sum(r['correct'] for r in picks), 'orders': len(picks),
                         'confidence_min': min(confidences), 'confidence_max': max(confidences)})
    positions = {}
    for row in rows:
        positions[row['position']] = positions.get(row['position'], 0) + 1
    summary = {'backend': backend, 'model': args.model, 'cases': len(CASES), 'requests': len(rows),
               'stable_cases': sum(c['stable'] for c in per_case),
               'correct_requests': sum(r['correct'] for r in rows),
               'position_histogram': dict(sorted(positions.items())),
               'scope': 'Candidate-order sensitivity on one model and a small fixed case set. '
                        'Not a calibration study and not a general accuracy claim.'}
    print('\n' + json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (args.out / 'per_case.json').write_text(json.dumps(per_case, indent=2, ensure_ascii=False))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    files = sorted(p for p in args.out.iterdir() if p.name != 'SHA256SUMS')
    (args.out / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files))
    return summary['stable_cases'] == len(CASES)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--api-key', default=None)
    p.add_argument('--model', default='/model')
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

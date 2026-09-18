"""Standard-library client; DECISION_URL and DECISION_API_KEY configure access."""
import json
import os
from pathlib import Path
import urllib.request

headers = {'Content-Type': 'application/json'}
if os.environ.get('DECISION_API_KEY'):
    headers['Authorization'] = 'Bearer ' + os.environ['DECISION_API_KEY']
request = urllib.request.Request(os.environ.get('DECISION_URL', 'http://127.0.0.1:8000') + '/plugins/decision/infer',
    data=Path(__file__).with_name('request.json').read_bytes(), headers=headers)
with urllib.request.urlopen(request, timeout=200) as response:
    result = json.load(response)
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result['accepted']:
    raise SystemExit('Decision abstained; do not dispatch diagnostic candidates.')

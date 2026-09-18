"""Standard-library client; JEV_DECISON_URL and JEV_DECISON_API_KEY configure access."""
import json
import os
from pathlib import Path
import urllib.request

headers = {'Content-Type': 'application/json'}
if os.environ.get('JEV_DECISON_API_KEY'):
    headers['Authorization'] = 'Bearer ' + os.environ['JEV_DECISON_API_KEY']
request = urllib.request.Request(os.environ.get('JEV_DECISON_URL', 'http://127.0.0.1:8000') + '/plugins/jev-decison/infer',
    data=Path(__file__).with_name('request.json').read_bytes(), headers=headers)
with urllib.request.urlopen(request, timeout=200) as response:
    result = json.load(response)
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result['accepted']:
    raise SystemExit('Decision abstained; do not dispatch diagnostic candidates.')

"""Read-only HTTP checks of the running viewer; does not issue UI commands."""
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
import argparse, hashlib, json, time
from PIL import Image

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8880'

def get(path):
    with urlopen(URL + path, timeout=15) as response:
        return response.read(), response.headers.get_content_type()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-paused', action='store_true')
    args = parser.parse_args()
    folder = ROOT / args.out
    folder.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, passed): checks.append(dict(name=name, passed=bool(passed)))
    capabilities = json.loads(get('/capabilities')[0])
    state = json.loads(get('/state')[0])
    check('correct isolated application', state['app'] == 'pbl-shadow-strategy-integration-v1')
    check('current physical/source admission', capabilities['admitted'] and not capabilities['problems'])
    check('two actual methods and sixteen scheduled scenarios', capabilities['methods'] == ['ratio','unified'] and len(capabilities['scenarios']) == 16)
    frame, mime = get('/frame.png?snapshot=' + str(state['snapshot_id']))
    image = Image.open(BytesIO(frame)); image.load()
    check('bound physical frame is valid 1000 by 700 PNG', mime == 'image/png' and image.size == (1000,700))
    (folder/'physical_frame.png').write_bytes(frame)
    if args.expect_paused:
        check('pause command acknowledged', state['paused'] and state.get('last_command',{}).get('status') == 'executed')
        time.sleep(1.1)
        later = json.loads(get('/state')[0])
        check('paused simulation time and physical forces unchanged during fresh render',
              later['time'] == state['time'] and later['normal'] == state['normal'] and later['palm'] == state['palm'] and later['object_lift'] == state['object_lift'])
        (folder/'later_state.json').write_text(json.dumps(later, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder/'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    result = dict(passed=all(c['passed'] for c in checks), checks=checks, run_id=state.get('run_id'),
                  time=state['time'], method=state.get('method'), shape=state.get('shape'),
                  snapshot_id=state['snapshot_id'], frame_sha256=hashlib.sha256(frame).hexdigest(),
                  scope='Read-only own viewer API and its physical frame. Browser UI actions checked separately. Not a browser screenshot.')
    (folder/'viewer_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)

if __name__ == '__main__': main()

"""Fetch only licensed model data from the official MuJoCo Menagerie repository."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib,time
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'upstream_shadow';OUT.mkdir(exist_ok=True)
def get(url):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'PBL-five-finger-simulation'})
            with urllib.request.urlopen(request,timeout=25) as response:return response.read()
        except Exception:
            if attempt==2:raise
            time.sleep(.5)
sha=json.loads(get('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
tree=json.loads(get(f'https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{sha}?recursive=1'))
files=[entry['path'] for entry in tree['tree'] if entry['type']=='blob' and entry['path'].startswith('shadow_hand/') and (entry['path'].startswith('shadow_hand/assets/') or entry['path'].split('/')[-1] in ['right_hand.xml','README.md','LICENSE','CHANGELOG.md'])]
def fetch(path):
    rel=Path(path[len('shadow_hand/'):]);dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sha}/{path}'
    blob=get(url);dest.write_bytes(blob)
    return dict(path=rel.as_posix(),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),url=url)
with ThreadPoolExecutor(max_workers=5) as pool:manifest=list(pool.map(fetch,files))
(OUT/'download_manifest.json').write_text(json.dumps(dict(commit=sha,repository='google-deepmind/mujoco_menagerie',model='shadow_hand',files=manifest),indent=2),encoding='utf-8')
print(json.dumps(dict(commit=sha,files=len(files),bytes=sum(x['bytes'] for x in manifest))))

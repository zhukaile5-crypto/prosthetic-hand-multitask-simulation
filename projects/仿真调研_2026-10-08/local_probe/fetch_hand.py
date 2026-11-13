"""Download one licensed public example model at a recorded immutable commit."""
from pathlib import Path
import json, urllib.request, hashlib
OUT=Path(__file__).resolve().parent/'menagerie_allegro'
OUT.mkdir(exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'PBL-local-feasibility-audit'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()
sha=json.loads(get('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
prefix='wonik_allegro/'
tree=json.loads(get(f'https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{sha}?recursive=1'))
files=[x['path'] for x in tree['tree'] if x['type']=='blob' and x['path'].startswith(prefix)]
manifest=[]
for path in files:
    rel=Path(path[len(prefix):]);dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sha}/{path}'
    blob=get(url);dest.write_bytes(blob)
    manifest.append(dict(path=str(rel),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),url=url))
(OUT/'download_manifest.json').write_text(json.dumps(dict(commit=sha,repository='google-deepmind/mujoco_menagerie',model='wonik_allegro',files=manifest),indent=2),encoding='utf-8')
print(json.dumps({'commit':sha,'files':len(files),'bytes':sum(x['bytes'] for x in manifest)}))

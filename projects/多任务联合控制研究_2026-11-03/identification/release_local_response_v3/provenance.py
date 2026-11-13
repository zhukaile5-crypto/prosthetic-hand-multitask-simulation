"""Frozen237 V2 closure plus seven fresh V3 inputs; old sources stay unchanged."""
from pathlib import Path
import hashlib,json,sys,shutil
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_local_response_v2 import provenance as parent
PACKAGE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_files():
 inherited=parent.source_files();old=json.loads((ROOT/'identification/release_local_response_v2/dependency_lock.json').read_text(encoding='utf-8'))
 if {k:sha(p) for k,p in inherited.items()}!=old['sources']:raise RuntimeError('Immutable237 V2 source closure changed')
 new={f'research/identification/release_local_response_v3/{p.name}':p for p in PACKAGE.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ('plan.json','README.md'))}
 return {**inherited,**new}
def identity():return {k:sha(p) for k,p in sorted(source_files().items())}
def require_lock():
 path=PACKAGE/'dependency_lock.json'
 if not path.exists():raise RuntimeError('Root must pin reviewed source before execution')
 lock=json.loads(path.read_text(encoding='utf-8'))
 if lock['sources']!=identity():raise RuntimeError('Pilot frozen source changed; use a fresh revision')
 return lock
def snapshot(target,lock):
 target=Path(target);target.mkdir(parents=True,exist_ok=False);files=source_files();records={}
 for key,path in files.items():
  q=target/key;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,q)
  if sha(q)!=lock['sources'][key]:raise RuntimeError('Source snapshot mismatch')
  records[key]=dict(original_path=str(path),snapshot_path=str(q.relative_to(target.parent)),sha256=lock['sources'][key],bytes=q.stat().st_size)
 with (target.parent/'source_snapshot_manifest.json').open('x',encoding='utf-8') as f:json.dump(dict(files=records,all_snapshot_bytes_verified=True),f,ensure_ascii=False,indent=2)

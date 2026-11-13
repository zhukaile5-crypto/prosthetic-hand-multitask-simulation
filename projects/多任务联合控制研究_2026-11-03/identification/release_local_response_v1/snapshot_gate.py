"""Source-checked snapshot envelope and fail-closed restore/replay interface.

No hydration adapter or physics runner is installed. Reading full persisted
data is not a claim that Python object identity/aliases or MjData were restored.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import base64,hashlib,json
from copy import deepcopy
import numpy as np

class RestoreUnavailable(RuntimeError):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
def normalized(v):
 if isinstance(v,dict):
  r={k:normalized(x) for k,x in v.items()}
  if 'mapping_items' in r:r['mapping_items']=[p for p in r['mapping_items'] if p[0] not in ('qp_seconds','control_ms','wall_seconds')]
  return r
 if isinstance(v,list):return [normalized(x) for x in v]
 return v
def mapping(v):return dict(v.get('state',v)['mapping_items'])
def encode(v):
 if isinstance(v,np.ndarray):return dict(array_dtype=v.dtype.str,descr=v.dtype.descr if v.dtype.fields else None,shape=list(v.shape),bytes_b64=base64.b64encode(np.ascontiguousarray(v).tobytes()).decode('ascii'))
 if v is None or isinstance(v,(str,bool,int)):return v
 if isinstance(v,(float,np.floating)):return dict(float_hex=float(v).hex())
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,dict):return dict(mapping_type='builtins.dict',mapping_items=sorted([[encode(k),encode(x)] for k,x in v.items()],key=lambda p:json.dumps(p[0],sort_keys=True,ensure_ascii=False)))
 raise TypeError(type(v))
@dataclass(frozen=True)
class SavedEntry:
 directory:Path
 physics:dict
 encoded_side:dict
 code_references:tuple
 signature:dict
 source_identity:dict
 entry_files_sha256:dict
 @property
 def causal_digest(self):return digest(dict(physics=encode(self.physics),side=normalized(self.encoded_side)))
 @classmethod
 def load_v2(cls,directory,source_run):
  directory=Path(directory);source_run=Path(source_run)
  read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
  manifest=read(source_run/'manifest.json');signature=read(directory/'entry_signature.json')
  with np.load(directory/'entry_physics.npz',allow_pickle=False) as z:physics={k:z[k].item() if z[k].shape==() else z[k].copy() for k in z.files}
  side=read(directory/'entry_side_state.json');refs=read(directory/'entry_code_references.json')
  if signature['integration_state_sha256']!=digest(encode(physics['integration_state'])) or signature['physics_arrays_sha256']!=digest(encode(physics)) or signature['python_causal_side_state_sha256']!=digest(normalized(side)):raise ValueError('Persisted entry checksum mismatch')
  required={'controller','sensors','preparation','skin_geometry','session_causal_state','plant_side'}
  if not required<=set(mapping(side)):raise ValueError('Full declared causal side state missing')
  sensor=mapping(mapping(side)['sensors']);skin=mapping(mapping(side)['skin_geometry'])
  if not {'rng','queue','last_frame','samples'}<=set(sensor) or 'fk_data' not in skin or 'samples' not in skin:raise ValueError('RNG/queue/FK state missing')
  for ref in refs:
   if ref['origin_sha256']!=manifest['sources'][ref['origin_source_key']] or any(manifest['sources'][k]!=h for k,h in ref['covered_native_binding_sha256'].items()):raise ValueError('Module implementation reference differs from locked source')
  def check_implementations(node):
   if isinstance(node,dict):
    if 'origin_source_key' in node and node.get('origin_sha256')!=manifest['sources'].get(node['origin_source_key']):raise ValueError('Serialized instance implementation differs from locked source')
    for value in node.values():check_implementations(value)
   elif isinstance(node,list):
    for value in node:check_implementations(value)
  check_implementations(side)
  for value in physics.values():
   if isinstance(value,np.ndarray):value.setflags(write=False)
  names=['entry_physics.npz','entry_side_state.json','entry_code_references.json','entry_signature.json']
  return cls(directory,physics,side,tuple(refs),signature,dict(manifest['sources']),{n:sha(directory/n) for n in names})

class FullStateRestoreAdapter(Protocol):
 def source_identity(self)->dict:...
 def restore_without_sampling_or_advancing(self,snapshot:SavedEntry):...
 def capture_without_sampling_or_advancing(self,handle)->SavedEntry:...

def restore_zero_step_gate(snapshot:SavedEntry,adapter=None):
 if adapter is None:raise RestoreUnavailable('Full Python/MjData/FK/model-mutation hydration adapter MISSING; qpos-only or reset-dt substitutes forbidden')
 actual=adapter.source_identity()
 if any(actual.get(k)!=v for k,v in snapshot.source_identity.items()):raise RestoreUnavailable('Restore implementation does not match the complete230-source parent identity')
 expected=snapshot.causal_digest
 handle=adapter.restore_without_sampling_or_advancing(deepcopy(snapshot))
 captured=adapter.capture_without_sampling_or_advancing(handle)
 if captured.causal_digest!=expected or snapshot.causal_digest!=expected:raise RestoreUnavailable('Zero-step full-state roundtrip is not exact')
 return handle

def require_physical_replay_certificate(certificate=None):
 if certificate is None:raise RestoreUnavailable('Root-owned native continuation replay evidence MISSING; no physical branch measurement is executable')
 raise RestoreUnavailable('Replay certificate schema/verifier and root-only runner are intentionally NOT installed in this draft')

def verify_parent_sources(source_run,current_path_for_key):
 source_run=Path(source_run);manifest=json.loads((source_run/'manifest.json').read_text(encoding='utf-8-sig'));expected=manifest['sources']
 if len(expected)!=230:raise ValueError('The declared complete230-source parent is required')
 for key,h in expected.items():
  if sha(source_run/'sources'/key)!=h or sha(current_path_for_key(key))!=h:raise RestoreUnavailable('Snapshot/current source identity mismatch: '+key)
 return dict(expected)

"""Immutable fusion input and two independently factorized command channels."""
from pathlib import Path
import hashlib,json
import numpy as np
PACKAGE=Path(__file__).resolve().parent;ROOT=PACKAGE.parents[1]
COHORT=ROOT/'output/development_v13_joint_pilot_v1'
VARIANTS={'both':(True,True),'lift_only':(True,False),'fingers_only':(False,True),'neither':(False,False)}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def load():
    done=read(COHORT/'execution_complete.json');manifest=read(COHORT/'manifest.json')
    if not done['execution_complete'] or not done['sources_unchanged'] or done['runtime_failures']:raise ValueError('Complete pinned source pilot required')
    jobs=[j for j in manifest['jobs'] if (j['method'],j['scenario'],j['seed'],j['motion_mode'])==('fusion_effort','nominal',617901,'hold')]
    if len(jobs)!=1:raise ValueError('Fixed diagnostic source identity ambiguous')
    folder=Path(jobs[0]['destination']);result=read(folder/'result.json');meta=read(folder/'metadata.json');path=folder/'trace.npz'
    fingerprint=sha(path)
    if fingerprint!=result['trace_sha256'] or not meta['completed']:raise ValueError('Complete immutable source trace required')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files if k!='metadata'}
    if not exact(a['time'],np.arange(3201)*.005) or a['ctrl'].shape!=(3201,22) or a['executed_reference21'].shape!=(3201,21):raise ValueError('Full16s source clock/22ports/21references required')
    if not np.isfinite(a['ctrl']).all() or not np.isfinite(a['executed_reference21']).all():raise ValueError('Finite actual commands required')
    if meta['config']['timestep']!=.0005 or meta['release_at']!=12. or meta['mechanical_release_at']!=12.:raise ValueError('This plan requires actual12s release and original0.5ms step')
    k=2400;obs=json.loads(str(a['observation_json'][k]));entry=np.asarray(obs['robot_reference21'],float)
    incoming=np.asarray(obs['robot_actuator_target21'],float)
    if entry.shape!=(21,) or incoming.shape!=(21,) or obs['robot_control_time_s']!=12.:raise ValueError('Incoming actual entry21/clock missing')
    # This pilot had unchanged receipts. A future different source must use a
    # new plan rather than call post-projection values the original request.
    if not exact(a['requested_reference21'][k:],a['executed_reference21'][k:]):raise ValueError('Current plan source request/actual mismatch not supported')
    environments=[json.loads(str(v)) for v in a['environment_json']]
    if any(set(v)!={'force','torque','squeeze','obstacle_x','friction'} for v in environments):raise ValueError('Complete fixed environment schedule required')
    return dict(folder=folder,path=path,trace_sha256=fingerprint,metadata_sha256=sha(folder/'metadata.json'),metadata=meta,arrays=a,environments=environments,
        entry_reference21=entry,entry_engine21=incoming,release_index=k,source_manifest=manifest,source_result=result)
def reference_program(record,variant):
    if variant not in VARIANTS:raise ValueError('Four predeclared factors only')
    lift,fingers=VARIANTS[variant];ref=record['arrays']['executed_reference21'].copy();start=record['release_index'];entry=record['entry_reference21']
    if not fingers:ref[start:,:18]=entry[:18]
    if not lift:ref[start:,18]=entry[18]
    return ref
def unchanged(record):
    if sha(record['path'])!=record['trace_sha256'] or sha(record['folder']/'metadata.json')!=record['metadata_sha256']:raise ValueError('Original input changed')

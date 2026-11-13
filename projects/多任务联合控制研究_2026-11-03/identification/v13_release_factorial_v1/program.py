"""No feedback: typed source reference program -> frozen sole actor callback."""
from dataclasses import asdict
import sys
import numpy as np
from .inputs import ROOT,canonical,exact,reference_program,VARIANTS,unchanged
def imports():
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    import research_imports,mujoco
    from foundation_config import FoundationConfig
    from v13_joint.plant import JointPlant
    return mujoco,FoundationConfig,JointPlant
def config(record):
    _,Config,_=imports();c=Config(**record['metadata']['config'])
    if canonical(asdict(c))!=canonical(record['metadata']['config']):raise ValueError('Original config changed')
    return c
def set_release_event(plant,record,k):
    if k!=record['release_index']:return
    from v13_joint.lifecycle import ReleaseEntry
    from v13_a1.codec import Reference21
    actual=plant.reference_snapshot();entry=record['entry_reference21'];incoming=record['entry_engine21']
    if not exact(actual.values,entry) or not exact(plant.engine_snapshot().values,incoming):raise ValueError('Actual12s incoming entry did not match original; no invented entry')
    plant.release_started=12.;plant.joint_release_entry=ReleaseEntry(12.,Reference21(entry,12.),int(record['arrays']['mode_generation'][k]))
    plant._joint_mode='release';plant.mode_generation=int(record['arrays']['mode_generation'][k])
def apply(plant,record,k,reference21):
    from v13_a1.codec import Reference21
    # Alpha is historical memory solely for comparable instrument diagnostics;
    # _joint_committed prevents every subsequent callback from decoding alpha.
    plant.alpha=record['arrays']['alpha'][k].copy();plant.phase=str(record['arrays']['phase'][k])
    plant._joint_committed=Reference21(reference21,plant.time)
    plant._joint_mode='release' if k>=record['release_index'] else 'holding'
    plant.lift_ref=float(reference21[18]);plant.wrist_ref=reference21[19:].copy()
    plant._apply_ctrl()  # Frozen JointPlant -> frozen A1 sole robot21 writer.
def preflight(record):
    mj,_,Plant=imports();saved={};forbidden=[]
    for name in ('mj_step','mj_step1','mj_step2'):
        saved[name]=getattr(mj,name)
        def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('No preflight physics '+_name)
        setattr(mj,name,deny)
    reports=[]
    try:
        for variant in VARIANTS:
            p=Plant(config(record));a=record['arrays'];program=reference_program(record,variant)
            if p.model_sha256!=record['metadata']['model_sha256']:raise ValueError('Original model hash mismatch')
            initial=all((exact(p.data.qpos,a['qpos'][0]),exact(p.data.qvel,a['qvel'][0]),exact(p.data.ctrl,a['ctrl_before'][0])))
            if not initial:raise ValueError('Reset state mismatch; no future restoration')
            checks=0;max_rate_excess=0.
            for k,t in enumerate(a['time']):
                p.control_steps=k
                p.env={'force':np.zeros(3),'torque':np.zeros(3),'squeeze':0.,'obstacle_x':0.} if k==0 else dict(record['environments'][k-1])
                set_release_event(p,record,k)
                before=p.reference_snapshot().values.copy()
                for offset in (0.,.0025,.004875):
                    p.data.time=float(t+offset);apply(p,record,k,program[k]);checks+=1
                    if not exact(p.engine_snapshot().values,p.actor21.encode(p._joint_committed).values):raise ValueError('Actual21 encode mismatch')
                    if variant=='both' and not exact(p.data.ctrl,a['ctrl'][k]):raise ValueError('Both full22 source mismatch at'+str(k))
                limited=p.actor21.ctrl_limited21;port=p.engine_snapshot().values
                if np.any(port[limited]<p.actor21.ctrl_range21[limited,0]) or np.any(port[limited]>p.actor21.ctrl_range21[limited,1]):raise ValueError('Actual native control range exceeded')
                if k>=1500:
                    from v13_joint.design import reference_rates
                    excess=np.max(np.maximum(0.,abs(program[k]-before)-reference_rates(p._joint_mode)*.005))
                    max_rate_excess=max(max_rate_excess,float(excess))
                    if excess>1e-12:raise ValueError('Declared posthandoff5ms software rate exceeded')
            reports.append(dict(variant=variant,passed=True,initial_exact=initial,native21_checks=checks,max_rate_excess=max_rate_excess,model_sha256=p.model_sha256))
        unchanged(record)
        return dict(passed=True,reports=reports,integration_calls=forbidden)
    finally:
        for name,value in saved.items():setattr(mj,name,value)

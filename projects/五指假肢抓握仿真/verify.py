"""Audit complete five-digit model, free-object grasp, and on/off reflex behavior."""
from pathlib import Path
import json,hashlib
import mujoco,numpy as np
from simulation import GraspSimulation,png_bytes,PREFIX,GROUPS
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'evidence';DEST.mkdir(exist_ok=True)
sim=GraspSimulation();m=sim.model
hand_joints=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith('rh_')]
hand_actuators=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a) for a in range(m.nu) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a) or '').startswith('rh_')]
free_j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')
assert len(hand_joints)==24 and len(hand_actuators)==20
assert m.jnt_type[free_j]==mujoco.mjtJoint.mjJNT_FREE and m.neq==0
manifest=json.loads((ROOT/'upstream_shadow/download_manifest.json').read_text())
for file in manifest['files']:
    assert hashlib.sha256((ROOT/'upstream_shadow'/file['path']).read_bytes()).hexdigest()==file['sha256']
audit=dict(hand_joints=hand_joints,hand_actuators=hand_actuators,digit_groups=GROUPS,free_object=True,equality_constraints=int(m.neq),model_sha256=hashlib.sha256((ROOT/'model.xml').read_bytes()).hexdigest(),source_commit=manifest['commit'],source_hashes_verified=True)
audit['simulation_sha256']=hashlib.sha256((ROOT/'simulation.py').read_bytes()).hexdigest()
(DEST/'model-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
results=[]
for module,action in [('slip','disturb'),('force','squeeze'),('gate','release')]:
    for enabled in [True,False]:
        s=GraspSimulation();s.command(module,enabled);s.command('grasp');s.advance(8)
        baseline=s.snapshot()
        assert baseline['state']=='HOLDING' and min(baseline['normal'])>.3 and baseline['lift_mm']>40
        s.command(action);s.advance(3 if module=='gate' else 2)
        trace=[a for a in s.history if a[0]>=8]
        metrics=dict(over_limit_s=round(sum(max(a[1:6])>s.limit for a in trace)*.005,3),
            excess_integral_Ns=round(sum(max(0,max(a[1:6])-s.limit) for a in trace)*.005,4),
            peak_after_action_N=round(max(max(a[1:6]) for a in trace),3),
            final_shift_y_mm=round((s.data.xpos[s.object,1]-s.initial_y)*1000,2))
        results.append(dict(module=module,enabled=enabled,action=action,baseline=baseline,final=s.snapshot(),metrics=metrics))
        print(module,enabled,s.state,'F',s.normal.round(2),'events',s.slip_events,s.force_events,s.release_slip_events,metrics,flush=True)
assert results[0]['final']['state']=='HOLDING' and results[0]['final']['slip_events']>0
assert results[1]['final']['state']=='DROPPED'
assert results[2]['final']['force_events']>0 and results[3]['final']['force_events']==0
assert results[2]['metrics']['excess_integral_Ns']<results[3]['metrics']['excess_integral_Ns']*.7
assert results[4]['final']['state']=='RELEASED' and results[4]['final']['release_slip_events']==0
assert max(results[4]['final']['normal'])<.05 and abs(results[4]['final']['object_z']-.3655)<.005
assert results[5]['final']['release_slip_events']>0
(DEST/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

s=GraspSimulation();s.command('grasp');s.advance(8);height=s.object_z;q_before=s.data.qpos.copy();s.advance(30)
stability=dict(duration_s=30,state=s.state,height_drift_mm=(s.object_z-height)*1000,normal=s.snapshot()['normal'],slip_events=s.slip_events,force_events=s.force_events)
assert s.state=='HOLDING' and min(s.normal)>.3 and abs(s.object_z-height)<.005
(DEST/'stability.json').write_text(json.dumps(stability,indent=2),encoding='utf-8')
s=GraspSimulation();s.command('grasp');s.advance(8);s.command('disturb');s.advance(2)
assert s.state=='HOLDING'
s.command('release');s.advance(10)
assert s.state=='RELEASED' and max(s.normal)<.05 and abs(s.object_z-.3655)<.005 and s.release_slip_events==0
(DEST/'workflow.json').write_text(json.dumps(s.snapshot(),ensure_ascii=False,indent=2),encoding='utf-8')
long_trial=GraspSimulation();long_trial.command('grasp');long_trial.advance(133.87)
long_trial.command('disturb');long_trial.advance(6.57)
assert long_trial.state=='HOLDING'
long_trial.command('release');long_trial.advance(10)
assert long_trial.state=='RELEASED' and max(long_trial.normal)<.05 and abs(long_trial.object_z-.3655)<.005
assert long_trial.release_slip_events==0
(DEST/'long-workflow.json').write_text(json.dumps(long_trial.snapshot(),ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: complete articulated hand, free-body grasp, three reflex comparisons, 30-second hold.',flush=True)

from render_evidence import render_sequence
render_sequence()

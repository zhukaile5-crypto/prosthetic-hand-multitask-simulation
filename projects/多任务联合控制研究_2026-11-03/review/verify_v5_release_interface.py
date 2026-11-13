"""Independent actual release/FK/decoder phase audit; mj_step is forbidden."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,hashlib,json,runpy,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
def forbidden_step(*a,**k):raise AssertionError('Independent release audit must not integrate physics')
mujoco.mj_step=forbidden_step
from foundation_config import FoundationConfig
from lifecycle_plant import LifecyclePlant
from lifecycle_release import LifecycleRelease,ReleaseCommand
checkers=runpy.run_path(str(ROOT/'review/verify_v5_contract.py'))
OLD=ROOT/'output/development_v4_integrated/runs/nominal__joint__610103/trace.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(data):return SimpleNamespace(**{k:(np.asarray(v) if isinstance(v,list) and k!='skin_patches' else v) for k,v in data.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args()
    before={n:sha(ROOT/n) for n in ['lifecycle_release.py','lifecycle_plant.py','skin_geometry.py','runtime.py','task_contract.py']}
    with np.load(OLD,allow_pickle=False) as z:
        observed=json.loads(str(z['observation_json'][2400]));ctrl=z['ctrl_before'][2400].copy();meta=json.loads(str(z['metadata']))
    cfg=FoundationConfig(**{k:v for k,v in meta['config'].items() if k in FoundationConfig.__dataclass_fields__});checks={};details={}
    plant=LifecyclePlant(cfg);driver=LifecycleRelease(plant);aids=driver.aids
    def packet(t,kind):
        f=deepcopy(observed);f['time']=t;f['dt']=.005;f['sample_time']=t-.01
        if kind in ('clear','unknown'):
            f['normal']=[0.]*5;f['auxiliary_pressure']=0.;f['skin_patches']=[];f['valid']=[kind=='clear']*5;f['auxiliary_valid']=kind=='clear'
        elif kind=='recontact':
            f['valid']=[True]*5;f['auxiliary_valid']=True;f['auxiliary_pressure']=0.;f['normal']=[0.]*5
            p=deepcopy(f['skin_patches'][0]);p['normal']=.3;p['sample_time']=t-.01;p['age']=.01;f['normal'][p['digit']]=.3;f['skin_patches']=[p]
        return f
    # Naturally empty skin, real FK and real QP: ownership of an already open
    # command must not restart the earlier clearance posture or fake clear.
    initial=LifecycleRelease(plant);initial.request(frame(packet(12.,'unknown')),initial.full_open,.1,np.zeros(2));first=initial.step(frame(packet(12.,'unknown')))
    checks['already_open_ownership_does_not_restart_tuck']=initial.opening_committed and np.allclose(first.target,initial.full_open,atol=1e-10,rtol=0) and initial.first_clear is None and initial.was_clear is False
    # Exercise genuine nonempty archived public patch geometry, not an empty
    # replacement function; all state transitions remain off-policy inputs.
    f0=frame(packet(12.,'observed'));driver.request(f0,ctrl[aids],.1,np.zeros(2));plant.data.ctrl[:]=ctrl;plant.release_started=12.;plant.alpha=np.asarray(observed['alpha']).copy()
    rows,digits,bound=driver.kinematics.contact_rows(f0);checks['actual_nonempty_distributed_FK_exercised']=len(rows)>0
    # Unrelated object generalized coordinates must not affect robot rows.
    object_joint=int(np.flatnonzero((driver.kinematics.model.jnt_bodyid==plant.object)&(driver.kinematics.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE))[0]);adr=int(driver.kinematics.model.jnt_qposadr[object_joint]);driver.kinematics.data.qpos[adr:adr+3]=[20.,-30.,40.]
    rows2,digits2,bound2=driver.kinematics.contact_rows(f0)
    checks['FK_unrelated_object_pose_poison_invariant']=np.array_equal(rows,rows2) and digits==digits2 and np.array_equal(bound,bound2)
    outputs=[];envelope=True;decoder_ok=True;peak_step=0.;clear_seen=False;first_clear=None
    for i in range(130):
        t=12+i*.005;kind='observed' if i<4 else 'clear' if i<52 else 'unknown' if i<56 else 'recontact' if i<70 else 'clear';f=frame(packet(t,kind))
        command=driver.step(f);d=command.debug;before_cmd=np.asarray(d['actuator_before']);after=np.asarray(d['actual_actuator_target']);rate=np.asarray(d['actuator_rate_limit']);dt=float(d['dt'])
        envelope &= bool((after>=driver.lower-1e-8).all() and (after<=driver.upper+1e-8).all() and (abs(after-before_cmd)<=rate*dt+1e-8).all());peak_step=max(peak_step,float(abs(after-before_cmd).max()))
        plant.apply_release_command(command);decoder_ok &= np.array_equal(plant.data.ctrl[aids],after)
        clear_seen |= driver.first_clear is not None
        if first_clear is None and driver.first_clear is not None:first_clear=driver.first_clear
        outputs.append(dict(clear_confirmed=driver.first_clear is not None,release_authority=d['release_authority'],holding_called=False,monitoring_active=d['monitoring_active'],posture_mode=d['release_posture_mode'],recontact_count=d['observed_recontact_count'],observation_unknown=kind=='unknown'))
    latch=checkers['verify_clear_latch'](outputs)
    checks['actual_clear_unknown_recontact_latch']=latch['passed'];checks['first_clear_time_not_reset']=driver.first_clear==first_clear and first_clear is not None
    checks['exactly_one_observed_recontact']=driver.recontacts==1;checks['actual_command_bounds_and_rates']=envelope;checks['decoder_executes_logged_command']=decoder_ok
    # Hard decoder rejection is separate from the planner's own bound checks.
    before_cmd=plant.lifecycle_targets.copy();idx=int(np.flatnonzero((driver.rate>0)&(before_cmd<driver.upper-.05))[0]);bad=before_cmd.copy();bad[idx]+=driver.rate[idx]*.005*2.
    try:plant.apply_release_command(ReleaseCommand(bad,.1,np.zeros(2),'test',{}));checks['decoder_rejects_excess_rate']=False
    except ValueError:checks['decoder_rejects_excess_rate']=True
    # Same public patch geometry with hidden identity fields cannot alter QP.
    one=LifecycleRelease(plant);two=LifecycleRelease(plant);poison=packet(12.,'observed');poison.update(true_mass=99.,true_mu=.001,target_id='different',future_disturbance=[99.,0.,0.])
    for p in poison['skin_patches']:p.update(is_target=False,object_id='different')
    one.request(f0,ctrl[aids],.1,np.zeros(2));two.request(frame(poison),ctrl[aids],.1,np.zeros(2));x=one.step(f0);y=two.step(frame(poison))
    checks['actual_release_QP_hidden_identity_invariant']=np.array_equal(x.target,y.target) and x.stage==y.stage
    checks['source_frozen_during_interface_audit']=all(sha(ROOT/n)==v for n,v in before.items())
    result=dict(schema='independent_v5_actual_release_interface_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],source_sha256=before,
        input_trace_sha256=sha(OLD),reviewer_sha256=sha(__file__),first_clear_confirmation=first_clear,final_recontact_count=driver.recontacts,nonempty_geometry_rows=len(rows),maximum_actuator_step=peak_step,
        release_latch_checks=latch,physics_steps_executed=0,
        scope='Actual release constructor/request/step, distributed robot FK and decoder on archived/synthetic public observations. No mj_step. This verifies interfaces and phase ownership, not collision-free physical release, measured joint-rate safety or new performance.')
    if args.out:
        out=args.out.resolve()
        if not out.is_relative_to(ROOT/'review') or out.exists():raise ValueError('Fresh own review path required')
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

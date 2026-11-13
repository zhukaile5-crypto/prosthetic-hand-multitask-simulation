"""Bounded author algebra/device/archived-public interface gate; zero physics."""
from copy import deepcopy
from dataclasses import asdict,is_dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from contracts import SensorNoise
from sensors import HandSensors
from v17_contact_transfer.motion_domain_sensors import MotionDomainSensors,ADDED_FIELDS,SLIP_DOMAIN_ROLE
from v17_contact_transfer.motion_adapter import (PublicMotionAdapter,MotionTask,FingerMotionProjection,build_motion_task)

CHECKS=[]
def ck(name,value,details=None):CHECKS.append(dict(name=name,passed=bool(value),details=details))
def reject(fn):
    try:fn()
    except (TypeError,ValueError):return True
    return False
def bits(a,b):
    if type(a) is not type(b):return False
    if is_dataclass(a):return bits(asdict(a),asdict(b))
    if isinstance(a,np.ndarray):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return a.keys()==b.keys() and all(bits(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(bits(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)):return struct.pack('d',a)==struct.pack('d',b)
    return a==b
def calibration():
    return dict(force_matrix=np.eye(5)*2,force_tau=.015,reference_alpha=np.full(5,.5),
        response_error_bound=np.full(5,.2),nominal_force_weights=np.ones(5),
        effort_bias_N=.1,load_error_bound_N=.3)
def raw(t,ns):
    patches=[]
    for i,n in enumerate(ns):
        patches.append(dict(digit=i,normal=float(n),position=[.01*i,0,0],normal_dir=[0,0,1],
            force=[0,0,float(n)],slip_velocity=[.01,.02,0],torque=[0,0,0],is_target=bool(i%2)))
    return dict(time=t,patches=patches,joint=np.full(5,.5),joint_velocity=np.zeros(5),alpha=np.full(5,.5),
        closure_min=np.full(5,-1.),closure_max=np.full(5,1.5),closure_permission=np.ones(5,bool),
        raw_joint=np.zeros(24),raw_joint_velocity=np.zeros(24),lift=.05,lift_ref=.1,lift_velocity=0.,
        motor_effort=11.,bare_hand_weight=10.,known_hand_inertia=1.,lift_damping=3.,
        support_axis=[0,0,1],phase='HOLDING',object_mass=99.,object_pose=[8,8,8],future_schedule=[999.])

def main(out):
    noise=SensorNoise(force_sigma=.01,tangential_sigma=.005,position_sigma=.0001,slip_sigma=.0002,
        delay=.01,dropout_windows=((.005,.01,(0,)),))
    a=MotionDomainSensors(HandSensors(calibration(),noise,seed=71));b=HandSensors(calibration(),noise,seed=71)
    calls=[];original=a.inner.sample
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    a.inner.sample=counted
    levels=((.02,.021,0.,.3,.5),(.03,0.,.1,.3,.5),(.04,.04,.02,.3,.5),(.1,.1,.1,.1,.1),(.2,.2,.2,.2,.2))
    all_fields=True;all_rng=True;masks=[];draws=[]
    for k,ns in enumerate(levels):
        r=raw(k*.005,ns);fa=a.sample(r);fb=b.sample(r)
        stripped={key:v for key,v in vars(fa).items() if key not in ADDED_FIELDS}
        all_fields &= bits(stripped,vars(fb));all_rng &= bits(a.rng.bit_generator.state,b.rng.bit_generator.state)
        j=max(0,k-2);expected=(np.asarray(levels[j])>.02)&fb.valid
        masks.append(bits(fa.slip_valid5,expected) and fa.slip_domain_acquisition_time_s==fb.sample_time)
        draws.append((a.last_diagnostics['gaussian_draws_per_tick'],a.last_diagnostics['uniform_draws_per_tick']))
    ck('all original SensorFrame fields IEEE-identical to untouched sensor',all_fields)
    ck('original RNG state and exactly48Gaussian6uniform diagnostics unchanged',all_rng and all(x==(48,6) for x in draws))
    ck('same raw aggregation strict >.02 domain delayed with original valid/dropout',all(masks),masks)
    before=deepcopy(a.rng.bit_generator.state);n=len(calls);again=a.sample(raw(.02,[9]*5));again.slip_valid5[:]=False
    ck('duplicate tick calls inner zero times and returns isolated cached metadata',len(calls)==n and bits(before,a.rng.bit_generator.state) and np.any(a.sample(raw(.02,[0]*5)).slip_valid5))
    ck('one inner call per new public tick',len(calls)==5)
    c=MotionDomainSensors(HandSensors(calibration(),noise,seed=91));d=MotionDomainSensors(HandSensors(calibration(),noise,seed=91))
    rc=raw(0,[.1]*5);rd=deepcopy(rc)
    for p in rd['patches']:p['is_target']=not p['is_target'];p['external_geom']=99999
    ck('actor labels do not select slip validity or alter original public fields',bits(vars(c.sample(rc)),vars(d.sample(rd))))
    override=MotionDomainSensors(HandSensors(calibration(),noise));original2=HandSensors(calibration(),noise)
    rr=raw(99,[.1]*5);fo=override.sample(rr,time=0.,phase='READY');fu=original2.sample(rr,time=0.,phase='READY')
    ck('original time and phase overrides forwarded without raw reconstruction',bits({k:v for k,v in vars(fo).items() if k not in ADDED_FIELDS},vars(fu)))
    a.reset();b.reset();ck('explicit reset resets owned inner and metadata together',bits({k:v for k,v in vars(a.sample(raw(0,[.1]*5))).items() if k not in ADDED_FIELDS},vars(b.sample(raw(0,[.1]*5)))))
    ck('numeric-string/bool time rejected before new sampling',reject(lambda:a.sample(raw('0',[1]*5))) and reject(lambda:a.sample(raw(True,[1]*5))))
    advanced=HandSensors(calibration(),noise);advanced.sample(raw(0,[.1]*5));advanced.sample(raw(.005,[.1]*5))
    late=MotionDomainSensors(advanced).sample(raw(.01,[.1]*5))
    ck('unowned acquisition history remains unknown, no noisy-pressure guess',not late.slip_domain_available and not late.slip_valid5.any())
    # Algebra oracle: all velocities are absolute world velocities at acquisition.
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);B=np.array([[1.,0,0],[0,1.,0]])
    H=np.zeros((25,21));H[:21]=np.eye(21);J=np.zeros((3,25));J[0,0]=2.;J[1,1]=-3.
    row=FingerMotionProjection(0,B,[.1,.2,.3],J,[.01,-.02])
    kwargs=dict(time_s=.02,acquisition_time_s=.01,rotation_acquisition3x3=R,reference_to_robot25x21=H)
    task=build_motion_task(**kwargs,projections=[row]);M=np.zeros((2,21));M[0,1]=-3.;M[1,0]=-2.
    ck('absolute robot velocity plus external-minus-hand slip, rotation and correction sign',np.array_equal(task.matrix_Mx21,M) and np.allclose(task.target_M,[-.0042,.0024],atol=1e-18,rtol=0))
    ck('no invented stationary row when no valid projection',build_motion_task(**kwargs,projections=[]) is None)
    ck('at most one two-row measurement per finger',reject(lambda:build_motion_task(**kwargs,projections=[row,row])))
    ck('strict fields and immutable numeric task arrays',reject(lambda:MotionTask(0,np.full((2,21),'1'),[0,0],{})) and reject(lambda:FingerMotionProjection(True,B,[0,0,0],J,[0,0])) and not task.matrix_Mx21.flags.writeable)
    ck('task explicitly denies object state/servo/pressure/safety interpretation',not task.metadata['external_rigid_body_twist_available'] and not task.metadata['future_motion_prediction_available'] and not task.metadata['force_safety_certified'])
    # Small actual-model/public-packet interface. No physics or new sampling.
    import mujoco
    forbidden=[]
    def no(*args,**kw):forbidden.append('forbidden');raise RuntimeError('No physical/sensor pipeline')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):setattr(mujoco,name,no)
    from hand_plant import _model
    from foundation_config import FoundationConfig,OPEN
    from v16_release_geometry.public_adapter import PublicReleaseAdapter
    model,_=_model(FoundationConfig());adapter=PublicMotionAdapter(model,open18=np.concatenate(OPEN));parent=PublicReleaseAdapter(model,open18=np.concatenate(OPEN))
    p=ROOT/'output/v16b_goal_corrected_hold_v1/runs/fragile_conflict__ratio_effort__619732/trace.npz'
    with np.load(p,allow_pickle=False) as z:times=z['time'];obs=z['observation_json']
    selected=np.flatnonzero((times>=11.75-1e-9)&(times<=12.05+1e-9));frames=[];same=True
    # Synthetic validity metadata here is for interface replay only. The real
    # raw-derived mask is tested against real inner HandSensors above.
    for i in selected:
        f=SimpleNamespace(**json.loads(str(obs[i])));f.slip_valid5=np.asarray(f.valid,bool)&(np.asarray(f.normal)>.02)
        f.slip_domain_acquisition_time_s=f.sample_time;f.slip_domain_time_s=f.time;f.slip_domain_role=SLIP_DOMAIN_ROLE;f.slip_domain_available=True
        f.forbidden_object_truth='ignored';frames.append(f)
        ah=adapter.observe(f);ph=parent.observe(f);same &= bits(ah,ph)
        if abs(f.time-12)<1e-9:ap=adapter.begin_release(12);pp=parent.begin_release(12)
    ck('parent coacquired signed support and pre-release prior unchanged',same and bits(ap,pp))
    f=frames[-1];current=np.asarray(f.robot_reference21);rates=np.r_[np.full(18,3.),.18,.12,.12]
    lo=np.maximum(np.asarray(f.reference_lower21)-current,-rates*.02);hi=np.minimum(np.asarray(f.reference_upper21)-current,rates*.02)
    desired=np.clip(np.asarray(f.release_nominal_reference21),current+lo,current+hi)
    got=adapter.current_geometry_tasks(f,current,desired,lo,hi);base=parent.current_geometry_tasks(f,current,desired,lo,hi)
    ck('parent geometry arrays retained with independent current-time diagnostic',bits(got.normal_matrix_m_per_reference,base.normal_matrix_m_per_reference) and bits(got.tangent_matrix_m_per_reference,base.tangent_matrix_m_per_reference) and got.diagnostics['current_time_s']==f.time)
    mt=adapter.motion_task
    ck('real public robot geometry exercises optional motion rows with no writer',mt is not None and mt.matrix_Mx21.shape[0]<=10 and adapter.actor.write_count==0,adapter.last_motion_diagnostics)
    fbad=deepcopy(f);fbad.slip_valid5=np.zeros(5,bool)
    adapter.current_geometry_tasks(fbad,current,desired,lo,hi)
    ck('all slip-invalid yields None rather than target-zero stationary task',adapter.motion_task is None)
    fbad=deepcopy(f);fbad.slip_domain_acquisition_time_s-=.005
    adapter.current_geometry_tasks(fbad,current,desired,lo,hi)
    ck('slip domain clock mismatch yields explicit unknown',adapter.motion_task is None and 'clock mismatch' in adapter.last_motion_diagnostics['reason'])
    fresh=PublicMotionAdapter(model,open18=np.concatenate(OPEN));fresh.observe(f);fresh.current_geometry_tasks(f,current,desired,lo,hi)
    ck('missing coacquired robot history not replaced by current robot velocity',fresh.motion_task is None and 'history missing' in fresh.last_motion_diagnostics['reason'])
    changed=deepcopy(f);changed.robot_velocity25=list(f.robot_velocity25);changed.robot_velocity25[0]+=.1
    ck('duplicate consumed velocity mutation rejected',reject(lambda:adapter.observe(changed)))
    # Split one observed patch and replay one independent public adapter.
    split=PublicMotionAdapter(model,open18=np.concatenate(OPEN))
    for item in frames:
        ff=deepcopy(item)
        if abs(ff.time-f.time)<1e-9:
            first=deepcopy(ff.skin_patches[0]);second=deepcopy(first);first['normal']/=2;second['normal']/=2
            ff.skin_patches=[first,second]+ff.skin_patches[1:]
        split.observe(ff)
    split.current_geometry_tasks(ff,current,desired,lo,hi);sm=split.motion_task
    ck('patch subdivision conserves pair rows, correction and pressure-share geometry',mt is not None and sm is not None and mt.matrix_Mx21.shape==sm.matrix_Mx21.shape and np.allclose(mt.matrix_Mx21,sm.matrix_Mx21,atol=1e-14,rtol=0) and np.allclose(mt.target_M,sm.target_M,atol=1e-16,rtol=0))
    ck('no integration forward contact or writer calls',not forbidden and adapter.actor.write_count==0)
    sources={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (ROOT/'v17_contact_transfer/motion_adapter.py',ROOT/'v17_contact_transfer/motion_domain_sensors.py',Path(__file__))}
    report=dict(schema='v17_public_motion_adapter_author_v1',passed=all(c['passed'] for c in CHECKS),checks=CHECKS,
        check_count=len(CHECKS),source_sha256=sources,real_frame_test_scope='61 archived public frames; validity metadata synthetic only in interface replay, actual raw-domain mask independently tested with original device above',
        not_physical_performance_or_full_state_admission=True,physics_steps=0,forward_calls=0,contact_queries=0,writer_calls=0,forbidden_calls=forbidden)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=report['passed'],count=len(CHECKS),failures=[c['name'] for c in CHECKS if not c['passed']],source_sha256=sources),ensure_ascii=False))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

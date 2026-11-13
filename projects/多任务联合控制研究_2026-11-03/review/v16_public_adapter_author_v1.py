"""Public adapter analytic + archived packet zero-physics acceptance."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import fields
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
from hand_plant import _model
from foundation_config import FoundationConfig,OPEN
from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
from v16_release_geometry.public_adapter import PublicReleaseAdapter,affine_equilibrium_length_derivative


def main(out):
    checks=[];forbidden=[]
    def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),details=d))
    def rejects(n,fn):
        try:fn();ck(n,False)
        except (ValueError,TypeError):ck(n,True)
    def no(*a,**kw):forbidden.append('forbidden');raise RuntimeError('forbidden physics/sensor pipeline')
    for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
        if hasattr(mujoco,n):setattr(mujoco,n,no)
    model,_=_model(FoundationConfig())
    gid=int(model.geom('support').id);half=float(model.geom_size[gid,2])
    mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,half]),half,float(np.cos(np.pi/4)))
    new=lambda **kw:PublicReleaseAdapter(model,open18=np.concatenate(OPEN),**kw)
    a=new(receiver_mount=mount)
    ck('actual static receiver public dimensions/up available',a.receiver_half_size_xy_m.shape==(2,) and np.array_equal(a.receiver_up_axis3,[0,0,1]))
    p=ROOT/'output/v15A_prospective_candidate_A_hold_v2/runs/nominal__ratio_effort__619731'
    with np.load(p/'trace.npz',allow_pickle=False) as z:t=z['time'];obs=z['observation_json']
    selected=np.flatnonzero((t>=11.75-1e-9)&(t<=12.02+1e-9))
    packets={int(k):SimpleNamespace(**json.loads(str(obs[k]))) for k in selected}
    samples=[];prior=None
    for k in selected:
        f=packets[int(k)];h=a.observe(f);samples.append(h)
        if abs(f.time-12)<1e-9:prior=a.begin_release(f.time)
    ck('archived public signed hand samples become coacquired valid',sum(h.valid for h in samples)>0)
    ck('motor prior exactly pre-request and frozen',prior is not None and prior.reference_time_s<12 and a.begin_release(12) is prior)
    ck('post-request motor invalid not recycled as fresh prior',a.release_request_s==12 and a.pre_release_prior is prior)
    f=packets[int(selected[-1])];h=a.observe(f)
    before=(len(a._prior_history),len(a._order),len(a._samples))
    ck('same tick duplicate reuses same hand object',a.observe(f) is h and before==(len(a._prior_history),len(a._order),len(a._samples)))
    changed=deepcopy(f);changed.support_axis=[1,0,0]
    rejects('changed same-tick acquired payload rejected',lambda:a.observe(changed))
    rejects('release cannot be retimed',lambda:a.begin_release(12.005))
    # Real current snapshot + legal real control-box intersection only.
    current=np.asarray(f.robot_reference21,float);desired=np.asarray(f.release_nominal_reference21,float)
    lower=np.maximum(np.asarray(f.reference_lower21)-current,-np.r_[[3.]*18,.18,.12,.12]*.02)
    upper=np.minimum(np.asarray(f.reference_upper21)-current,np.r_[[3.]*18,.18,.12,.12]*.02)
    g=a.current_geometry_tasks(f,current,desired,lower,upper)
    ck('real packet geometry honestly experimental or unavailable, never certified',
       g.status in ('EXPERIMENTAL_GEOMETRY_TASKS','UNAVAILABLE_GEOMETRY_TASKS') and not g.diagnostics['force_safety_certified'],
       dict(status=g.status,reason=g.diagnostics.get('adapter_reason'),patches=g.diagnostics.get('observed_patch_count')))
    ck('known actual standard geometry path is exercised',g.status=='EXPERIMENTAL_GEOMETRY_TASKS',g.diagnostics.get('adapter_reason'))
    ck('actual legal bounds remain unchanged',np.array_equal(g.step_lower21,lower) and np.array_equal(g.step_upper21,upper))
    ck('actor only encode no write',a.actor.write_count==0)
    wrong=current.copy();wrong[0]+=.001
    ck('non-authoritative current ref unavailable',a.current_geometry_tasks(f,wrong,desired,lower,upper).normal_matrix_m_per_reference is None)
    fbad=deepcopy(f);fbad.robot_actuator_target21=list(f.robot_actuator_target21);fbad.robot_actuator_target21[0]+=.01
    ck('engine/reference mismatch unavailable',a.current_geometry_tasks(fbad,current,desired,lower,upper).normal_matrix_m_per_reference is None)
    fbad=deepcopy(f);fbad.valid[2]=False
    ck('partial pressure geometry not mistaken for absent contact',a.current_geometry_tasks(fbad,current,desired,lower,upper).normal_matrix_m_per_reference is None)
    raw=f.receiver_surface
    accepted={x.name for x in fields(ReceiverMeasurement)}
    receiver=ReceiverMeasurement(**{k:v for k,v in raw.items() if k in accepted})
    ti=a.transfer_inputs(receiver)
    ck('typed existing receiver no re-acquisition',ti['receiver'] is receiver and ti['hand'] is a.hand_sample)
    ck('bounding spheres report only possible/clear with coacquired robot time',ti['hand_receiver_confounded'] in (True,False) and abs(ti['geometry_acquisition_time_s']-a.hand_sample.acquisition_time_s)<1e-9,a.last_clearance_diagnostics)
    b=new();b.observe(f)
    ck('no explicit receiver definition remains unknown',b.transfer_inputs(receiver)['hand_receiver_confounded'] is None)
    rejects('receiver dictionary does not masquerade as typed input',lambda:a.transfer_inputs(raw))
    oldgroup=int(model.actuator_group[a.servo.aids[0]])
    model.actuator_group[a.servo.aids[0]]=oldgroup+1
    ck('robot model group mutation invalidates geometry',a.current_geometry_tasks(f,current,desired,lower,upper).normal_matrix_m_per_reference is None)
    model.actuator_group[a.servo.aids[0]]=oldgroup
    # Fixed and affine algebra independently closed-form, including nonzero g1.
    G=np.zeros((21,3));B=np.zeros((21,3));G[:,0]=2.;B[:,1]=-2.
    D,d=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,length21=np.zeros(21),speed21=np.zeros(21),engine21=np.zeros(21))
    ck('fixed position gain yields explicit Dref one',np.array_equal(D,np.ones(21)))
    G[:,1]=.5;G[:,2]=.25;length=np.ones(21)*.3;speed=np.ones(21)*.2;engine=np.ones(21)*.4
    D,d=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,length21=length,speed21=speed,engine21=engine)
    arithmetic_allowance=16*np.finfo(float).eps*max(1.,2.2/1.8)
    ck('affine gain denominator includes g1*u, not bias alone',np.allclose(D,2.2/1.8,atol=arithmetic_allowance,rtol=0),
       dict(max_difference=float(np.max(abs(D-2.2/1.8))),floating_operation_allowance=arithmetic_allowance))
    G[:,0]=0;G[:,1:]=0
    D,d=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,length21=length,speed21=speed,engine21=engine)
    ck('known zero reference influence distinct from unknown',D is not None and np.array_equal(D,np.zeros(21)) and all(d['known_zero_reference_influence21']))
    B[0,1]=0
    D,d=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,length21=length,speed21=speed,engine21=engine)
    ck('zero denominator unknown whole map, no epsilon',D is None and not d['known21'][0])
    rejects('numeric strings not accepted',lambda:affine_equilibrium_length_derivative(gain21x3=np.full((21,3),'1'),bias21x3=B,length21=length,speed21=speed,engine21=engine))
    # Coacquired up counterexample. Current up rotates but delayed force does not.
    q=np.asarray(f.robot_position25,float)
    def fixture(now,up,acq):
        M=np.zeros((6,5));M[2,0]=1
        return SimpleNamespace(time=now,support_axis=up,robot_position25=q,sample_time=acq,age=now-acq,
            normal=np.array([1.,0,0,0,0]),normal_wrench_map=M,finger_non_normal_by_finger=np.zeros((5,6)),
            tangent_basis=np.tile(np.array([[1.,0,0],[0,1.,0]]),(5,1,1)),auxiliary_wrench=np.zeros(6),
            valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),auxiliary_valid=True,auxiliary_age=now-acq,
            auxiliary_uncertainty=0.,sensor_characteristics=dict(force_sigma_N=0.,tangential_sigma_N=0.),
            load_estimate=1.,load_uncertainty=.1,load_valid=True)
    c=new();c.observe(fixture(0,[0,0,1],0));c.observe(fixture(.005,[0,0,1],0));hh=c.observe(fixture(.01,[1,0,0],0))
    ck('transport uses acquisition up, not current rotated up',hh.valid and hh.signed_support_N==1.)
    gap=new();gap.observe(fixture(0,[0,0,1],0));missing=gap.observe(fixture(.02,[1,0,0],.01))
    ck('missing acquisition up unknown not current-up/zero substitute',not missing.valid and missing.signed_support_N is None)
    auxbad=new();ff=fixture(0,[0,0,1],0);ff.auxiliary_age=.01
    hh=auxbad.observe(ff)
    ck('auxiliary clock mismatch explicit unknown',not hh.valid and hh.signed_support_N is None)
    ck('forbidden pipelines never called',not forbidden,forbidden)
    source=ROOT/'v16_release_geometry/public_adapter.py'
    doc=dict(schema='v16_public_adapter_author_zero_physics_v1',passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_trace_sha256=hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest(),
        scope=dict(physics_steps=0,forward_calls=0,contact_force_queries=0,sensor_samples=0,RNG_draws=0,writer_calls=0,
                   model_definition_constructed_once=True,private_robot_FK_and_transmission_only=True,
                   no_object_state_mass_friction_or_target_identity_input=True,
                   not_servo_pressure_or_delivery_certification=True),
        actual_geometry=g.diagnostics,clearance=a.last_clearance_diagnostics)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=doc['passed'],checks=len(checks),path=str(out),source=doc['source_sha256'],failed=[c['name'] for c in checks if not c['passed']]),ensure_ascii=False))
    return 0 if doc['passed'] else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

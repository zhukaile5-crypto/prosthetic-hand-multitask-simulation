"""Seven bounded independent sensor/projection/clock gates; no model/physics."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch
import hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from sensors import HandSensors
from contracts import SensorNoise
from v16_release_geometry.public_adapter import PublicReleaseAdapter
from v17_contact_transfer.motion_domain_sensors import MotionDomainSensors,ADDED_FIELDS,SLIP_DOMAIN_ROLE
from v17_contact_transfer.motion_adapter import PublicMotionAdapter,FingerMotionProjection,build_motion_task


def same(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)):return struct.pack('d',a)==struct.pack('d',b)
    return type(a)==type(b) and a==b


def main(out):
    checks=[]
    def ck(n,v,details=None):checks.append(dict(name=n,passed=bool(v),details=details))
    cal=dict(force_matrix=np.eye(5)*.9,force_tau=.006,reference_alpha=np.full(5,.35),response_error_bound=np.full(5,.1),nominal_force_weights=np.array([1.2,1.,.9,.8,1.1]),effort_bias_N=0.,load_error_bound_N=.4)
    noise=SensorNoise(delay=.01,force_sigma=.004,tangential_sigma=.002,position_sigma=.0003,slip_sigma=.0005,dropout_windows=((.01,.015,(2,)),))
    def raw(t,level):
        return dict(time=t,patches=[dict(digit=i,normal=float(n),position=np.array([.01*i,0.,0]),normal_dir=np.array([0.,0.,1.]),force=np.array([0.,0.,n]),torque=np.zeros(3),slip_velocity=np.array([.013,-.007,0.]),is_target=False) for i,n in enumerate(level)],
            joint=np.ones(5)*.35,joint_velocity=np.zeros(5),alpha=np.ones(5)*.35,closure_min=np.ones(5)*-.2,closure_max=np.ones(5)*1.2,closure_permission=np.ones(5,bool),raw_joint=np.zeros(24),raw_joint_velocity=np.zeros(24),lift=.05,lift_ref=.07,lift_velocity=0.,motor_effort=2.,bare_hand_weight=1.,known_hand_inertia=.5,lift_damping=3.,support_axis=np.array([0.,0.,1.]),phase='HOLDING')
    a=MotionDomainSensors(HandSensors(cal,noise,seed=823));b=HandSensors(cal,noise,seed=823)
    levels=[np.roll(np.array([.019,.02,np.nextafter(.02,np.inf),.035,.1]),i) for i in range(8)]
    fields_ok=rng_ok=input_ok=mask_ok=True
    for i,level in enumerate(levels):
        r=raw(i*.005,level);before=deepcopy(r);ra=deepcopy(r);rb=deepcopy(r)
        fa=a.sample(ra);fb=b.sample(rb)
        fields_ok &= same({k:v for k,v in vars(fa).items() if k not in ADDED_FIELDS},vars(fb))
        rng_ok &= same(a.rng.bit_generator.state,b.rng.bit_generator.state)
        input_ok &= same(ra,before) and same(rb,before)
        mask_ok &= np.array_equal(fa.slip_valid5,(levels[max(0,i-2)]>.02)&fb.valid)
    ck('real sensor originals/RNG/input-purity across delay, threshold and dropout',fields_ok and rng_ok and input_ok and mask_ok,dict(fields=fields_ok,rng=rng_ok,raw_inputs_unchanged=input_ok,delayed_threshold_mask=mask_ok))
    rng=deepcopy(a.rng.bit_generator.state);n=a.samples;f1=a.sample(raw(.035,[9]*5));f1.slip_valid5[:]=False
    duplicate=(a.samples==n and same(rng,a.rng.bit_generator.state) and np.any(a.sample(raw(.035,[0]*5)).slip_valid5))
    advanced=HandSensors(cal,noise,seed=823);advanced.sample(raw(0,[.1]*5));advanced.sample(raw(.005,[.1]*5));late=MotionDomainSensors(advanced).sample(raw(.01,[.1]*5))
    ck('duplicate no draws; already-advanced missing domain remains unknown',duplicate and not late.slip_domain_available and not late.slip_valid5.any())
    # Independent scalar component oracle for two different projection planes.
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
    H=np.zeros((25,21));H[:21]=np.diag(np.linspace(.4,1.4,21))
    B=np.array([[1.,0,0],[0,1.,0]])
    J=np.zeros((3,25));J[0,0]=1.7;J[1,1]=-2.3;J[2,2]=.9
    ps=[FingerMotionProjection(0,B,[.11,-.03,.02],J,[.017,-.009]),FingerMotionProjection(3,np.array([[0.,1,0],[0,0,1.]]),[-.04,.02,.08],J*.7,[-.004,.003])]
    task=build_motion_task(time_s=.02,acquisition_time_s=.01,rotation_acquisition3x3=R,reference_to_robot25x21=H,projections=ps)
    matrix=np.zeros((4,21));target=np.zeros(4)
    for i,p in enumerate(ps):
        for r in range(2):
            y=sum(p.tangent_basis2x3[r,c]*R[w,c]*p.acquired_robot_velocity_world3[w] for c in range(3) for w in range(3))+p.slip2_m_s[r]
            target[2*i+r]=-.02*y
            for u in range(21):matrix[2*i+r,u]=sum(p.tangent_basis2x3[r,c]*R[w,c]*p.current_robot_jacobian3x25[w,q]*H[q,u] for c in range(3) for w in range(3) for q in range(25))
    bound=128*np.finfo(float).eps*max(1.,float(abs(matrix).max()),float(abs(target).max()))
    ck('two-per-digit world-projection sign and m/s-to-m independent scalar oracle',np.max(abs(task.matrix_Mx21-matrix))<=bound and np.max(abs(task.target_M-target))<=bound,dict(arithmetic_bound=bound))
    Q=np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])
    rotated=[FingerMotionProjection(p.digit,p.tangent_basis2x3,Q@p.acquired_robot_velocity_world3,Q@p.current_robot_jacobian3x25,p.slip2_m_s) for p in ps]
    t2=build_motion_task(time_s=.02,acquisition_time_s=.01,rotation_acquisition3x3=Q@R,reference_to_robot25x21=H,projections=rotated)
    try:build_motion_task(time_s=.02,acquisition_time_s=.01,rotation_acquisition3x3=R,reference_to_robot25x21=H,projections=[ps[0],ps[0]]);duplicate_rejected=False
    except ValueError:duplicate_rejected=True
    ck('world rotation covariance; duplicate slip pairs rejected not new independent observations',np.max(abs(task.matrix_Mx21-t2.matrix_Mx21))<=bound and np.max(abs(task.target_M-t2.target_M))<=bound and duplicate_rejected and not task.metadata['external_rigid_body_twist_available'] and not task.metadata['complete_error_covariance_available'])
    # Current-geometry dispatch with explicit fake robot FK, not a plant/model.
    base=SimpleNamespace(status='UNAVAILABLE_GEOMETRY_TASKS',diagnostics=dict(parent_reason='independent partial fixture'))
    # dataclasses.replace requires the real frozen type.
    from v16_release_geometry.geometry_tasks import GeometryTasks
    base=GeometryTasks('UNAVAILABLE_GEOMETRY_TASKS',None,None,None,None,None,np.zeros(21),-np.ones(21),np.ones(21),dict(parent_reason='independent partial fixture'))
    adapter=object.__new__(PublicMotionAdapter)
    adapter._motion_samples={2:dict(time_s=.01,q=np.zeros(25),v=np.zeros(25),R=R,reason=None)};adapter._motion_task=None
    Jold=J.copy();Jcur=J*1.2
    def fk(sample,p):return SimpleNamespace(point_velocity_world=np.array([.11,-.03,.02])+(p.link_point[0]*np.array([.02,.03,0.])),jacobian3x25=Jcur+(p.link_point[0]*np.eye(3,25)))
    adapter.geometry=SimpleNamespace(raw_joint_names=tuple('rh_'+str(i) for i in range(24)),point_kinematics=fk)
    adapter._motion_mapping=lambda *args:(SimpleNamespace(H25x21=H,robot_axes25=tuple('q'+str(i) for i in range(25)),reference_axes21=tuple('r'+str(i) for i in range(21))),dict(role='fake typed public algebra'),SimpleNamespace(time_s=.02))
    def patchrow(x,n):return dict(digit=0,robot_body=1,link_point=[x,0,0],link_normal=[0,0,1],sample_time=.01,reference_position=[0,0,0],reference_normal=[0,0,1],normal=n,age=.01)
    f=SimpleNamespace(time=.02,sample_time=.01,age=.01,slip_domain_role=SLIP_DOMAIN_ROLE,slip_domain_available=True,slip_domain_time_s=.02,slip_domain_acquisition_time_s=.01,
        slip_valid5=np.array([True,False,False,False,False]),valid=np.ones(5,bool),tangent_basis=np.tile(B,(5,1,1)),slip_local=np.tile([.017,-.009],(5,1)),skin_patches=[patchrow(.1,.3),patchrow(.2,.7)])
    args=(f,np.zeros(21),np.zeros(21),-np.ones(21),np.ones(21))
    with patch.object(PublicReleaseAdapter,'current_geometry_tasks',return_value=base):
        got=adapter.current_geometry_tasks(*args);original=adapter.motion_task
        g=deepcopy(f);g.skin_patches=[patchrow(.1,.1),patchrow(.1,.2),patchrow(.2,.7)]
        adapter.current_geometry_tasks(g,*args[1:]);split=adapter.motion_task
        ck('same digit patch subdivision preserves two rows and shared-measurement target',original.matrix_Mx21.shape==(2,21) and split.matrix_Mx21.shape==(2,21) and np.max(abs(original.matrix_Mx21-split.matrix_Mx21))<=bound and np.max(abs(original.target_M-split.target_M))<=bound)
        g=deepcopy(f);g.slip_domain_time_s=.025;adapter.current_geometry_tasks(g,*args[1:]);wrongclock=adapter.motion_task is None
        g=deepcopy(f);g.slip_valid5[:]=False;adapter.current_geometry_tasks(g,*args[1:]);no_measurement=adapter.motion_task is None
        keep=adapter._motion_samples;adapter._motion_samples={};adapter.current_geometry_tasks(f,*args[1:]);missing=adapter.motion_task is None;adapter._motion_samples=keep
        ck('clock mismatch/missing acquisition/all invalid yield None not stationary rows',wrongclock and no_measurement and missing)
        adapter.current_geometry_tasks(f,*args[1:]);ck('partial motion does not promote unavailable parent geometry or invent6D state',got.normal_matrix_m_per_reference is None and adapter.motion_task.matrix_Mx21.shape==(2,21) and not adapter.motion_task.metadata['normal_velocity_available'] and got.diagnostics['current_time_s']==.02)
    sources={n:hashlib.sha256((ROOT/'v17_contact_transfer'/n).read_bytes()).hexdigest() for n in ('motion_domain_sensors.py','motion_adapter.py')}
    report=dict(schema='v17_motion_independent_finite_v1',passed=all(c['passed'] for c in checks),check_count=len(checks),checks=checks,source_sha256=sources,
        scope=dict(models_constructed=0,FK_calls_on_actual_models=0,physics_steps=0,raw_archived_runs_read=0,
            real_existing_sensor_math_and_RNG_only=True,fake_robot_FK_for_dispatch_explicit=True,no_covariance_or_servo_or_force_certificate=True),
        static_limits=['Metadata is diagnostic provenance, not a proof that projections share one external rigid body.',
            'Independent device row is one2D pair per finger; patch splits do not add independent measurements.',
            'Masked slip zero is not stationary external motion; projection loss after separation remains unknown.'])
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),path=str(out),sha=hashlib.sha256(out.read_bytes()).hexdigest(),failed=[c['name'] for c in checks if not c['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

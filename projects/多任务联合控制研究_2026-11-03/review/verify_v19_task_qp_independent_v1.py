"""Independent small coupled reference/progress audit and one public native packet."""
from pathlib import Path
from itertools import combinations
from dataclasses import replace
from unittest.mock import patch
import argparse,gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from v19_contact_reference import task_qp as task
from v19_contact_reference.native_branch import NativeBranchBuilder21
from v13_a1.codec import Reference21,ReferenceActor21
from v11.robot_servo_effort import RobotServoEffort
from hand_plant import _model
from foundation_config import FoundationConfig

def fixture(P=0):
    return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),gain21=np.zeros(21),offset21=np.zeros(21),
        Jpx25=np.zeros((P,25)),y_p=np.ones(P)*.4,tau_extra25=np.zeros(25),delta_lower21=-np.ones(21),delta_upper21=np.ones(21),
        open_step21=np.zeros(21),motion_step21=np.zeros(21),unload_fraction=.5,patch_digit_p=np.arange(P,dtype=int)%5,
        caps5=np.ones(5)*3,total_budget=10.,gamma_lower=0.,gamma_upper=1.,
        scales=dict(residual25=np.ones(25),delta21=np.ones(21),reference21=np.ones(21),intentP=np.ones(P),progress=1.))

def residual_vector(p,u,z,gamma,actual=None):
    d=p['offset21']+p['gain21']*u if actual is None else actual
    r=p['A21x25'].T@d-(p['Jpx25'].T@(z-p['y_p'])+p['tau_extra25'])
    return np.concatenate((p['scales']['residual25']*r,p['scales']['delta21']*d,
        p['scales']['reference21']*(u-gamma*p['open_step21']-p['motion_step21']),
        p['scales']['intentP']*(z+gamma*p['unload_fraction']*p['y_p']-p['y_p']),
        [p['scales']['progress']*(gamma-1.)]))

def value(p,u,z,gamma,actual=None):
    v=residual_vector(p,u,z,gamma,actual);return float(v@v)

def four_dimensional_oracle(p):
    # Generic active inequalities, with total normal budget coupling the two
    # groups. Build an independent affine residual design from basis vectors.
    def unpack(x):
        u=np.zeros(21);u[0]=x[0];return u,x[1:3],x[3]
    zero=np.zeros(4);f0=residual_vector(p,*unpack(zero))
    E=np.column_stack([residual_vector(p,*unpack(np.eye(4)[i]))-f0 for i in range(4)])
    G=np.array([[1,0,0,0],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],
        [0,1,0,0],[0,0,1,0],[0,1,1,0],[0,0,0,-1],[0,0,0,1]],float)
    h=np.array([p['delta_upper21'][0],-p['delta_lower21'][0],0,0,p['caps5'][0],p['caps5'][1],
                p['total_budget'],-p['gamma_lower'],p['gamma_upper']])
    candidates=[]
    for count in range(5):
        for subset in combinations(range(len(h)),count):
            W=G[list(subset)] if count else np.empty((0,4))
            K=np.block([[E.T@E,W.T],[W,np.zeros((count,count))]])
            try:solution=np.linalg.solve(K,np.r_[-E.T@f0,h[list(subset)]])
            except np.linalg.LinAlgError:continue
            x=solution[:4]
            if np.all(G@x<=h+2e-12):candidates.append((value(p,*unpack(x)),x))
    return min(candidates,key=lambda x:x[0]),len(candidates)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    path=ROOT/'v19_contact_reference/task_qp.py';pin=hashlib.sha256(path.read_bytes()).hexdigest()
    if pin!='197d565bc639fbf1a69a47d60f490143ca1e72c35358d9c81e39b7f2c6c4a978':raise ValueError('unexpected task source')
    checks=[];details={}
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    # 1: unlike a native inverse, a zero gain preserves geometric progress.
    p=fixture(1);p['open_step21'][0]=.12;p['motion_step21'][0]=.015;p['gamma_upper']=.8;p['unload_fraction']=.7
    r=task.solve_task_qp(**p)
    ck('zero effort slope still jointly progresses reference and unload intent',r.available and abs(r.gamma-.8)<1e-10 and
       abs(r.du21[0]-(.015+.8*.12))<1e-10 and abs(r.zP[0]-.4*(1-.8*.7))<1e-10 and r.d21[0]==0.)
    # 2: force a real nonzero affine intercept with unequal physical scales.
    p=fixture();p['A21x25'][0,0]=1.;p['gain21'][0]=-2.;p['offset21'][0]=.3;p['tau_extra25'][0]=.2
    p['motion_step21'][0]=.05;p['gamma_lower']=p['gamma_upper']=0.
    p['scales']['residual25'][0]=1.3;p['scales']['delta21'][0]=.7;p['scales']['reference21'][0]=.5
    g=-2.;b=.3;expected=(.5**2*.05-g*(1.3**2*(b-.2)+.7**2*b))/(g*g*(1.3**2+.7**2)+.5**2)
    r=task.solve_task_qp(**p)
    ck('negative slope nonzero offset closed form with scales multiplying',r.available and abs(r.du21[0]-expected)<1e-12 and abs(r.d21[0]-(b+g*expected))<1e-12)
    # 3: one four-dimensional global finite-face oracle, with two cap groups
    # and a tighter aggregate budget (not the author's box-only 3D oracle).
    p=fixture(2);p['delta_lower21'][:]=p['delta_upper21'][:]=0.;p['delta_lower21'][0]=.04;p['delta_upper21'][0]=.18
    p['A21x25'][0,0]=1.2;p['gain21'][0]=-1.7;p['offset21'][0]=.25;p['Jpx25'][:,0]=[.8,-.6]
    p['y_p'][:]=[.6,.4];p['tau_extra25'][0]=.13;p['open_step21'][0]=.15;p['motion_step21'][0]=.025
    p['unload_fraction']=.65;p['caps5'][:2]=[.4,.25];p['total_budget']=.45;p['gamma_lower']=.1;p['gamma_upper']=.7
    p['scales']['residual25'][0]=1.3;p['scales']['delta21'][0]=.6;p['scales']['reference21'][0]=4.
    p['scales']['intentP'][:]=[.9,1.1];p['scales']['progress']=.7
    (expected_value,x),count=four_dimensional_oracle(p);r=task.solve_task_qp(**p)
    ck('independent4D active faces include shared budget and no zero anchor',r.available and
       np.max(abs(np.r_[r.du21[0],r.zP,r.gamma]-x))<1e-9 and abs(r.objective-expected_value)<1e-11,
       dict(candidates=count,expected=x.tolist(),objective=expected_value))
    # 4: FD all25 variables independently, without using production gradients.
    rng=np.random.default_rng(719064);p=fixture(3);p['gain21']=rng.normal(size=21);p['offset21']=rng.normal(size=21)*.05
    p['A21x25']=rng.normal(size=(21,25))*.1;p['Jpx25']=rng.normal(size=(3,25))*.1;p['tau_extra25']=rng.normal(size=25)*.1
    p['open_step21']=rng.normal(size=21)*.1;p['motion_step21']=rng.normal(size=21)*.02
    x=np.r_[rng.normal(size=21)*.03,[.1,.2,.3],.35];ev=task.evaluate_task(p,x[:21],x[21:24],x[-1]);fd=[]
    for i in range(25):
        a=x.copy();b=x.copy();a[i]+=1e-6;b[i]-=1e-6
        fd.append((value(p,a[:21],a[21:24],a[-1])-value(p,b[:21],b[21:24],b[-1]))/2e-6)
    err=float(np.max(abs(np.r_[ev['gradient_du21'],ev['gradient_zP'],ev['gradient_gamma']]-fd)))
    ck('independent all25 finite difference gradient',err<2e-9,err)
    # 5: SEARCH does not erase motion when progress gamma is fixed0.
    p=fixture();p['motion_step21'][18]=-.0007;p['open_step21'][0]=.01;p['gamma_lower']=p['gamma_upper']=0.;r=task.solve_task_qp(**p)
    ck('fixed gamma0 preserves declared motion target without OPEN',r.available and r.gamma==0. and abs(r.du21[18]+.0007)<1e-12 and r.du21[0]==0.)
    # 6: no command authority cannot silently absorb an unmatched effort.
    p=fixture();p['delta_lower21'][:]=p['delta_upper21'][:]=0.;p['tau_extra25'][24]=.6;r=task.solve_task_qp(**p)
    ck('fixed references retain unreachable signed generalized residual',r.available and r.residual25[24]==-.6 and np.all(r.du21==0.))
    # 7: p0 is bookkeeping, never used to rederive and round away d.
    p=fixture();p['gain21'][0]=2.;p['A21x25'][0,0]=1.;p['tau_extra25'][0]=.3;r1=task.solve_task_qp(**p)
    p['p0_21'][0]=2.**54;r2=task.solve_task_qp(**p)
    ck('huge raw p0 does not retarget sub-ULP delta',r1.d21.tobytes()==r2.d21.tobytes() and r1.du21.tobytes()==r2.du21.tobytes() and r2.p21[0]==p['p0_21'][0])
    # 8: unknown normal or effort remains unknown, invalid signs are refused.
    p=fixture(1);p['y_p'][0]=-.1;a=task.solve_task_qp(**p)
    p=fixture();p['tau_extra25']=None;b=task.solve_task_qp(**p)
    p=fixture();p['unload_fraction']=None;c=task.solve_task_qp(**p)
    ck('signed-normal invalid and unknown efforts/progress not zero filled',a.status=='invalid_input' and b.status=='unknown' and c.status=='unknown')
    # 9: wrong objective/affine delta cannot pass same-goal merit.
    p=fixture();p['gain21'][0]=1.;p['A21x25'][0,0]=1.;p['open_step21'][0]=.2;r=task.solve_task_qp(**p)
    bad=replace(r,d21=r.d21+np.eye(21)[0]*.01)
    ck('same-problem gate rejects retargeted delta and objective',not task.check_same_problem_candidate(p,bad)['accepted'] and
       not task.check_same_problem_candidate(p,replace(r,objective=r.objective+.001))['accepted'])
    # 10--12: one actual archived PUBLIC q/v/u/TASK interval; current native
    # value is recomputed independently, no plant/solver/integration replay.
    run=ROOT/'output/v18_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';at=13.935
    with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];encoded=z['observation_json']
    obs=json.loads(str(encoded[int(np.argmin(abs(times-at)))]))
    with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:
        saved=next(json.loads(line) for line in f if abs(json.loads(line)['time_s']-at)<1e-9)
    model=_model(FoundationConfig())[0];guards=[patch('mujoco.'+n,side_effect=RuntimeError('forbidden world/physics pipeline')) for n in
        ('mj_step','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce')]
    guards.append(patch.object(ReferenceActor21,'apply',side_effect=RuntimeError('forbidden actor write')))
    for guard in guards:guard.start()
    try:
        builder=NativeBranchBuilder21(model);q=np.asarray(obs['robot_position25']);v=np.asarray(obs['robot_velocity25']);u=np.asarray(obs['robot_reference21'])
        branch=builder.build(q25=q,v25=v,robot_axes25=builder.robot_axes25,time_s=at,current_reference=Reference21(u,at),
            actual_engine=builder.actor.encode(Reference21(u,at)),preferred_reference=Reference21(saved['posture_reference21'],at),
            reference_lower21=saved['actual_reference_lower21'],reference_upper21=saved['actual_reference_upper21'])
        if not branch.available:raise ValueError('actual public branch failed: '+branch.reason)
        original=saved['qp_problem'];P=len(original['y_p']);p=fixture(P)
        p.update(p0_21=branch.p0_effort21,A21x25=branch.transmission21x25,gain21=branch.gain21,offset21=branch.offset21,
            Jpx25=np.asarray(original['Jpx25']).reshape((P,25)),y_p=np.asarray(original['y_p']),
            tau_extra25=np.asarray(original['tau_extra25']),delta_lower21=branch.delta_lower21,delta_upper21=branch.delta_upper21,
            patch_digit_p=np.asarray(original['patch_digit_p'],int),caps5=np.asarray(original['caps5']),total_budget=original['total_budget'])
        preferred=branch.preferred_reference21-u;p['open_step21'][:18]=preferred[:18];p['motion_step21'][18:]=preferred[18:]
        p['gamma_upper']=.5;p['unload_fraction']=.6
        p['scales']['residual25']=np.asarray(original['scales']['residual25']);p['scales']['delta21']=np.asarray(original['scales']['delta21'])
        p['scales']['reference21']=1./(np.r_[np.full(18,3.),.18,.12,.12]*.005);p['scales']['intentP']=np.asarray(original['scales']['intentP'])
        r=task.solve_task_qp(**p);merit=task.check_same_problem_candidate(p,r)
        ck('archived public native branch and new QP same-goal candidate accepted',r.du21 is not None and merit['accepted'],dict(status=r.status,reason=r.reason))
        proposal=np.minimum(branch.reference_upper21,np.maximum(branch.reference_lower21,u+r.du21))
        actual=builder.evaluate_reference(branch,Reference21(proposal,at));oracle=RobotServoEffort(model).evaluate(q,v,builder.actor.encode(Reference21(proposal,at)).values,time_s=at)
        ck('final fixed-qv native value independently F067 byte exact no inverse',actual['available'] and actual['native_effort21'].tobytes()==oracle.predicted_effort21.tobytes())
        du=proposal-u;real=oracle.predicted_effort21-branch.p0_effort21;score=task.evaluate_task(p,du,r.zP,r.gamma,actual_delta21=real)
        independent=value(p,du,r.zP,r.gamma,real)
        ck('actual native SAME du-z-gamma total cost and affine error retained',score['available'] and score['feasible'] and
           abs(score['objective']-independent)<1e-11*max(1.,abs(independent)) and
           score['affine_realization_error21'].tobytes()==(real-(p['offset21']+p['gain21']*du)).tobytes(),
           dict(total=score.get('objective'),independent_total=independent,affine_error_inf=float(np.max(abs(actual['affine_error21'])))))
        details['actual_public_packet']=dict(time_s=at,qp_status=r.status,accepted=merit['accepted'],gamma=r.gamma,
            raw_current_p0=branch.p0_effort21.tolist(),fixed_p0_preserved=True,
            branch_offset_nonzero_count=int(np.count_nonzero(branch.offset21)),actual_cost=score['objective'],
            projection_change_inf=float(np.max(abs(proposal-(u+r.du21)))),no_pressure_or_future_response_certificate=True)
    finally:
        for guard in reversed(guards):guard.stop()
    after=hashlib.sha256(path.read_bytes()).hexdigest()
    payload=dict(schema='v19_task_qp_independent_v1',passed=all(c['passed'] for c in checks) and pin==after,
        checks=checks,check_count=len(checks),details=details,
        source_sha256={'v19_contact_reference/task_qp.py':after,'v19_contact_reference/native_branch.py':hashlib.sha256((ROOT/'v19_contact_reference/native_branch.py').read_bytes()).hexdigest()},
        reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0,world_forward_or_FK_calls=0,contact_reads=0,actor_writes=0,
        scope='independent current task math + one actual archived public packet native arithmetic, not closed-loop performance')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],
                         out=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),ensure_ascii=False))
    return 0 if payload['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

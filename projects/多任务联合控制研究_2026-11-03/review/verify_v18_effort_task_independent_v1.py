"""Small independent current-task audit; no simulation or contact pipeline."""
from pathlib import Path
from itertools import combinations
from unittest.mock import patch
import argparse, copy, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from v18_effort_transfer import effort_task as task
from v14_coord.effort_domain import FastEffortGoal21
from v13_a1.codec import Reference21
from v11.robot_servo_effort import RobotServoEffort
from hand_plant import _model
from foundation_config import FoundationConfig


def fixture(P=0):
    return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),Jpx25=np.zeros((P,25)),
        y_p=np.zeros(P),z_intent_p=np.zeros(P),effort_lower21=-np.ones(21),effort_upper21=np.ones(21),
        patch_digit_p=np.zeros(P,dtype=int),caps5=np.ones(5),total_budget=5.,
        support_coeff_p=np.zeros(P),load_proxy_N=0.,tau_extra25=np.zeros(25),
        scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.))


def independent(p,d,z):
    # Do not call any production objective/evaluator in this oracle.
    change=p['A21x25'].T@d
    normal_change=p['Jpx25'].T@(z-p['y_p'])
    correction=normal_change+p['tau_extra25'];r=change-correction
    terms=[np.dot(p['scales']['residual25']*r,p['scales']['residual25']*r),
           np.dot(p['scales']['delta21']*d,p['scales']['delta21']*d),
           np.dot(p['scales']['intentP']*(z-p['z_intent_p']),p['scales']['intentP']*(z-p['z_intent_p'])),
           (p['scales']['load']*max(0.,p['load_proxy_N']-np.dot(p['support_coeff_p'],z)))**2]
    if p.get('extra_task_rows') is not None:
        e=p['extra_task_rows'];res=e['matrix']@np.concatenate((d,z))-e['target'];terms.append(np.dot(res,res))
    return float(sum(terms)),r


def two_dimensional_oracle(p):
    """Enumerate independent stationary faces of the two hinge polygons."""
    a=p['A21x25'][0,0];j=p['Jpx25'][0,0];sr=p['scales']['residual25'][0]
    sf=p['scales']['intentP'][0];sd=p['scales']['delta21'][0];sl=p['scales']['load']
    c=p['support_coeff_p'][0];load=p['load_proxy_N'];E=p['extra_task_rows']
    options=[]
    for hinge in (False,True):
        T=np.array([[sr*a,-sr*j],[sd,0.],[0.,sf],[E['matrix'][0,0],E['matrix'][0,21]]])
        b=np.array([sr*(p['tau_extra25'][0]-j*p['y_p'][0]),0.,sf*p['z_intent_p'][0],E['target'][0]])
        if hinge:T=np.vstack((T,[0.,sl*c]));b=np.r_[b,sl*load]
        G=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[0.,c if hinge else -c]])
        h=np.array([p['effort_upper21'][0],-p['effort_lower21'][0],p['caps5'][0],0.,load if hinge else -load])
        for count in range(3):
            for subset in combinations(range(5),count):
                W=G[list(subset)] if count else np.empty((0,2))
                K=np.block([[T.T@T,W.T],[W,np.zeros((count,count))]])
                try:sol=np.linalg.solve(K,np.r_[T.T@b,h[list(subset)]])[:2]
                except np.linalg.LinAlgError:continue
                if np.all(G@sol<=h+1e-12):
                    d=np.zeros(21);d[0]=sol[0];value,_=independent(p,d,np.array([sol[1]]));options.append((value,sol))
    return min(options,key=lambda x:x[0])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v18_effort_transfer/effort_task.py';before=hashlib.sha256(source.read_bytes()).hexdigest()
    if before!='64730a61667aa32637467365e32dc0a50e094ddf368f071cc65dc37d1e58763b':raise ValueError('unexpected task source')
    checks=[];details={}
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    p=fixture();p['A21x25'][0,0]=2.;p['tau_extra25'][0]=-.75
    p['scales']['residual25'][0]=1.5;p['scales']['delta21'][0]=.4
    result=task.solve_effort_task(**p)
    want=2.*1.5**2*(-.75)/(2.**2*1.5**2+.4**2)
    ck('nonzero signed rhs closed form with multiplicative scales',result.available and abs(result.d21[0]-want)<1e-12)
    p=fixture(1);p['A21x25'][0,0]=1.;p['Jpx25'][0,0]=.75;p['y_p'][0]=.3;p['z_intent_p'][0]=.4
    p['tau_extra25'][0]=-.2;p['effort_lower21'][0]=-.3;p['effort_upper21'][0]=.35;p['caps5'][0]=.45
    p['scales']['residual25'][0]=1.3;p['scales']['intentP'][0]=.8;p['scales']['delta21'][0]=.7
    p['support_coeff_p'][0]=.6;p['load_proxy_N']=.5;p['scales']['load']=.9
    E=np.zeros((1,22));E[0,0]=1.;E[0,21]=.4;p['extra_task_rows']=dict(matrix=E,target=np.array([-.1]))
    value,want=two_dimensional_oracle(p);result=task.solve_effort_task(**p)
    ck('coupled d-z hinge-cap-nonzero-soft-target independent face enumeration',result.available and
       np.max(abs(np.r_[result.d21[0],result.zP[0]]-want))<1e-10 and abs(result.objective-value)<1e-11,dict(expected=want.tolist(),objective=value))
    rng=np.random.default_rng(418020);p=fixture(2);p['A21x25']=rng.normal(size=(21,25))*.1
    p['Jpx25']=rng.normal(size=(2,25))*.1;p['tau_extra25']=rng.normal(size=25)*.2;p['z_intent_p'][:]=.5
    p['load_proxy_N']=1.;p['support_coeff_p'][:]=.3
    p['extra_task_rows']=dict(matrix=rng.normal(size=(3,23))*.2,target=np.array([.2,-.1,.4]))
    x=np.r_[rng.normal(size=21)*.05,.1,.2];ev=task.evaluate_effort_task(p,x[:21],x[21:]);fd=[]
    for i in range(len(x)):
        plus=x.copy();minus=x.copy();plus[i]+=2e-6;minus[i]-=2e-6
        fd.append((independent(p,plus[:21],plus[21:])[0]-independent(p,minus[:21],minus[21:])[0])/4e-6)
    error=float(np.max(abs(np.r_[ev['gradient_d21'],ev['gradient_zP']]-fd)))
    ck('all23 raw gradient finite differences independently include rhs and extra rows',error<2e-9,error)
    p=fixture(1);p['A21x25'][0,0]=1.;p['Jpx25'][0,0]=1.;p['tau_extra25'][0]=.7
    p['effort_lower21'][0]=p['effort_upper21'][0]=0.;p['z_intent_p'][0]=.4
    result=task.solve_effort_task(**p)
    ck('known singleton effort axis valid and unattainable rhs retained',result.available and result.d21[0]==0. and abs(result.residual25[0])>.6)
    p=fixture();p['tau_extra25'][24]=-.3;result=task.solve_effort_task(**p)
    ck('unreachable generalized direction remains residual, not fake zero gain',result.available and result.residual25[24]==.3)
    bad=fixture(1);bad['z_intent_p'][0]=-.1;ck('signed tangent cannot be injected as negative normal intent',task.solve_effort_task(**bad).status=='invalid_input')
    bad=fixture();bad['tau_extra25']=None;ck('unknown extra effort not zero filled',task.solve_effort_task(**bad).status=='unknown')
    # A real native-law fixture, not a contact/plant or command execution.
    model=_model(FoundationConfig())[0]
    forbidden=('mj_step','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce')
    guards=[patch('mujoco.'+n,side_effect=RuntimeError('forbidden world/physics pipeline')) for n in forbidden]
    for guard in guards:guard.start()
    try:
        native=FastEffortGoal21(model);q=model.qpos0[native.servo.qadr].copy();v=np.zeros(25);q[24]=.08;v[24]=-.02
        ref=np.zeros(21);ref[18]=.08;now=12.02;current=Reference21(ref,now)
        domain=native.build_domain(q25=q,v25=v,robot_axes25=native.robot_axes25,time_s=now,
            current_reference=current,nominal_reference=current,actual_engine=native.actor.encode(current),
            reference_lower21=ref-.005,reference_upper21=ref+.005)
        ck('actual F067 reference/effort domain available',domain.available,domain.reason)
        p=fixture(1);p['p0_21']=domain.p0_effort21;p['A21x25']=domain.transmission21x25
        p['Jpx25']=np.asarray([domain.transmission21x25[0]]);p['y_p'][0]=.3;p['z_intent_p'][0]=.1
        p['effort_lower21']=domain.effort_lower21;p['effort_upper21']=domain.effort_upper21
        p['tau_extra25']=.25*domain.transmission21x25[1]-.2*domain.transmission21x25[18]
        E=np.zeros((1,22));E[0,1]=.2;E[0,21]=.4;p['extra_task_rows']=dict(matrix=E,target=np.array([.3]))
        result=task.solve_effort_task(**p);merit=task.check_same_problem_candidate(p,result)
        correction=task.native_correction25(p,result.zP)
        inverse=native.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=result.d21,
            correction25=correction,residual_scale25=p['scales']['residual25'],delta_scale21=p['scales']['delta21'],requested_target_effort21=result.p21)
        ck('same-goal QP candidate and actual native realization available',merit['accepted'] and inverse.available,
           dict(qp=result.status,native=inverse.status))
        oracle=RobotServoEffort(model).evaluate(q,v,inverse.engine_port21.values,time_s=now)
        ck('final native21 effort independently full F067 byte equal',oracle.predicted_effort21.tobytes()==inverse.native_effort21.tobytes())
        actual=oracle.predicted_effort21-domain.p0_effort21;raw_value,raw_residual=independent(p,actual,result.zP)
        score=task.score_native_realization(p,original_d21=result.d21,zP=result.zP,native_result=inverse)
        ck('native canonical correction/residual preserve fixed original p0 and requested delta',
           inverse.p0_effort21.tobytes()==p['p0_21'].tobytes() and inverse.requested_delta_effort21.tobytes()==result.d21.tobytes() and
           inverse.residual25.tobytes()==raw_residual.tobytes() and correction.tobytes()==(p['Jpx25'].T@(result.zP-p['y_p'])+p['tau_extra25']).tobytes())
        ck('final total score adds normal and extra tasks to native mechanical score',score['available'] and score['feasible'] and
           abs(score['objective']-raw_value)<1e-12 and score['objective']>inverse.objective,
           dict(total=score.get('objective'),independent=raw_value,mechanical_only=inverse.objective))
        details['actual_native']=dict(qp_status=result.status,native_status=inverse.status,
            realization_error_linf=float(np.max(abs(inverse.actual_delta_effort21-result.d21))),
            full_native_recheck=inverse.diagnostics['prepared_value_final_full_recheck'],native_mechanical_objective=inverse.objective,total_task_objective=score.get('objective'))
    finally:
        for guard in reversed(guards):guard.stop()
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    payload=dict(schema='v18_effort_task_independent_v1',passed=all(c['passed'] for c in checks) and before==after,
       checks=checks,check_count=len(checks),details=details,source_sha256={str(source.relative_to(ROOT)).replace('\\','/'):after},
       physics_steps=0,world_forward_or_FK_calls=0,contact_reads=0,robot_model_compile_only=True,
       scope='current effort task arithmetic and real native realization; no contact-force/future-response or controller performance certificate')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],out=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),ensure_ascii=False))
    return 0 if payload['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

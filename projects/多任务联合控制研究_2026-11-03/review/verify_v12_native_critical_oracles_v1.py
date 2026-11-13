"""Four independent actual-solve oracles, with explicit algebra-only fake servo.

No real model/data construction or dynamics/FK/contact calls. The actual frozen
tracker is imported; its solve/_port/failure code runs unchanged. Synthetic
native effort uses an independently specified floating operation sequence.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,importlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='160eb7129237d695240d0639a6562dd7fb21c6bacbaea1c06414cf61ea098915'
OUT=ROOT/'review/v12_native_critical_oracles_independent_v1.json'

def main():
    path=ROOT/'v12_native/effort_tracker.py';payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==EXPECTED
    sys.path.insert(0,str(ROOT));import mujoco
    forbidden=('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_fwdActuation','mj_kinematics','mj_comPos',
               'mj_tendon','mj_transmission','mj_fwdPosition','mj_fwdVelocity','mj_sensorPos','mj_contactForce')
    counts={name:0 for name in forbidden};original={}
    def guard(name):
        def reject(*args,**kwargs):counts[name]+=1;raise AssertionError('Forbidden physics/FK/native pipeline call '+name)
        return reject
    for name in forbidden:
        if hasattr(mujoco,name):original[name]=getattr(mujoco,name);setattr(mujoco,name,guard(name))
    cases=[];checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    try:
        module=importlib.import_module('v12_native.effort_tracker');Tracker=module.EffortTracker
        groups=(5,3,3,3,4);starts=np.r_[0,np.cumsum(groups)]
        C=np.zeros((21,8));opens=[];closes=[]
        for i,count in enumerate(groups):
            o=.03125+np.arange(count)*.015625;d=.5+np.arange(count)*.0625
            opens.append(o);closes.append(o+d);C[starts[i]:starts[i+1],i]=d
        C[18:21,5:]=np.eye(3);offset=np.r_[np.concatenate(opens),.02481276,0.,0.]
        A=np.zeros((21,25));A[:18,:18]=np.eye(18);A[18,24]=1.;A[19,18]=1.;A[20,19]=1.
        gains=np.array([8.,11.,7.,13.,9.,12.,10.,6.,14.,9.,11.,7.,8.,13.,10.,12.,6.,9.,1500.,15.,17.])
        class NativeOracle:
            def __init__(self):
                self.aids=self.ctrladr=np.arange(21);self.ctrl_limited=np.ones(21,bool);self.ctrl_range=np.tile([-2.,2.],(21,1))
                self.force_limited=np.zeros(21,bool);self.force_range=np.tile([-10000.,10000.],(21,1))
                self.model=NS(actuator_gainprm=np.column_stack((gains,np.zeros((21,2)))),actuator_gaintype=np.full(21,int(mujoco.mjtGain.mjGAIN_FIXED)))
                self.unsupported=False;self.accepted_ports=None;self.evaluations=0
            def evaluate(self,q,v,u):
                self.evaluations+=1;u=np.asarray(u,float)
                if self.unsupported or self.accepted_ports is not None and u.tobytes() not in self.accepted_ports:
                    return NS(status='UNAVAILABLE_MODEL',reasons=('synthetic_unknown_native_value',))
                length=A@q;velocity=A@v
                # Explicit native sequence differs from beta-extraction arithmetic.
                raw=gains*u + (-gains)*length + (-.3)*velocity
                p=np.where(self.force_limited,np.clip(raw,self.force_range[:,0],self.force_range[:,1]),raw)
                at_kink=self.force_limited&((raw==self.force_range[:,0])|(raw==self.force_range[:,1]))
                clipped=self.force_limited&((raw<self.force_range[:,0])|(raw>self.force_range[:,1]))
                return NS(status='KNOWN_ROBOT_ACTUATION_ALGEBRA',reasons=(),transmission21x25=A.copy(),
                    generalized_actuator_effort25=A.T@p,actuator_length21=length,actuator_velocity21=velocity,
                    unbounded_effort21=raw,predicted_effort21=p,effective_target21=u.copy(),
                    diagnostics=dict(control_clipped21=[False]*21,force_clipped21=clipped.tolist()),
                    derivative_status='FORCE_KINK_UNKNOWN_DERIVATIVE' if at_kink.any() else 'KNOWN',Ju=None if at_kink.any() else A.T@gains,
                    K=None if at_kink.any() else np.zeros((25,25)),D=None if at_kink.any() else np.zeros((25,25)))
        def tracker():
            t=Tracker.__new__(Tracker);t.servo=NativeOracle();t.C=C.copy();t.port_offset=offset.copy();t.port_order_matches=True
            t.basis=NS(open_pose=[x.copy() for x in opens],closed_pose=[x.copy() for x in closes],lift_motor_offset=offset[18],
                held_lower8=np.array([0.]*5+[-.2,-.5,-.5]),held_upper8=np.array([1.]*5+[.2,.5,.5]))
            t.rows=tuple(np.flatnonzero(t.C[:,i]) for i in range(8));return t
        def inputs(t):
            current=np.array([.43]*5+[.05,.01,-.02]);q=np.linspace(.11,.41,25);v=np.linspace(-.003,.004,25)
            return dict(q25=q,v25=v,current_reference8=current,actual_target21=t._port(current),desired_torque_correction25=np.zeros(25),
                nominal_motion_reference3=current[5:].copy(),step_lower8=np.full(8,-.02),step_upper8=np.full(8,.02),
                motion_reference_bounds=dict(lower=[-.2,-.5,-.5],upper=[.2,.5,.5]),tracking_tolerances3=[None]*3)

        # 1: cancellation-sensitive nonzero objective, fixed goal and byte checks.
        t=tracker();kw=inputs(t);kw['nominal_motion_reference3']+=np.array([.001,.003,-.002])
        kw['desired_torque_correction25'][:18]=np.linspace(-2e-9,3e-9,18);kw['desired_torque_correction25'][22]=.007
        nominal=np.r_[kw['current_reference8'][:5],kw['nominal_motion_reference3']]
        target=t.servo.evaluate(kw['q25'],kw['v25'],t._port(nominal)).generalized_actuator_effort25+kw['desired_torque_correction25']
        r=t.solve(**kw);check('case1_available',r.available,r.status)
        check('case1_original_target_exact',r.desired_torque25.tobytes()==target.tobytes())
        native=t.servo.evaluate(kw['q25'],kw['v25'],r.port_target21)
        pnom=t.servo.evaluate(kw['q25'],kw['v25'],t._port(nominal)).predicted_effort21
        residual=A.T@(native.predicted_effort21-pnom)-kw['desired_torque_correction25']
        check('case1_independent_canonical_final_bytes',residual.tobytes()==r.residual25.tobytes())
        check('case1_all_scalar_final_bytes',all(x['residual_bytes_equal'] and x['loss_bytes_equal'] for x in r.debug['scalar_native_recheck']))
        differences=[abs(c['loss']-c['analytic_loss_diagnostic_only']) for s in r.debug['scalar_solutions'] for c in s['candidates']]
        check('case1_native_is_not_analytic_score_authority',max(differences)>0.,max(differences))
        check('case1_orthogonal_unknown_control_not_zero_residual',r.residual25[22]==-.007)
        # Same oracle also checks the exact no-correction nominal decoder path.
        zero=inputs(t);zero['nominal_motion_reference3']+=np.array([.001,.003,-.002]);rz=t.solve(**zero)
        expected_ref=np.r_[zero['current_reference8'][:5],zero['nominal_motion_reference3']]
        check('case1_zero_correction_exact_nominal',rz.available and rz.reference8.tobytes()==expected_ref.tobytes() and rz.port_target21.tobytes()==t._port(expected_ref).tobytes())
        cases.append(dict(id='native_cancellation_same_goal_and_bytes',passed=all(x['passed'] for x in checks),
            native_analytic_score_max_difference=max(differences),native_candidate_evaluations=r.debug['native_candidate_evaluations'],
            native_final_bytes_match=True,uncontrolled_residual_Nm=float(r.residual25[22]),zero_nominal_passthrough=rz.available))

        # 2: exact force kink, known clipped values, derivative stays unknown.
        begin=len(checks);t=tracker();kw=inputs(t);kw['current_reference8'][:5]=.5;kw['actual_target21']=t._port(kw['current_reference8'])
        kw['v25'][:]=0.;kw['q25'][0]=kw['actual_target21'][0]-.125 # raw effort exactly 1N*m for gain8
        t.servo.force_limited[0]=True;t.servo.force_range[0]=[-1.,1.]
        initial=t.servo.evaluate(kw['q25'],kw['v25'],kw['actual_target21'])
        kw['desired_torque_correction25'][0]=.1;r=t.solve(**kw)
        check('case2_exact_kink_unknown_derivative_preserved',initial.derivative_status=='FORCE_KINK_UNKNOWN_DERIVATIVE' and initial.Ju is None and initial.K is None)
        check('case2_known_kink_scores_without_zero_force',r.available and abs(r.residual25[0])>=.0999999 and not r.debug['physical_safety_certified'],r.status)
        check('case2_kink_is_candidate_boundary',any(abs(b-.5)<1e-12 for b in r.debug['scalar_solutions'][0]['breakpoints']))
        plateau_port=kw['actual_target21'].copy();plateau_port[0]+=.01
        plateau=t.servo.evaluate(kw['q25'],kw['v25'],plateau_port)
        check('case2_strict_clip_known_value',plateau.predicted_effort21[0]==1. and plateau.diagnostics['force_clipped21'][0])
        cases.append(dict(id='known_clip_and_kink_no_fabricated_derivative',passed=all(c['passed'] for c in checks[begin:]),status=r.status,
                          kink_derivative=initial.derivative_status,residual_first_Nm=float(r.residual25[0])))

        # 3: unreachable tube fails and retains goal based on original nominal.
        begin=len(checks);t=tracker();kw=inputs(t);kw['step_lower8'][5]=-.001;kw['step_upper8'][5]=.001
        kw['nominal_motion_reference3'][0]=.15;kw['tracking_tolerances3'][0]=.001
        nominal=np.r_[kw['current_reference8'][:5],kw['nominal_motion_reference3']]
        goal=t.servo.evaluate(kw['q25'],kw['v25'],t._port(nominal)).generalized_actuator_effort25+kw['desired_torque_correction25']
        before=kw['nominal_motion_reference3'].copy();r=t.solve(**kw)
        check('case3_empty_tube_true_failure',not r.available and r.status=='INFEASIBLE_REFERENCE_SET',r.status)
        check('case3_original_goal_not_projected',r.desired_torque25.tobytes()==goal.tobytes())
        check('case3_user_request_unchanged',np.array_equal(before,kw['nominal_motion_reference3']))
        cases.append(dict(id='empty_corridor_original_goal_retained',passed=all(c['passed'] for c in checks[begin:]),status=r.status,
                          requested_lift_m=float(before[0]),actual_step_lift_interval_m=[.049,.051]))

        # 4: both initially unsupported and later unknown native calls fail.
        begin=len(checks);t=tracker();kw=inputs(t);t.servo.unsupported=True;r=t.solve(**kw)
        check('case4_unsupported_model_true_unavailable',not r.available and r.status=='UNAVAILABLE_MODEL' and r.residual25 is None,r.status)
        t=tracker();kw=inputs(t);kw['desired_torque_correction25'][0]=.01
        t.servo.accepted_ports={kw['actual_target21'].tobytes()};r=t.solve(**kw)
        check('case4_unknown_candidate_native_value_not_zero',not r.available and r.status=='NUMERICAL_FAILURE' and r.desired_torque25 is not None,r.status)
        cases.append(dict(id='unsupported_and_unknown_native_never_zero',passed=all(c['passed'] for c in checks[begin:]),
                          unknown_candidate_status=r.status))
    finally:
        for name,fn in original.items():setattr(mujoco,name,fn)
    stable=payload==path.read_bytes();check('source_stable',stable);check('no_forbidden_physics_FK_contact_calls',not any(counts.values()),counts)
    assert not OUT.exists()
    record=dict(schema='v12_native_critical_oracles_independent_v1',role='independent_key_oracle_reviewer',source_sha256=EXPECTED,
        actual_module_solve_executed=True,strict_fake_native_oracle=True,real_model_constructors=0,real_data_constructors=0,physics_steps=0,
        forbidden_call_counts=counts,cases=cases,checks=checks,passed=sum(c['passed'] for c in checks),count=len(checks),
        PASS=all(c['passed'] for c in checks),source_stable=stable,reviewer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_review=dict(native_candidate_scoring_lines=[149,182],candidate_enumeration_lines=[184,233],canonical_final_recheck_lines=[243,274],
            goal_fixed_before_bounds=True,analytic_formula_is_diagnostic_not_score_authority=True,canonical_residual_identical_for_candidates_and_final=True,
            caches_local_to_one_q_v_target=True,unknown_native_candidate_returns_failure=True,no_new_physics_entrypoints=True),
        limitations=['Four synthetic algebra/interface cases only; author actual-model tests and parent675 public-packet replay are separate evidence.',
            'Known clipped force values do not imply known kink derivatives; effort matching is not pressure prediction or physical safety.',
            'No old V12 source/result changed, no physical replay or performance claim.'])
    OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),PASS=record['PASS'],cases=len(cases),checks=record['count'],passed=record['passed']),ensure_ascii=False))
    if not record['PASS']:raise SystemExit(1)

if __name__=='__main__':main()

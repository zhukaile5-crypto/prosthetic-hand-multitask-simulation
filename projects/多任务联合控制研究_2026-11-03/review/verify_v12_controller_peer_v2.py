"""Independent AST-isolated controller contract oracles; no plant/model/physics.

The controller methods and force intents are actual captured source. Robot FK,
servo evaluation and tracker outputs are explicit interface doubles, so this is
neither a real-model constructor test nor an actuator/pressure validation.
"""
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from copy import deepcopy
import ast, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v12_controller_peer_v2.json'

def main():
    files=['v12/controller.py','v12/observation.py','v12/intents.py','v12/effort_tracker.py','v11/object_belief.py']
    captured={f:(ROOT/f).read_bytes() for f in files}
    package=ModuleType('independent_v12');package.__path__=[];sys.modules[package.__name__]=package
    modules={}
    for short in ('observation','intents'):
        name='independent_v12.'+short;mod=ModuleType(name);mod.__package__='independent_v12';sys.modules[name]=mod
        exec(compile(captured['v12/'+short+'.py'],str(ROOT/'v12'/f'{short}.py'),'exec'),mod.__dict__);modules[short]=mod
    obs,intents=modules['observation'],modules['intents']
    tree=ast.parse(captured['v12/controller.py'])
    keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('plain','FusionIntent','TaskController')]
    class RobotSample:
        @staticmethod
        def from_public_frame(*args,**kwargs):return NS(q25=np.zeros(25),velocity25=np.zeros(25))
    env=dict(np=np,deepcopy=deepcopy,RatioIntent=intents.RatioIntent,FixedIntent=intents.FixedIntent,
             project_force=intents.project_force,RELEASE_PHASES=intents.RELEASE_PHASES,ACQUISITION=intents.ACQUISITION,
             array=obs.array,get=obs.get,RobotJointSample=RobotSample,
             ControlAction=lambda delta,predicted,label,pause,debug:NS(delta_closure=delta,predicted=predicted,label=label,pause=pause,debug=debug))
    qp_calls=[]
    def solver(Q,q,A,b,**kwargs):
        x=kwargs['feasible_initial'].copy();qp_calls.append(dict(A=A.copy(),b=b.copy(),x=x.copy()))
        return NS(x=x,converged=True,status='ORACLE_FEASIBLE_START_ONLY')
    env['solve_qp']=solver
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(ROOT/'v12/controller.py'),'exec'),env)
    TC,FI=env['TaskController'],env['FusionIntent']
    g=NS(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,slip_boost_rate=3.,release_rate=.8,
         force_margin=.12,decision_period=.02,open_rate=1.,close_rate=1.,support_contact_floor=.3,
         acquisition_force=(2.,)*5,characteristic_length=.05,friction_margin=.82,load_uncertainty_weight=1.,
         support_buffer=.2,slip_load_gain=110.,force_cost=.06,reference_continuity=3.,balance_weight=.6,
         qp_passes=120,qp_tolerance=2e-5)
    results=[]
    def check(name,ok,detail):results.append(dict(name=name,passed=bool(ok),detail=detail))

    class Tracker:
        def __init__(self,unrealized=False):
            self.unrealized=unrealized;self.calls=[];self.orthogonal=False;self.clips=unrealized
            self.C=np.zeros((21,8));self.C[:5,:5]=np.eye(5);self.C[18:21,5:]=np.eye(3)
            self.rows=tuple(np.flatnonzero(self.C[:,i]) for i in range(8))
            self.servo=NS(evaluate=self.evaluate,ctrladr=np.arange(21),model=NS(actuator_gainprm=np.column_stack((np.ones(21),np.zeros((21,2))))))
        @staticmethod
        def _port(ref):return np.r_[ref[:5],np.zeros(13),ref[5:]]
        @staticmethod
        def evaluate(q,v,port):return NS(generalized_actuator_effort25=np.r_[port[:5],np.zeros(20)],transmission21x25=np.column_stack((np.eye(21),np.zeros((21,4)))))
        def solve(self,**kwargs):
            self.calls.append(kwargs);correction=kwargs['desired_torque_correction25']
            delta=np.zeros(8) if self.unrealized else np.r_[np.clip(correction[:5],kwargs['step_lower8'][:5],kwargs['step_upper8'][:5]),np.zeros(3)]
            residual=-correction if self.unrealized else np.zeros(25)
            if self.orthogonal:residual=np.eye(25)[23]
            debug=dict(native_force_clipped21=[self.clips]*21,residual25=residual.tolist())
            return NS(available=True,increment8=delta,residual25=residual,debug=debug,
                to_dict=lambda:dict(status='KNOWN_CURRENT_STATE_EFFORT_LS_WITH_RESIDUAL' if self.unrealized else 'KNOWN_CURRENT_STATE_EFFORT_MATCH',
                                    native_force_clipped21=[self.unrealized]*21,residual25=(-correction).tolist() if self.unrealized else [0.]*25))
    def fixture(pressure,present,intent=None,unrealized=False):
        c=TC.__new__(TC);c.g=g;c.method='ratio_effort' if intent is not None else 'fixed_effort'
        c.state=NS(valid=np.ones(5,bool),observed_normal=np.full(5,pressure),contact=np.full(5,present,bool),time=1.,_last_time=1.,
                   normal_time=np.ones(5),normal=np.full(5,pressure),delayed_normal=np.full(5,pressure),nominal_weights=np.ones(5),
                   shear_magnitude=np.full(5,.4),slip_speed=0.)
        c.intent=intent if intent is not None else intents.FixedIntent(g);c.limits=np.full(5,5.);c.budget=25.
        c.geometry=NS(raw_joint_names=tuple('j'+str(i) for i in range(24)));c._patches=lambda frame,sample:[]
        c.tracker=Tracker(unrealized);c.infeasible_ticks=0;c.numerical_failures=0;c.last_decision=-np.inf;c.release_latched=False
        c.predicted=np.full(5,np.nan);c.desired=np.zeros(5);c.plan_start=None;c.plan_reference=None;c.plan_delta=np.zeros(8);c.plan_ratio_memory=None
        c.task_request=dict(time_s=1.,mode='hold',limits=[5.]*5,budget=25.,motion_target=[0.,0.,0.],executed_motion_reference=[0.,0.,0.],
            motion_reference_bounds=dict(lower=[0.,-.4,-.4],upper=[.1,.4,.4]),tracking_tolerances=[None]*3)
        c.debug={};c.last_debug={};c.motion_reference=np.zeros(3)
        frame=NS(time=1.,dt=.005,phase='HOLDING',robot_control_time_s=1.,lift_velocity_m_s=0.,alpha=np.full(5,.5),
            closure_min=np.zeros(5),closure_max=np.ones(5),closure_permission=np.ones(5,bool),motion_reference_rate_limits=[.035,.12,.12],
            robot_actuator_target21=Tracker._port(np.r_[np.full(5,.5),np.zeros(3)]),contact_present=np.full(5,present,bool))
        return c,frame
    c,f=fixture(6.,True);c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('valid_high_pressure_missing_geometry_no_closing_search',np.array_equal(c.debug['common_contact_search8'],np.zeros(8)),
          dict(observed_N=6.,desired_N=c.desired.tolist(),search8=c.debug['common_contact_search8'],correction25=c.debug['virtual_force_error_torque25']))
    c,f=fixture(.05,False);c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('explicit_absent_low_pressure_allows_bounded_search',np.allclose(c.debug['common_contact_search8'][:5],.02),c.debug['common_contact_search8'])
    c,f=fixture(6.,False);c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('false_presence_cannot_override_high_pressure',np.array_equal(c.debug['common_contact_search8'],np.zeros(8)),c.debug['common_contact_search8'])
    c,f=fixture(1.,True);c.plan_start=.995;c.plan_reference=np.r_[f.alpha,np.zeros(3)];c.plan_delta=np.ones(8)*.01;c.last_decision=.995
    before=len(c.tracker.calls);a=c.step(f,'RELEASING')
    check('release_mid20ms_plan_preempts_without_tracker',len(c.tracker.calls)==before and np.allclose(a.delta_closure,-.004) and np.array_equal(c.desired,np.zeros(5)) and np.isnan(a.predicted).all(),
          dict(delta=a.delta_closure.tolist(),force_intent_updated=a.debug['force_intent_updated'],tracker_calls=len(c.tracker.calls)))
    c,f=fixture(1.,True);c.last_decision=1.;c.plan_start=1.;c.plan_reference=np.r_[f.alpha,np.zeros(3)];c.plan_delta=np.r_[np.full(5,.02),np.zeros(3)]
    increments=[]
    for k in range(4):
        f.time=1.+.005*k;c.state.time=f.time;c.state._last_time=f.time
        a=c.step(f,'HOLDING');increments.append(a.delta_closure.copy());f.alpha+=a.delta_closure
    check('20ms_plan_four5ms_quarters_from_actual_reference',np.allclose(increments,.005) and np.allclose(f.alpha,.52),
          dict(increments=np.array(increments).tolist(),final_alpha=f.alpha.tolist()))
    fi=FI.__new__(FI);fi.g=g;fi.fallback=intents.RatioIntent(g);fi.previous=None
    fi.snapshot=None;fi.belief=NS(update=lambda frame:NS(grouping_unambiguous=True,selected_index=0,hypotheses=[NS(inlier_indices=(0,1,2))]))
    st=NS(valid=np.ones(5,bool),load_valid=True,mu=np.zeros(5),slip_speed=0.,load=20.,load_uncertainty=0.)
    ps=[dict(index=i,digit=i,normal=1.,patch=NS(reference_normal=np.array(n),reference_position=np.array([float(i),0.,0.])))
        for i,n in enumerate(([0.,0.,-1.],[0.,0.,1.],[0.,0.,1.]))]
    force,_=fi.step(st,'HOLDING',np.full(5,5.),25.,NS(),ps)
    check('signed_downward_support_and_capacity_maximum',np.allclose(force['debug']['capacity_per_normal5'],[-1.,1.,1.,0.,0.]) and np.isclose(force['desired_normal5'][0],.3),
          dict(capacity_per_normal5=force['debug']['capacity_per_normal5'].tolist(),desired_N=force['desired_normal5'].tolist(),solver='feasible-start double; optimization convergence not tested'))

    ratio=intents.RatioIntent(g);ratio.target_ratio=.1;ratio.scalar=2.;ratio.integral=.1;ratio.required_group=np.ones(5,bool)
    c,f=fixture(1.,True,ratio,True)
    c._patches=lambda frame,sample:[dict(index=i,digit=i,normal=1.,row=np.eye(25)[i]) for i in range(5)]
    c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('available_force_plateau_rolls_back_PI',
          ratio.integral==.1 and c.debug['ratio_effort_saturation_memory_rollback'] is True and np.array_equal(c.plan_delta,np.zeros(8)),
          dict(before_integral=.1,after_integral=ratio.integral,desired_N=c.desired.tolist(),actual_increment8=c.plan_delta.tolist(),
               rollback=c.debug['ratio_effort_saturation_memory_rollback']))
    ratio=intents.RatioIntent(g);ratio.target_ratio=.1;ratio.scalar=2.;ratio.integral=.1;ratio.required_group=np.ones(5,bool)
    c,f=fixture(1.,True,ratio,True);c.tracker.clips=False;c.tracker.orthogonal=True
    c._patches=lambda frame,sample:[dict(index=i,digit=i,normal=1.,row=np.eye(25)[i]) for i in range(5)]
    c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('orthogonal_uncontrolled_residual_does_not_freeze_PI',ratio.integral>.1 and not c.debug['ratio_effort_saturation_memory_rollback'],
          dict(integral=ratio.integral,shortfall5=c.debug['finger_reference_effort_shortfall5'],rollback=c.debug['ratio_effort_saturation_memory_rollback']))
    ratio=intents.RatioIntent(g);ratio.target_ratio=.1;ratio.scalar=2.;ratio.integral=.1;ratio.required_group=np.ones(5,bool)
    c,f=fixture(1.,True,ratio,True);c.tracker.clips=False
    c._patches=lambda frame,sample:[dict(index=i,digit=i,normal=1.,row=np.eye(25)[i]) for i in range(5)]
    c._decide(f,'HOLDING',np.r_[f.alpha,np.zeros(3)])
    check('directional_effort_shortfall_without_clip_report_rolls_back_PI',ratio.integral==.1 and c.debug['ratio_effort_saturation_memory_rollback'],
          dict(integral=ratio.integral,shortfall5=c.debug['finger_reference_effort_shortfall5'],rollback=c.debug['ratio_effort_saturation_memory_rollback']))
    ratio=intents.RatioIntent(g);ratio.integral=.106;ratio.boost=.2
    c,f=fixture(1.,True,ratio);c.plan_ratio_memory=dict(integral=.1,boost=.1)
    c.last_decision=1.;c.plan_start=1.;c.plan_reference=np.r_[f.alpha,np.zeros(3)];c.plan_delta=np.r_[np.full(5,.02),np.zeros(3)]
    f.closure_permission[:]=False
    a=c.step(f,'HOLDING')
    check('late5ms_closing_permission_override_rolls_back_PI_boost',np.array_equal(a.delta_closure,np.zeros(5)) and ratio.integral==.1 and ratio.boost==.1 and a.debug['late_execution_ratio_memory_rollback'],
          dict(delta=a.delta_closure.tolist(),integral=ratio.integral,boost=ratio.boost,late_rollback=a.debug['late_execution_ratio_memory_rollback']))
    stable=all((ROOT/f).read_bytes()==b for f,b in captured.items())
    output=dict(schema='v12_controller_independent_peer_v2',role='independent_contract_reviewer',physics_steps=0,plant_constructors=0,
        real_robot_model_constructed=False,method_code='AST-captured actual controller methods; actual observation/intents; explicit FK/servo/tracker doubles',
        source_sha256={f:hashlib.sha256(b).hexdigest() for f,b in captured.items()},sources_stable_during_check=stable,
        fixture_results=results,all_fixture_assertions_passed=all(r['passed'] for r in results),
        findings=[dict(id='AUDIT_EXPORT_NOT_RESTART',status='scope_limit_acknowledged_by_author',file=str(ROOT/'v12/controller.py'),
            explanation='Export is audit-only: task_request and belief last_sample/_last_packet_snapshot/unique_packet_updates are not a restore bundle; no restore API.')],
        resolved_during_review=[dict(source_sha256='64c739ded2d12d80e92098350bc27e14e3c7eb5d2b56d13540c1557fe12db2df',
            issue='missing geometry with valid 6N pressure requested positive search; replaced by explicit absent+low pressure guard in captured current source'),dict(source_sha256='e4ca3142fb630547ca91affed47b9d728cbf3da6ba824c1e0804e72906a3d837',issue='available force plateau originally advanced PI; current source adds native clipping, directional residual, and late execution rollback')],
        limitations=['No dynamics, actuator response, pressure effectiveness, retention or release mechanics tested.',
                     'Geometry grouping is conditional declared-template evidence; selected group is not target identity.',
                     'Current-state effort algebra and virtual contact effort are not a contact-force response model or N safety envelope.'])
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),fixtures=len(results),fixture_assertions_passed=output['all_fixture_assertions_passed'],sources_stable=stable),ensure_ascii=False))
    if not stable or not output['all_fixture_assertions_passed']:raise SystemExit(1)

if __name__=='__main__':main()

"""No-plant checks of V12 observation and force-target separation."""
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import ast,json,sys,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v12.observation import PublicTactileState
from v12.intents import RatioIntent,FixedIntent,project_force
G=dict(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,slip_boost_rate=3.,baseline_tracking_gain=.25,release_rate=.8)


def frame(time=0.,sample=None,normal=None):
    sample=time if sample is None else sample
    m=np.zeros((6,5));m[2]=1.
    return SimpleNamespace(time=time,sample_time=sample,age=time-sample,
        normal=np.full(5,2.) if normal is None else np.asarray(normal,float),
        tangential=np.zeros((5,2)),slip_local=np.zeros((5,2)),valid=np.ones(5,bool),
        shear_magnitude=np.full(5,.1),slip_magnitude=np.zeros(5),
        alpha=np.full(5,.8),joint_velocity=np.zeros(5),phase='HOLDING',
        nominal_force_weights=np.array([3.,1.8,1.8,1.8,1.8]),support_axis=np.array([0.,0.,1.]),
        contact_present=np.ones(5,bool),normal_wrench_map=m,
        finger_non_normal_by_finger=np.zeros((5,6)),distributed_valid=np.ones(5,bool),
        normal_support_coeff=np.ones(5),tangent_support_coeff=np.zeros(5),
        auxiliary_wrench=np.zeros(6),auxiliary_valid=True,auxiliary_pressure=0.,
        auxiliary_age=time-sample,auxiliary_uncertainty=.01,
        load_estimate=10.,load_uncertainty=.8,load_valid=True,
        sensor_characteristics=dict(force_sigma_N=.01))


def run():
    checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    a=PublicTactileState(G);f=frame(.02,.01);a.update(f,.005)
    check('acquisition_pressure_not_present_prediction',np.array_equal(a.normal,np.full(5,2.)) and np.allclose(a.normal_time,.01) and not a.pressure_prediction_available and np.isnan(a.predicted_normal).all())
    n=a.normal.copy();a.update(f,.005);check('duplicate_measurement_no_repeated_filter',np.array_equal(a.normal,n))
    f2=frame(.025,.01);a.update(f2,.005);check('delayed_repeat_age_increases_without_forecast',np.array_equal(a.normal,n) and np.allclose(a.observation_age,.015))
    f3=frame(.03,.02,normal=np.full(5,4.));a.update(f3,.005)
    check('filter_uses_acquisition_delta',np.allclose(a.normal,2+(1-np.exp(-.01/.03))*2,rtol=0,atol=1e-14))
    f4=frame(.04,.03);f4.valid[1]=False;a.update(f4,.005)
    check('dropout_is_unknown_not_zero',not a.valid[1] and np.isnan(a.normal[1]) and not a.contact[1] and a.confidence[1]==0)
    late=frame(.3,.04);a.update(late,.005);check('stale_pressure_is_unknown',not a.valid.any() and np.isnan(a.normal).all())
    class PoisonFrame(SimpleNamespace):
        @property
        def force_matrix(self):raise AssertionError('Forbidden force matrix read')
        @property
        def force_tau(self):raise AssertionError('Forbidden response time read')
    p=PoisonFrame(**vars(frame()));b=PublicTactileState(G);b.update(p,.005)
    p.alpha[:]=1.1;p.time=.005;p.sample_time=.005;b.update(p,.005)
    check('poisoned_B_tau_and_changed_alpha_never_predict_pressure',np.array_equal(b.normal,np.full(5,2.)))
    for name in ('object_state','truth_position','target_id','true_mass','future_disturbances'):
        x=frame();setattr(x,name,object())
        try:PublicTactileState(G).update(x,.005);ok=False
        except ValueError:ok=True
        check('reject_'+name,ok)
    bad=frame(.02,.03)
    try:PublicTactileState(G).update(bad,.005);ok=False
    except ValueError:ok=True
    check('reject_future_acquisition',ok)
    bad=frame(.02,.01);bad.age=0.
    try:PublicTactileState(G).update(bad,.005);ok=False
    except ValueError:ok=True
    check('reject_clock_age_conflict',ok)
    for value in ('0.01',True):
        bad=frame();bad.age=value
        try:PublicTactileState(G).update(bad,.005);ok=False
        except ValueError:ok=True
        check('reject_nonreal_age_'+str(type(value)),ok)
    unknown=frame();unknown.valid[:]=False;unknown.auxiliary_valid=False;unknown.load_valid=False
    u=PublicTactileState(G).update(unknown,.005)
    check('no_unknown_channel_becomes_zero_load',u.load is None and not u.load_valid and u.auxiliary_support is None)
    c=PublicTactileState(G).update(frame(),.005)
    check('distributed6D_units_and_allhand_fusion',np.array_equal(c.normal_wrench_map[2],np.ones(5)) and c.load_valid and abs(c.load-10)<1e-12)
    mu=PublicTactileState(G)
    def sliding(t,valid=True):
        packet=frame(t);packet.shear_magnitude[:]=.4;packet.slip_magnitude[:]=.01;packet.valid[:]=valid
        return packet
    for t in np.arange(9)*.005:mu.update(sliding(float(t)),.005)
    mu.update(sliding(.045,False),.005);mu.update(sliding(.25),.005)
    check('mu_missing_gap_is_not_sustained_slip',not mu.mu_valid.any() and np.array_equal(mu.sliding_since,np.zeros(5)))
    for t in np.arange(.255,.336,.005):mu.update(sliding(float(t)),.005)
    check('mu_requires_new_contiguous_evidence',mu.mu_valid.all())
    # Load only the frozen pure target method / projector AST, never the
    # model-dependent class constructor or a plant/observer from the old code.
    foundation=ROOT.parent/'五指策略接入_2026-11-03'
    source=foundation/'policy_sources.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MigratedRatio')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_ratio_target')
    env={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
    g=SimpleNamespace(**G,decision_period=.02,acquisition_force=(2.,)*5)
    state=SimpleNamespace(time=0.,contact=np.ones(5,bool),valid=np.ones(5,bool),
        delayed_normal=np.full(5,2.),shear_magnitude=np.full(5,.1),slip_speed=0.,nominal_weights=np.array([3.,1.8,1.8,1.8,1.8])/2.04)
    oracle=SimpleNamespace(state=state,g=g,samples=deque(maxlen=5),target_ratio=None,boost=0.,stable_since=None,scalar=2.,integral=0.,debug={})
    intent=RatioIntent(G);all_exact=True
    for i in range(240):
        state.time=i*.02;state.shear_magnitude[:]=.1 if i<100 else (.4 if i<160 else .01)
        state.slip_speed=.006 if 15<=i<20 else 0.
        phase='CLOSING' if i<5 else 'HOLDING'
        expected=project_force(env['_ratio_target'](oracle,phase),np.full(5,4.88),17.76)
        result=intent.step(state,phase,np.full(5,5.),18.)
        all_exact&=np.array_equal(result['desired_normal5'],expected) and intent.integral==oracle.integral and intent.target_ratio==oracle.target_ratio
    check('240_same_input_target_mechanism_steps_exact',all_exact)
    snap=intent.checkpoint();state.time+=.02;state.valid[2]=False;state.delayed_normal[2]=np.nan;state.contact[2]=False
    last=result['desired_normal5'];result=intent.step(state,'HOLDING',np.full(5,5.),18.)
    check('missing_required_group_freezes_PI_and_target',intent.integral==snap['integral'] and np.array_equal(result['desired_normal5'],last) and not result['debug']['memory_updated'])
    intent.restore(snap);check('explicit_actor_memory_rollback',intent.integral==snap['integral'] and intent.boost==snap['boost'])
    r=intent.step(state,'RELEASE',np.full(5,5.),18.)
    check('release_outputs_zero_force_intent_only',np.array_equal(r['desired_normal5'],np.zeros(5)) and set(r)=={'desired_normal5','debug'})
    before=intent.checkpoint();r=intent.step(state,'RELEASING',np.full(5,5.),18.)
    check('real_plant_RELEASING_preempts_ratio_memory',np.array_equal(r['desired_normal5'],np.zeros(5)) and intent.integral==before['integral'] and intent.boost==before['boost'])
    check('real_plant_RELEASING_preempts_fixed',np.array_equal(FixedIntent(G).step(state,'RELEASING',np.full(5,5.),18.)['desired_normal5'],np.zeros(5)))
    before=intent.checkpoint();state.time+=.02
    try:intent.step(state,'HOLDING',np.full(5,-1.),18.);ok=False
    except ValueError:ok=True
    check('bad_limits_never_partially_commit_PI',ok and intent.integral==before['integral'] and intent._last_time==before['last_time'])
    r=FixedIntent(G).step(state,'HOLDING',np.full(5,1.),3.)
    check('fixed_and_ratio_share_force_cap_budget_projector',r['desired_normal5'].sum()<=2.76+1e-7 and (r['desired_normal5']<=.88+1e-12).all())
    return dict(schema='v12_public_observation_force_intents_pure_v1',passed=all(c['passed'] for c in checks),checks=checks,
        physics_steps=0,old_model_or_controller_constructed=False,oracle_scope='same supplied public filtered state only; new observer is not old CoupledState/full-paper equivalence',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'v12/observation.py',ROOT/'v12/intents.py',Path(__file__))})


if __name__=='__main__':
    x=run();p=ROOT/'review/v12_observation_intents_pure_v2.json'
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=x['passed'],checks=len(x['checks']),failed=[c['name'] for c in x['checks'] if not c['passed']],output=str(p))))
    raise SystemExit(0 if x['passed'] else 2)

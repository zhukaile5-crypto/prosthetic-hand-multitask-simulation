"""Static/interface review, isolated step method; never advances a simulator."""
from pathlib import Path
from types import SimpleNamespace as S
from datetime import datetime,timezone
import ast,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={f:sha(ROOT/f) for f in ('coordinator.py','coordinator_model.py','runtime.py','evaluation.py','protocol.json')}
module=ast.parse((ROOT/'coordinator.py').read_text(encoding='utf-8'))
cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='JointCoordinator')
step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
namespace=dict(np=np,ControlAction=lambda d,p,label,pause,debug:S(delta_closure=d,predicted_normal=p,label=label,pause_transport=pause,debug=debug))
exec(compile(ast.Module(body=[step],type_ignores=[]),'isolated_joint_step','exec'),namespace)
alpha=np.zeros(5);motion=np.array([.09,0.,0.]);calls=[];updates=[]
state=S(valid=np.ones(5,bool),closure_permission=np.ones(5,bool),alpha=alpha.copy(),closure_min=np.full(5,-.02),closure_max=np.full(5,.02))
def update(frame,dt):state.alpha=frame.alpha.copy();updates.append(dt)
state.update=update
fake=S(previous_time=None,state=state,model=S(update=lambda *args:None),executed_motion=motion.copy(),
    control_ticks=0,last_decision=-np.inf,g=S(decision_period=.02,release_rate=.8),
    interval_action=np.zeros(5),interval_motion=np.zeros(3),motion_reference=motion.copy(),motion_request=np.array([.10,.02,.01]),
    predicted=np.ones(5),label='synthetic',debug={})
planned=np.array([.008,-.004,.002,-.006,.004]);planned_motion=np.array([.0004,.001,.0008])
def solve():
    calls.append(fake.previous_time);fake.interval_action=planned.copy();fake.interval_motion=planned_motion.copy()
fake._solve=solve
for index in range(4):
    fake.executed_motion=motion.copy()
    frame=S(time=7.5+index*.005,phase='HOLDING',alpha=alpha.copy())
    action=namespace['step'](fake,frame,'HOLDING','hold')
    alpha+=action.delta_closure;motion=fake.motion_reference.copy()
checks=dict(four_ticks_equal_planned_action=bool(np.allclose(alpha,planned,atol=1e-12,rtol=0)),
    four_ticks_equal_planned_motion=bool(np.allclose(motion-np.array([.09,0.,0.]),planned_motion,atol=1e-12,rtol=0)),
    exactly_one_decision_in_four_ticks=len(calls)==1,
    observer_updates_every_5ms=bool(np.allclose(updates,.005,atol=1e-12,rtol=0)),
    no_policy_mass_friction_target_identity_access=True)
for filename in ('coordinator.py','coordinator_model.py'):
    tree=ast.parse((ROOT/filename).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value in ('mass','friction','is_target','truth','object_pose','target_id','config'):
            checks['no_policy_mass_friction_target_identity_access']=False
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and node.attr in ('mass','friction','is_target','truth','object_pose','target_id'):
            checks['no_policy_mass_friction_target_identity_access']=False
calibration=json.loads((ROOT.parent/'五指策略接入_2026-11-03/review/foundation_calibration_final.json').read_text(encoding='utf-8'))
clipped={}
for shape,row in calibration['shapes'].items():
    errors=np.asarray(row['response_error_bound'])
    if np.any(errors>.4):clipped[shape]=dict(admitted_errors=errors.tolist(),v1_errors=np.minimum(.4,errors).tolist(),lost_bound=(errors-np.minimum(.4,errors)).tolist())
counter=json.loads((ROOT/'review/independent_joint_budget_counterexample.json').read_text(encoding='utf-8'))
findings=[dict(priority='correctness_revision_required',file='coordinator_model.py',line=98,
    issue='Admitted pressure error bounds are capped downward at .40N before entering pressure constraints.',evidence=clipped),
    dict(priority='correctness_revision_required',file='coordinator_model.py',line=82,
    issue='Drift error-rate estimates are capped downward at5N/s without a physical upper-bound certificate.'),
    dict(priority='correctness_revision_required',file='coordinator.py',line=93,
    issue='No lower bound enforces nonnegative affine contact pressure. Negative forecasts credit total budget and wrench; final display clips them.',
    negative_pressure_credit_N=counter['clipped_sum']-counter['raw_sum'],
    counterexample='review/independent_joint_budget_counterexample.json'),
    dict(priority='verification_fixture_revision_required',file='coordinator.py',line=215,
    issue='self_test SimpleNamespace lacks normal read at_solve debug line167; actual CoupledTactileState has it.'),
    dict(priority='claim_limitation',file='coordinator.py',line=116,
    issue='Stage1 is a normalized quadratic pressure-slack minimization with finite solver passes. A feasible fallback is not proof of a global or strict lexicographic safety optimum; stage2 preserves the obtained componentwise slack bound within .0005N.'),
    dict(priority='claim_limitation',file='coordinator_model.py',line=39,
    issue='Motion gain is zero/unidentified; only an engineering absolute-motion envelope links progress to pressure constraints. This is not an identified coupled object-dynamics predictor.'),
    dict(priority='model_limitation',file='coordinator.py',line=135,
    issue='Wrench anchor remains in palm coordinates across wrist rotation; anchor_up is stored but not used to transform the anchor. Gravity/task wrench consistency needs separate validation.')]
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),interface_checks=checks,
    interface_checks_passed=all(checks.values()),source_sha256=sources,
    policy_model_correctness_fully_accepted=False,formal_admission=False,development_observation_not_interrupted=True,
    findings=findings,scope='Pure static/isolated four-tick interface/QP evidence. No dynamics, hand model, performance comparison or frozen source modifications. First18 run sources deliberately remain unchanged.')
out=ROOT/'review/independent_joint_interface_review_v1.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(interface_passed=result['interface_checks_passed'],checks=checks,findings=len(findings),path=str(out),sha256=sha(out)),ensure_ascii=False))

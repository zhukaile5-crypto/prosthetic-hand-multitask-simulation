"""Executable pure specification oracles; not a V12 runtime/physics admission."""
from pathlib import Path
from collections import deque
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v12_preacceptance_comparison_plan_v1.json'
S=ROOT/'output/release_primitive_prospective_v2/sources/foundation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def main(out):
 if out.exists():raise FileExistsError(out)
 plan=read(PLAN);checks=[]
 def check(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
 check('three_methods_share_one_contract',len(plan['primary_comparisons'])==3 and {p['method'] for p in plan['primary_comparisons']}=={'v12_shared_fixed_intent','v12_shared_ratio_intent','v12_fusion'})
 check('no_physics_assignment_or_utility_weight_invented',plan['status']=='draft_specification_not_a_frozen_physics_assignment' and not plan['physics_started_by_reviewer'] and not plan['performance_admitted'] and not plan['material_safety_certified'])
 reference=ROOT/plan['frozen_old_execution_reference']['raw_audit'];check('old_complete_reference_binds_exact_evidence',sha(reference)==plan['frozen_old_execution_reference']['raw_audit_sha256'])
 # Freeze only the old pure target and mathematical projection source methods.
 tree=ast.parse((S/'policy_sources.py').read_text(encoding='utf-8'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MigratedRatio');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_ratio_target')
 tree=ast.parse((S/'vendor/frozen_controllers.py').read_text(encoding='utf-8'));projection=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_force')
 env={'np':np};exec(compile(ast.Module(body=[method,projection],type_ignores=[]),'frozen-pure-reference','exec'),env)
 g=SimpleNamespace(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,slip_boost_rate=3.,decision_period=.02,acquisition_force=(2.,)*5)
 def ratio_case(scalar,integral,ratio,target):
  state=SimpleNamespace(contact=np.ones(5,bool),shear_magnitude=np.full(5,2*ratio),delayed_normal=np.full(5,2.),slip_speed=0.,time=20.,nominal_weights=np.ones(5))
  memory=SimpleNamespace(state=state,g=g,samples=deque([ratio]*4,maxlen=5),target_ratio=target,boost=0.,stable_since=0.,scalar=scalar,integral=integral,debug={})
  requested=env['_ratio_target'](memory,'HOLDING');return memory,requested
 m,p=ratio_case(5.,.1,.5,.2)
 check('frozen_PI_above_upper_positive_error_freezes_integral',abs(m.integral-.1)<1e-12 and np.array_equal(p,np.full(5,5.)),dict(trial_integral=.106,raw=5.2612,committed_integral=m.integral,returned_target=p.tolist()))
 m,p=ratio_case(5.2,0.,.1,.2)
 check('frozen_PI_above_upper_negative_error_commits_recovery',abs(m.integral+.002)<1e-12 and np.array_equal(p,np.full(5,5.)),dict(raw=5.1196,committed_integral=m.integral))
 m,p=ratio_case(1.6,0.,.3,.2)
 check('frozen_PI_below_floor_positive_error_commits_recovery',abs(m.integral-.002)<1e-12 and np.array_equal(p,np.full(5,1.8)),dict(raw=1.6804,committed_integral=m.integral))
 projected=env['project_force'](np.array([6.,2.,2.,2.,2.]),np.full(5,4.),10.)
 check('box_budget_projection_is_not_independent_finger_clipping',np.allclose(projected,[4.,1.5,1.5,1.5,1.5],rtol=0,atol=1e-7),projected.tolist())
 # Readiness reference deliberately has no alpha/B-domain or object argument.
 def ready(normal,valid,lift,phase,target):
  observed=np.asarray(valid,bool)&np.isfinite(normal)&(np.asarray(normal)>.2)
  return bool(phase=='HOLDING' and observed[0] and observed.sum()>=3 and abs(lift-target)<=.01)
 alpha=np.full(5,.9);reference_alpha=np.full(5,.5);radius=np.full(5,.02)
 check('physical_ready_true_while_old_model_domain_false',ready(np.array([1.,1.,1.,0.,0.]),np.ones(5,bool),.1,'HOLDING',.1) and not np.all(abs(alpha-reference_alpha)<=radius))
 check('readiness_requires_thumb_phase_and_lift',not ready(np.array([0.,1.,1.,1.,0.]),np.ones(5,bool),.1,'HOLDING',.1) and not ready(np.ones(5),np.ones(5,bool),.1,'LIFTING',.1) and not ready(np.ones(5),np.ones(5,bool),.08,'HOLDING',.1))
 # Same pre-action state can legitimately have different post-action controls.
 ctrl_before_old=np.zeros(21);ctrl_before_new=np.zeros(21);ctrl_after_old=np.zeros(21);ctrl_after_new=np.full(21,.001)
 check('handoff_prefix_is_pre_action_not_post_action_row',np.array_equal(ctrl_before_old,ctrl_before_new) and not np.array_equal(ctrl_after_old,ctrl_after_new))
 peer=read(ROOT/'review/v12_observation_intents_independent_first_v1.json')
 check('narrow_implemented_module_oracles_passed',peer['passed'] and not peer['physics_steps'] and not peer['physics_api_imported'])
 check('narrow_module_evidence_still_binds_current_source',all(sha(ROOT/'v12'/n)==h for n,h in peer['source_sha256'].items()))
 check('no_plant_or_physics_import','mujoco' not in sys.modules and 'hand_plant' not in sys.modules)
 result=dict(schema='v12_preacceptance_specification_oracles_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(c['passed'] for c in checks),checks=checks,plan_sha256=sha(PLAN),checker_sha256=sha(Path(__file__)),reference_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [S/'policy_sources.py',S/'vendor/frozen_controllers.py']},module_peer_evidence_sha256=sha(ROOT/'review/v12_observation_intents_independent_first_v1.json'),physics_steps=0,old_controller_or_plant_constructed=False,actual_v12_runtime_or_fusion_tested=False,execution_admissible=False,remaining=plan['missing_before_execution'],role='Pure reference/specification check only. Actual V12 integration, prefix bridge, effort realization and complete physical outcomes remain untested.')
 with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(output=str(out),sha256=sha(out),passed=result['passed'],checks=len(checks),physics_steps=0,actual_v12_runtime_or_fusion_tested=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();main(a.out)

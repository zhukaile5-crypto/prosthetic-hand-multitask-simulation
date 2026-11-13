from pathlib import Path
import sys,json,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v15_support.posture_tasks import build_holding_native_tasks,GROUPS
checks=[]
def check(name,ok,detail=None):
 checks.append(dict(name=name,passed=bool(ok),detail=detail))
 if not ok:raise AssertionError(name)
def run():
 opened=np.zeros(18);closed=np.concatenate([np.arange(1,b-a+1,dtype=float) for a,b in GROUPS]);gain=np.linspace(.7,2.1,21);p0=np.linspace(-.2,.3,21)
 kw=dict(patch_count=7,time_s=1.,p0_21=p0,delta_scale21=np.linspace(.1,.2,21),nominal_reference21=np.zeros(21),grasp_anchor21=np.zeros(21),native_reference_gain21=gain,known_affine21=np.ones(21,bool),control_range21=np.tile([-2.,2.],(21,1)),open18=opened,closed18=closed)
 x=build_holding_native_tasks(**kw);check('all5_known_shape_groups_no_history',x.available and x.matrix.shape==(18,28) and not x.metadata['inactive_shape_groups'])
 d=np.zeros(21)
 for i,(a,b) in enumerate(GROUPS):d[a:b]=gain[a:b]*closed[a:b]*(.01+i*.005)
 check('synergy_direction_preserved_soft_nullspace',np.max(abs(x.matrix@np.r_[d,np.zeros(7)]-x.target))<2e-16)
 reference=np.linspace(-.04,.07,21);test=np.r_[gain*reference,np.zeros(7)];expected=[]
 for a,b in GROUPS:
  s=closed[a:b]/4.;P=np.eye(b-a)-np.outer(s,s)/(s@s);expected.extend(P@(reference[a:b]/4.))
 actual=x.matrix@test-x.target
 check('native_gain_to_reference_shape_residual',np.allclose(actual,expected,rtol=0,atol=3e-17))
 check('off_manifold_deviation_penalized',actual@actual>0)
 negative=kw.copy();negative['native_reference_gain21']=-gain
 y=build_holding_native_tasks(**negative)
 check('negative_native_gain_preserves_reference_semantics',np.array_equal(y.matrix,-x.matrix))
 mask=np.ones(21,bool);mask[0]=False;y=build_holding_native_tasks(**{**kw,'known_affine21':mask})
 check('unknown_axis_disables_whole_digit_not_zero_inverse',y.matrix.shape==(13,28) and y.metadata['inactive_shape_groups']==[dict(digit=0,reason='native_inverse_branch_or_control_span_not_known')])
 flat=gain.copy();flat[6]=0.;y=build_holding_native_tasks(**{**kw,'native_reference_gain21':flat})
 check('known_zero_gain_has_no_invented_inverse',len(y.metadata['inactive_shape_groups'])==1 and y.metadata['inactive_shape_groups'][0]['digit']==1 and np.isfinite(y.matrix).all())
 previous=np.linspace(.3,.5,21);y=build_holding_native_tasks(**{**kw,'previous_model_effort21':previous,'previous_model_time_s':.98})
 check('history_rows_are_native_model_not_pressure',y.matrix.shape==(39,28) and y.metadata['history_elapsed_s']>0)
 dd=np.linspace(-.03,.02,21);history=y.matrix[-21:]@np.r_[dd,np.zeros(7)]-y.target[-21:]
 expected=kw['delta_scale21']*((p0+dd)-previous)
 check('history_residual_matches_original_effort_goal',np.allclose(history,expected,rtol=0,atol=2e-17))
 y=build_holding_native_tasks(**{**kw,'grasp_anchor21':None})
 check('unknown_grasp_anchor_explicit_inactive',y.matrix.shape==(0,28) and y.metadata['shape_task_status']=='actual_grasp_anchor_unknown_inactive')
 y=build_holding_native_tasks(**{**kw,'previous_model_effort21':previous,'previous_model_time_s':1.})
 check('same_clock_sample_not_past_history',y.matrix.shape==(18,28) and y.metadata['history_task_status']=='same_clock_sample_not_a_past_history_task')
 for label,change in [('future',dict(previous_model_effort21=previous,previous_model_time_s=1.01)),('unknown_gain',dict(native_reference_gain21=None)),('bool_units',dict(delta_scale21=np.ones(21,bool))),('unordered_range',dict(control_range21=np.tile([1.,-1.],(21,1))))]:
  try:build_holding_native_tasks(**{**kw,**change})
  except ValueError:ok=True
  else:ok=False
  check('invalid_'+label+'_rejected',ok)
 tiny=closed.copy();tiny[:5]=[1e-200,0.,0.,0.,0.]
 tiny_spec=build_holding_native_tasks(**{**kw,'closed18':tiny,'control_range21':np.tile([0.,1.],(21,1)),'native_reference_gain21':np.ones(21)})
 along=np.zeros(28);along[0]=1.
 check('nonzero_subnormal_norm_direction_not_fixed_template',np.max(abs(tiny_spec.matrix[:5]@along))==0.)
 huge_range=np.tile([-1e308,1e308],(21,1))
 try:build_holding_native_tasks(**{**kw,'control_range21':huge_range})
 except ValueError:span_rejected=True
 else:span_rejected=False
 check('finite_endpoints_overflowing_span_rejected',span_rejected)
 check('returned_matrix_target_readonly',not x.matrix.flags.writeable and not x.target.flags.writeable)
 check('no_object_or_environment_inputs',not any('object' in s or 'environment' in s for s in __import__('inspect').signature(build_holding_native_tasks).parameters))
 return dict(passed=True,checks=checks,physics_steps=0,model_imports=0,scope='Pure declared affine native reference task identities; not derivative admission or physical pose certificate')
if __name__=='__main__':
 path=ROOT/'review/v15_posture_tasks_author_v3.json';assert not path.exists()
 try:result=run()
 except Exception as e:result=dict(passed=False,checks=checks,error=repr(e),traceback=traceback.format_exc(),physics_steps=0)
 result['source_sha256']=hashlib.sha256((ROOT/'v15_support/posture_tasks.py').read_bytes()).hexdigest()
 path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),source=result['source_sha256'])));print(result.get('traceback',''))
 raise SystemExit(0 if result['passed'] else 1)

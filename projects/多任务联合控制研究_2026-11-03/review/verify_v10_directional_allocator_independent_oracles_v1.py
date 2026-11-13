"""Four independent closed-form algebra oracles, no simulator imports."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import v10.directional_allocator as subject
SOURCE=Path(subject.__file__);EXPECTED='acf70f91cd53426c014a048cf29c80edb5013d7cd2f90688bb6c847cdddfbb8e'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(SOURCE)!=EXPECTED:raise RuntimeError('Independent oracle source binding changed')
OUT=ROOT/'review/v10_directional_allocator_independent_oracles_v1.json'
if OUT.exists():raise FileExistsError(OUT)
checks=[];examples=[]
def check(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
def fixture():
 B=np.zeros((25,8));B[24,5]=1.;B[0,6]=1.;B[1,7]=1.
 return dict(nominal_increment8=np.zeros(8),step_lower8=np.full(8,-1.),step_upper8=np.full(8,1.),robot_free25=np.zeros(25),robot_response25x8=B,output_axes25=subject.ROBOT_AXES25,user_target3=np.zeros(3),tracking_tolerances3=[.001,.002,.003],empirical_robot_error25=np.zeros(25),contact_rows_mx8=np.zeros((1,8)),contact_free_delta_m=np.zeros(1),contact_error_m=np.zeros(1),contact_urgency_m=np.ones(1),model_scope_label='INDEPENDENT_CLOSED_FORM_SYNTHETIC_ONLY')
def run(name,kw):
 before=deepcopy(kw);r=subject.solve(**kw)
 check(name+'_no_mutation',all(np.array_equal(v,before[k]) if isinstance(v,np.ndarray) else v==before[k] for k,v in kw.items()))
 check(name+'_actual_bounds',np.all(r.increment8>=kw['step_lower8']) and np.all(r.increment8<=kw['step_upper8']))
 check(name+'_no_force_certificate',r.debug['pressure_safety_certified'] is False and r.debug['global_model_admitted'] is False and r.debug['force_safety_slack_N'] is None)
 examples.append(dict(name=name,increment8=r.increment8.tolist(),debug=r.debug));return r

#1 Motion itself withdraws. Inconsistent/narrow model-proposal search region
#must not interfere with the exact healthy nominal bypass.
k=fixture();k['nominal_increment8'][[5,6]]=[-.07,.023];k['user_target3']=[-.07,.023,0.];k['contact_rows_mx8'][0,5]=1.;k['contact_error_m'][0]=.001;k['step_lower8'][5]=-.2;k['step_upper8'][5]=.3
k['proposal_lower8']=np.zeros(8);k['proposal_upper8']=np.zeros(8);k['proposal_l1_weights8']=np.ones(8);k['proposal_l1_budget']=0.
r=run('away_motion_exact_outside_proposal',k)
check('away_nominal_bitexact',np.array_equal(r.increment8,k['nominal_increment8']))
check('outside_proposal_not_mislabeled_valid',r.debug['increment_within_declared_proposal_region'] is False and r.debug['prediction_scope_status']=='EXPERIMENTAL_OUTSIDE_DECLARED_PROPOSAL_REGION')

#2 A coupled inward row with unequal true halfspans has a CLOSED FORM
#weighted projection. This also documents the limit of raw tangent invariance.
k=fixture();k['step_lower8'][:]=k['step_upper8'][:]=0.;k['step_lower8'][:2]=[-2.,-1.];k['step_upper8'][:2]=[2.,1.];k['nominal_increment8'][:2]=[.4,.4];k['tracking_tolerances3']=[None,None,None]
k['robot_response25x8'][:]=0.;k['robot_response25x8'][2,0]=1.;k['robot_response25x8'][3,1]=1.;k['contact_rows_mx8'][0,:2]=[2.,1.]
r=run('coupled_halfspan_metric',k);a=np.array([2.,1.]);D2=np.diag([4.,1.]);nom=np.array([.4,.4]);oracle=nom-(D2@a)*(a@nom)/(a@D2@a);tangent=np.array([1.,-2.]);tangent_change=float(tangent@(r.increment8[:2]-nom))
check('weighted_projection_closed_form',np.allclose(r.increment8[:2],oracle,rtol=0,atol=3e-8),dict(oracle=oracle.tolist(),actual=r.increment8[:2].tolist()))
check('general_raw_tangent_preservation_is_not_promised',abs(tangent_change)>.4,dict(tangent_vector=tangent.tolist(),change=tangent_change,meaning='Normalized-command metric can alter raw Euclidean tangent when the contact row is coupled; protect a desired tangent via its explicit task constraints.'))

#3 One-sided unreachable tube must retain the ORIGINAL goal and the
#minimum attained tube slack before softer inward geometry.
k=fixture();k['step_lower8'][:]=k['step_upper8'][:]=0.;k['step_lower8'][5]=0.;k['step_upper8'][5]=.4;k['nominal_increment8'][5]=.13;k['user_target3']=[.9,0.,0.];k['tracking_tolerances3']=[.02,None,None];k['empirical_robot_error25'][24]=.01;k['contact_rows_mx8'][0,5]=1.;k['contact_free_delta_m'][0]=.02;k['contact_error_m'][0]=.03
r=run('unreachable_tube_with_empirical_margin',k)
check('original_goal_and_tube_priority',r.debug['user_target3']==[.9,0.,0.] and abs(r.increment8[5]-.4)<1e-8)
check('tube_and_geometry_slack_closed_form',abs(r.debug['tube_point_violation3'][0]-.48)<1e-8 and abs(r.debug['tube_empirical_violation3'][0]-.49)<1e-8 and abs(r.debug['unresolved_geometry_m'][0]-.45)<1e-8)
stages=r.debug['stages'];first=next(s for s in stages if s['name']=='original_user_tube')['attained_tube_slack3'][0]
check('every_later_stage_protects_attained_tube',all(s['attained_tube_slack3'][0]<=first+1e-8 for s in stages if s.get('independently_checked')))
check('no_physical_infeasibility_claim',r.debug['physical_infeasibility_certified'] is False)

#4 A true zero geometric derivative with positive free drift/error remains
#unresolved; a missing row is unknown, not that measured zero derivative.
k=fixture();k['tracking_tolerances3']=[None,None,None];k['nominal_increment8'][:2]=[.012,-.023];k['step_lower8']=k['nominal_increment8'].copy();k['step_upper8']=k['nominal_increment8'].copy();k['contact_rows_mx8']=None;k['contact_free_delta_m']=k['contact_error_m']=k['contact_urgency_m']=None
unknown=run('unknown_geometry',k)
k['contact_rows_mx8']=np.zeros((1,8));k['contact_free_delta_m']=[.006];k['contact_error_m']=[.001];k['contact_urgency_m']=[3.]
zero=run('measured_true_zero_derivative',k)
check('unknown_truezero_semantically_distinct',unknown.debug['geometry_status']=='UNKNOWN_ROWS_NOT_SUPPLIED' and zero.debug['zero_geometry_rows']==[0] and zero.debug['geometry_status']=='MEASURED_ROWS_WITH_EMPIRICAL_ERROR')
check('truezero_unresolved_free_drift_retained',abs(zero.debug['unresolved_geometry_m'][0]-.007)<1e-12 and np.array_equal(zero.increment8,k['nominal_increment8']))
check('no_mujoco_or_physics_import','mujoco' not in sys.modules)
report=dict(schema='v10_directional_allocator_independent_oracles_v1',allocator_source_sha256=sha(SOURCE),checker_sha256=sha(Path(__file__)),passed=all(c['passed'] for c in checks),checks=checks,examples=examples,physics_steps=0,production_modified=False,source_review=dict(normalisation='center+halfspan removes fixed axes; final least modification uses ACTUAL mechanical halfspans, not an unscaled Euclidean command metric',stages='actual attained normalized tube then geometry slack are re-evaluated and protected componentwise; final outputs rechecked inside original real bounds',early_bypass='healthy legal nominal passes before model-proposal trust region; explicit outside-proposal label preserves this distinction',limits=['Generic raw Euclidean tangent invariance is false for coupled rows/unequal halfspans without an explicit tangent task.','Absent/invalid active risk rows retain legal nominal even if its predicted motion misses a user tube; this allocator is a conditional directional intervention, not a standalone tracking controller.','Synthetic closed-form QA does not admit any measured mixed sequence, finite FK error, force forecast, material safety or plant benefit.']),source_lines=dict(healthy_bypass=[150,157],normalization=[165,183],actual_stage_recheck_and_protection=[236,257],normalized_least_change=[268,272]))
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),passed=report['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],weighted_projection=oracle.tolist(),raw_tangent_change=tangent_change,physics_steps=0)))

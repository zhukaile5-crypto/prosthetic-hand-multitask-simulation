"""V22/V23 completed joint friction: first emitted divergence, finite local same-input CF."""
from pathlib import Path
import sys,json,gzip,re,hashlib,time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
forbidden=[]
def deny(*a,**kw):forbidden.append('engine_model_step_forward_contact_writer');raise AssertionError(forbidden[-1])
class NoModel:
 def __new__(cls,*a,**kw):return deny()
 from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
mujoco.MjModel=NoModel;mujoco.MjData=deny
for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 if hasattr(mujoco,k):setattr(mujoco,k,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=deny
from v22_joint_task_fusion import task_solver as old
from v23_task_execution import task_solver as new
from v23_task_execution.runtime import source_identity
from v21_stable_handoff.prefix_supervisor import score_emitted_reference
from v21_original_matrix_kkt_verifier_v1 import verify_stage
from v14_coord.design import reference_rates
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if hasattr(x,'__dataclass_fields__'):return plain({k:getattr(x,k) for k in x.__dataclass_fields__})
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def phash(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def diff(a,b):
 if a is None or b is None:return {'exact':a is None and b is None,'left_is_none':a is None,'right_is_none':b is None}
 x=np.asarray(a);y=np.asarray(b)
 if x.shape!=y.shape:return {'exact':False,'shape_left':list(x.shape),'shape_right':list(y.shape)}
 if x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf':
  x=x.astype(float);y=y.astype(float);return {'exact':x.tobytes()==y.tobytes(),'shape':list(x.shape),'difference_inf':float(np.max(abs(y-x),initial=0)),'changed_count':int(np.count_nonzero(x.view(np.uint64)!=y.view(np.uint64)))}
 return {'exact':phash(a)==phash(b)}
def cert_summary(cert):
 return dict(name=cert['name'],squared_loss=cert['squared_loss'],attained=cert['attained'],KKT_certified=cert['KKT'].get('certified'),A_sha256=phash(cert['matrix']),b_sha256=phash(cert['target']),G_sha256=phash(cert['hard_domain']['original_inequality_matrix_with_box']),h_sha256=phash(cert['hard_domain']['original_inequality_upper_with_box']),E_sha256=phash(cert['hard_domain']['fixed_equality_matrix']),f_sha256=phash(cert['hard_domain']['fixed_equality_target']))
def verify_certificate(cert):
 h=cert['hard_domain'];x=np.asarray(cert['x']);n=len(x);E=np.asarray(h['fixed_equality_matrix'],float).reshape((-1,n));f=np.asarray(h['fixed_equality_target'],float);frozen=cert['previous_frozen']
 E=np.vstack([E]+[np.asarray(z['matrix'],float) for z in frozen]);f=np.r_[f,*[np.asarray(z['attained'],float) for z in frozen]]
 k=cert['KKT'];p=dict(kind='least_squares',A=cert['matrix'],b=cert['target'],G=h['original_inequality_matrix_with_box'],h=h['original_inequality_upper_with_box'],E=E,f=f,previous_frozen=frozen,atol=1e-9,rtol=1e-8)
 v=verify_stage(p,x,duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='counterfactual_current_candidate_dual_independently_checked_original_rows'))
 return dict(name=cert['name'],independent_KKT=v['kkt_passed'],objective=v.get('objective'),stationarity_inf=v.get('stationarity_inf'),primal_passed=v.get('primal',{}).get('passed'),original_problem_sha256=v.get('original_problem_sha256'))
def kwargs(z):
 d=z['hierarchy']['diagnostics'];c=z['contact_tasks'];a=lambda x:None if x is None else np.asarray(x,float)
 return dict(time_s=z['time_s'],dt_s=.005,current_reference21=a(z['current_reference21']),encoder_reference21=a(c['encoder_reference21']),step_lower21=a(z['actual_reference_lower21'])-a(z['current_reference21']),step_upper21=a(z['actual_reference_upper21'])-a(z['current_reference21']),reference_rates21=reference_rates('release'),normal_jacobian_Px25=a(c['normal_jacobian_Px25']),proposal_map25x21=a(c['proposal_map25x21']),robot_velocity25=a(c['robot_velocity25']),pressure_P_N=a(c['pressure_P_N']),reference_target_delta21=a(z['reference_target_delta21']),motion_matrix_Mx21=a(c.get('motion_matrix_Mx21')),motion_error_M_m_s=a(c.get('motion_error_M_m_s')),motion_time_s=c.get('motion_time_s'),path_reference_target_delta21=a(z.get('path_reference_target_delta21')),max_iterations=400,atol=1e-9,rtol=1e-8,formulation='joint')
runs={v:ROOT/f'output/{v}_small_joint_friction_drop_transport_619742_v1/runs/friction_drop__ratio_effort__619742' for v in ('v22','v23')};source=source_identity(['ratio_effort']);identity=numerical_identity();assert len(source)==186
cache={};records={};input_pins={};times=[13.625,13.63,13.635,13.64];ticks=[round(t*200) for t in times]
for v,p in runs.items():
 with np.load(p/'trace.npz',allow_pickle=False) as z:
  # Each needed compressed column is loaded ONCE, then indexed in memory.
  cache[v]={k:z[k] for k in ('time','executed_reference21','requested_reference21','reference_increment21','observation_json')}
  cache[v]['public_selected']={i:json.loads(str(cache[v]['observation_json'][i])) for i in ticks};del cache[v]['observation_json']
 keep={}
 with gzip.open(p/'release_decisions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   t=float(re.search(r'"time_s"\s*:\s*([0-9.eE+-]+)',line).group(1));i=round(t*200)
   if i in ticks:keep[i]=json.loads(line)
   if i>=ticks[-1]:break
 assert sorted(keep)==ticks;records[v]=keep
 input_pins[v]={name:sha(p/name) for name in ('trace.npz','release_decisions.jsonl.gz','metadata.json','result.json')}
t=cache['v22']['time'];assert np.array_equal(t.view(np.uint64),cache['v23']['time'].view(np.uint64));a=cache['v22']['executed_reference21'];b=cache['v23']['executed_reference21'];bound=64*np.finfo(float).eps*np.maximum(1,np.maximum(abs(a),abs(b)));bitmask=np.any(a.view(np.uint64)!=b.view(np.uint64),axis=1);largemask=np.any(abs(b-a)>bound,axis=1)
first={}
for name,mask in [('bitwise',bitmask),('beyond_original_64eps',largemask)]:
 i=int(np.flatnonzero(mask)[0]);delta=b[i]-a[i];first[name]=dict(index=i,time_s=t[i],axes_bit_different=np.flatnonzero(a[i].view(np.uint64)!=b[i].view(np.uint64)),axes_beyond_64eps=np.flatnonzero(abs(delta)>bound[i]),delta21=delta,max_abs=float(np.max(abs(delta))),original_64eps_bound21=bound[i],prefix_rows_before_all_within_test=bool(not np.any(mask[:i])))
observed=[]
for i in ticks:
 z0=records['v22'][i];z1=records['v23'][i];row=dict(time_s=t[i],reference_difference=diff(a[i],b[i]),public_input_differences={})
 for key in ('robot_position25','robot_velocity25','sample_time','time','normal','valid','robot_reference21','robot_actuator_target21','robot_control_time_s'):
  f0=cache['v22']['public_selected'][i];f1=cache['v23']['public_selected'][i]
  row['public_input_differences'][key]=dict(available_left=key in f0,available_right=key in f1,comparison=diff(f0.get(key),f1.get(key)))
 summaries={}
 for v,z in [('v22',z0),('v23',z1)]:
  h=z.get('hierarchy');q=dict(phase=z['phase'],geometry_proposal_accepted=z['geometry_proposal_accepted'],fallback=z['fallback'],numerical_error=z['numerical_error'],decision_content_sha256=phash(z),current_reference21=z['current_reference21'],proposed_reference21=z['proposed_reference21'],execution_receipt=z['execution_receipt'],trace_equals_actual_receipt=diff(cache[v]['executed_reference21'][i],z['execution_receipt']['actual_reference21'])['exact'],saved_input_acquisition_time=cache[v]['public_selected'][i].get('sample_time'),native_current_status=None if z.get('native_current') is None else z['native_current'].get('status'),native_proposed_status=None if z.get('native_proposed') is None else z['native_proposed'].get('status'))
  if h is not None:
   d=h['diagnostics'];q.update(status=h['status'],completed_prefix=h['completed_prefix'],fully_converged=h['fully_converged'],failed_layer=h['failed_layer'],normal_witness_eta_P=d['normal_witness_optimal_eta_P'],normal_witness_xA=d['certified_L1_extended_anchor'],motion_budget=d['motion_task']['normalized_absolute_budget_M'],joint_certificates=[cert_summary(c) for c in h['certified_prefix']],emitted_score_available=z['emitted_hierarchy_score']['available'],reference_units_reconstruction=diff(np.asarray(reference_rates('release'))*.005,d['reference_step_units21']))
  summaries[v]=q
 row['saved_decisions']=summaries
 if z0.get('hierarchy') is not None and z1.get('hierarchy') is not None:
  k0=kwargs(z0);k1=kwargs(z1);row['same_task_call_inputs']={key:diff(k0[key],k1[key]) for key in k0};d0=z0['hierarchy']['diagnostics'];d1=z1['hierarchy']['diagnostics'];row['normal_witness_comparison']={key:diff(d0[key],d1[key]) for key in ('normal_witness_optimal_eta_P','certified_L1_extended_anchor')};row['motion_budget_comparison']=diff(d0['motion_task']['normalized_absolute_budget_M'],d1['motion_task']['normalized_absolute_budget_M']);row['joint_attained_comparison']=diff(z0['hierarchy']['certified_prefix'][0]['attained'],z1['hierarchy']['certified_prefix'][0]['attained'])
 observed.append(row)
# Fixed local mathematical CF: first roundoff record and first material-difference input from BOTH trajectories.
cf=[]
for input_version,tick in [('v22',2726),('v22',2728),('v23',2728)]:
 z=records[input_version][tick];kw=kwargs(z);solutions={}
 for solver_version,module in [('v22_f304',old),('v23_4b2ac6',new)]:
  print('CF_START '+input_version+' '+str(kw['time_s'])+' '+solver_version,flush=True);start=time.perf_counter();r=module.solve_contact_exit(**kw);elapsed=time.perf_counter()-start;d=r.diagnostics;scale=kw['reference_rates21']*.005;current=kw['current_reference21'];lo=np.asarray(z['actual_reference_lower21']);hi=np.asarray(z['actual_reference_upper21']);raw=current+r.x*scale;proposed=np.minimum(hi,np.maximum(lo,raw));rounding=64*np.finfo(float).eps*np.maximum(1,np.maximum(abs(raw),abs(proposed)));score=score_emitted_reference(r,current21=current,proposed21=proposed,scale21=scale) if r.available else None;verifications=[verify_certificate(c) for c in r.certified_prefix];ext=np.asarray(d.get('extended_normalized_x',r.x));last=r.certified_prefix[-1] if r.certified_prefix else None;strict=None if last is None else bool(np.all(ext>=np.asarray(last['hard_domain']['lower'])) and np.all(ext<=np.asarray(last['hard_domain']['upper'])))
  q=dict(available=r.available,status=r.status,completed_prefix=r.completed_prefix,fully_converged=r.fully_converged,failed_layer=None if r.failed_layer is None else {'index':r.failed_layer['index'],'name':r.failed_layer['name']},elapsed_s=elapsed,normalized_x21=r.x,actual_reference21_if_emitted=proposed,more_than_roundoff_projection=bool(np.any(abs(raw-proposed)>rounding)),strictbox=strict,original_certificates_independently_checked=verifications,emitted_score_available=None if score is None else score['available'],normal_witness_eta_P=d.get('normal_witness_optimal_eta_P'),normal_witness_xA=d.get('certified_L1_extended_anchor'),motion_budget=d.get('motion_task',{}).get('normalized_absolute_budget_M'),joint_certificates=[cert_summary(c) for c in r.certified_prefix],same_source_saved_reference_difference=diff(z['proposed_reference21'],proposed) if input_version==solver_version.split('_')[0] else None)
  solutions[solver_version]=q;print('CF_DONE '+input_version+' '+str(kw['time_s'])+' '+solver_version+' '+str(elapsed),flush=True)
 c0=solutions['v22_f304'];c1=solutions['v23_4b2ac6'];comparison={key:diff(c0.get(key),c1.get(key)) for key in ('normal_witness_eta_P','normal_witness_xA','motion_budget','normalized_x21','actual_reference21_if_emitted')}
 comparison['joint_cost_delta']=None if not c0['joint_certificates'] or not c1['joint_certificates'] else c1['joint_certificates'][0]['squared_loss']-c0['joint_certificates'][0]['squared_loss'];comparison['joint_original_A_same']=bool(c0['joint_certificates'] and c1['joint_certificates'] and c0['joint_certificates'][0]['A_sha256']==c1['joint_certificates'][0]['A_sha256']);comparison['joint_original_b_same']=bool(c0['joint_certificates'] and c1['joint_certificates'] and c0['joint_certificates'][0]['b_sha256']==c1['joint_certificates'][0]['b_sha256']);comparison['joint_motion_domain_h_same']=bool(c0['joint_certificates'] and c1['joint_certificates'] and c0['joint_certificates'][0]['h_sha256']==c1['joint_certificates'][0]['h_sha256'])
 cf.append(dict(input_trajectory=input_version,time_s=kw['time_s'],all_kwargs=kw,input_sha256=phash(kw),solutions=solutions,same_input_backend_comparison=comparison))
assert source_identity(['ratio_effort'])==source and numerical_identity()==identity and not forbidden
out=ROOT/'review/v23_joint_friction_first_divergence_v1.json';assert not out.exists();j=dict(schema='v23_joint_friction_first_divergence_v1',complete=True,context_key=['friction_drop','transport',619742],source_sha256=source,numerical_identity=identity,input_sha256=input_pins,first_executed_reference_divergence=first,observed_nearby_records=observed,local_same_public_input_counterfactuals=cf,checker_sha256=sha(__file__),forbidden_calls=forbidden,physics_steps=0,model_constructions=0,source_changes=0,scope='All3201 actual-reference rows compared; four fixed nearby completed release records inspected. Six local mathematical solves use only three saved public task inputs, no engine/object truth/new sensor or physical replay. Downstream different-state solutions are not same-problem evidence; local counterfactuals are NOT physical outcomes or complete rolling causality.')
with out.open('x',encoding='utf8') as f:json.dump(plain(j),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(complete=True,out=str(out),sha256=sha(out),first=plain(first),counterfactual_comparisons=plain([z['same_input_backend_comparison'] for z in cf])),ensure_ascii=False),flush=True)

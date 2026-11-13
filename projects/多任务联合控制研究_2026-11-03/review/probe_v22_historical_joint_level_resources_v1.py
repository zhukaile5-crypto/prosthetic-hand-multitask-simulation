"""Fixed16 historical joint LS: current frozen f304, one Level, compact resource evidence."""
from pathlib import Path
from collections import Counter
from dataclasses import is_dataclass,fields
from collections.abc import Mapping
import sys,json,hashlib,time,gc,ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
from v22_joint_task_fusion.finite_solver import solve_hierarchy,Level
from v21_original_matrix_kkt_verifier_v1 import verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf8'))
SRC=ROOT/'v22_joint_task_fusion/finite_solver.py';EXPECTED='f30419b471c614b6ef81101ac52ce6679013fb2ed0caef58225b5699dc6c5c3b';assert sha(SRC)==EXPECTED
artifact=ROOT/'review/v22_joint_active_face_refinement_v1.json';corpuspath=ROOT/'review/v21_numerical_corpus_v1.json';old=read(artifact);corpus=read(corpuspath);assert len(old['rows'])==16 and sha(corpuspath)==old['corpus_sha256']
index={tuple(s['context_key']):s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion'};assert len(index)==16
kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p;process=kernel.GetCurrentProcess()
class PMC(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
def memory():
 m=PMC();m.cb=ctypes.sizeof(m)
 if not psapi.GetProcessMemoryInfo(process,ctypes.byref(m),m.cb):return None
 return dict(working_set_bytes=int(m.WorkingSetSize),peak_working_set_cumulative_bytes=int(m.PeakWorkingSetSize),pagefile_usage_bytes=int(m.PagefileUsage))

def array_size(obj):
 seen_nodes=set();seen_arrays=set();counts=dict(unique_ndarray_objects=0,unique_ndarray_nbytes=0,array_occurrences_nbytes=0,array_occurrences=0,container_nodes=0);stack=[obj]
 while stack:
  x=stack.pop()
  if isinstance(x,np.ndarray):
   counts['array_occurrences']+=1;counts['array_occurrences_nbytes']+=int(x.nbytes)
   if id(x) not in seen_arrays:seen_arrays.add(id(x));counts['unique_ndarray_objects']+=1;counts['unique_ndarray_nbytes']+=int(x.nbytes)
  elif is_dataclass(x):
   if id(x) in seen_nodes:continue
   seen_nodes.add(id(x));counts['container_nodes']+=1;stack.extend(getattr(x,f.name) for f in fields(x))
  elif isinstance(x,Mapping):
   if id(x) in seen_nodes:continue
   seen_nodes.add(id(x));counts['container_nodes']+=1;stack.extend(x.values())
  elif isinstance(x,(tuple,list)):
   if id(x) in seen_nodes:continue
   seen_nodes.add(id(x));counts['container_nodes']+=1;stack.extend(x)
 return counts

def bits(a,b):return np.shape(a)==np.shape(b) and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
rows=[];identity=numerical_identity();gc.collect();startmemory=memory();allstart=time.perf_counter();out=ROOT/'review/v22_joint_level_resource_probe_v1.json';assert not out.exists()
for item in old['rows']:
 ctx=tuple(item['context_key']);sample=index[ctx];assert sample['source']==item['source'];first=sample['stage_problems'][0];p=item['original_joint_problem'];A=np.asarray(p['A']);b=np.asarray(p['b']);G=np.asarray(p['G']);h=np.asarray(p['h']);E=np.asarray(p['E']);f=np.asarray(p['f']);lo=np.asarray(first['lower']);hi=np.asarray(first['upper']);x0=np.asarray(sample['failed_stage_problem']['initial_x']);n=len(lo);free=np.flatnonzero(lo!=hi);fixed=np.flatnonzero(lo==hi);P=len(first['b']);box=2*len(free)
 if E.shape==(0,):E=np.empty((0,n))
 assert bits(E,np.eye(n)[fixed]) and bits(f,lo[fixed]);permutation=np.r_[np.arange(P),np.arange(P+box,len(G)),np.arange(P,P+box)].astype(int);inputids=permutation[:len(G)-box];Gin=G[inputids];hin=h[inputids];expectedG=np.vstack((Gin,np.eye(n)[free],-np.eye(n)[free]));expectedh=np.r_[hin,hi[free],-lo[free]];assert bits(expectedG,G[permutation]) and bits(expectedh,h[permutation])
 print('CASE_START '+json.dumps(ctx,ensure_ascii=False),flush=True);before=memory();began=time.perf_counter();result=solve_hierarchy(lower=lo,upper=hi,levels=[Level('joint_normal_and_observed_projection',A,b)],initial=x0,inequality_matrix=Gin,inequality_upper=hin,max_iterations=400,atol=p['atol'],rtol=p['rtol']);elapsed=time.perf_counter()-began;after=memory();attempts=result.levels[0]['attempts'];types=Counter(a['candidate'] for a in attempts);sizes=array_size(result);strict=bool(np.all(result.x>=lo) and np.all(result.x<=hi));domain_exact=False;verification=None
 if result.available:
  cert=result.certified_prefix[-1];k=cert['KKT'];lam=np.zeros(len(G));lam[permutation]=np.asarray(k['full_normalized_inequality_dual']);nu=np.asarray(k['equality_raw_dual']);verification=verify_stage(dict(p,initial_x=x0),result.x,duals=dict(inequality_unit=lam,equality_raw=nu,provenance='f304_full_unit_dual_inverse_row_permutation_to_original_bundle_G'),comparison_x=x0);hd=cert['hard_domain'];domain_exact=bits(hd['original_inequality_matrix_with_box'],G[permutation]) and bits(hd['original_inequality_upper_with_box'],h[permutation]) and bits(hd['fixed_equality_matrix'],E) and bits(hd['fixed_equality_target'],f) and bits(cert['matrix'],A) and bits(cert['target'],b)
 else:verification=verify_stage(p,result.x,comparison_x=x0)
 metric={k:verification.get(k) for k in ('available','kkt_passed','objective','stationarity_inf','gradient_tolerance','complementarity_inf','status','original_problem_sha256')};metric['primal_passed']=verification.get('primal',{}).get('passed');metric['primal_positive_violation_inf']=verification.get('primal',{}).get('inequality_positive_violation_inf')
 row=dict(context_key=list(ctx),available=result.available,fully_converged=result.fully_converged,status=result.status,strictbox=strict,all_original_rows_retained_under_explicit_permutation=domain_exact,original_G_row_permutation_used_by_kernel=permutation.tolist(),input_initial_source='same fixed old joint runner: original old-L3 initial_x, not saved joint candidate/after-face point',dimension=n,normal_rows=P,motion_rows=len(A)-P,solver_elapsed_s=elapsed,attempt_count=len(attempts),attempt_counts=dict(types),FW_direction_count=types['original_polytope_gradient_LP_exact_segment'],dual_LP_calls=sum(bool(a.get('KKT',{}).get('dual_witness',{}).get('lp_called')) for a in attempts),face_refinement_count=types['one_tight_face'],box_new_candidate_count=types['explicit_original_coordinate_box_candidate_after_face'],evidence_array_sizes=sizes,evidence_size_scope='Returned result graph, unique ndarray objects by id; not Python container/JSON/native transient total',process_memory_before=before,process_memory_holding_result=after,independent_original_bundle_verification=metric)
 rows.append(row);print(json.dumps(dict(context=ctx,elapsed_s=elapsed,available=result.available,KKT=metric['kkt_passed'],FW=row['FW_direction_count'],attempts=len(attempts),evidence_bytes=sizes['unique_ndarray_nbytes']),ensure_ascii=False),flush=True)
 del result,attempts
 if 'cert' in locals():del cert
 if 'k' in locals():del k
 if 'hd' in locals():del hd
 gc.collect();row['process_memory_after_result_release']=memory()
summary=dict(assigned=16,assessed=len(rows),accepted=sum(r['available'] for r in rows),independent_KKT=sum(r['independent_original_bundle_verification']['kkt_passed'] is True for r in rows),strictbox=sum(r['strictbox'] for r in rows),exact_original_domains=sum(r['all_original_rows_retained_under_explicit_permutation'] for r in rows),total_FW_directions=sum(r['FW_direction_count'] for r in rows),total_attempts=sum(r['attempt_count'] for r in rows))
for key in ('solver_elapsed_s','FW_direction_count','attempt_count'):
 vals=[r[key] for r in rows];summary[key]=dict(minimum=min(vals),median=float(np.median(vals)),p95=float(np.percentile(vals,95)),maximum=max(vals))
bs=[r['evidence_array_sizes']['unique_ndarray_nbytes'] for r in rows];summary['unique_ndarray_nbytes']=dict(total=sum(bs),median=float(np.median(bs)),maximum=max(bs))
payload=dict(schema='v22_historical_joint_level_resources_v1',complete=True,summary=summary,rows=rows,finite_solver_sha256=sha(SRC),finite_solver_unchanged=sha(SRC)==EXPECTED,numerics_sha256=sha(ROOT/'v22_joint_task_fusion/numerics.py'),numerical_identity=identity,numerical_identity_unchanged=numerical_identity()==identity,input_joint_artifact_sha256=sha(artifact),corpus_sha256=sha(corpuspath),checker_sha256=sha(__file__),process_memory_initial=startmemory,process_memory_final=memory(),total_wall_s=time.perf_counter()-allstart,host_load_scope='Other root physical workers concurrently active; single-host diagnostic only, not benchmark speed comparison or 5ms realtime certificate',physical_outputs_read=False,production_changes=0,model_constructions=0,physics_steps=0)
with out.open('x',encoding='utf8') as fp:json.dump(payload,fp,ensure_ascii=False,indent=2,allow_nan=False);fp.write('\n')
print(json.dumps(dict(complete=True,summary=summary,out=str(out),sha256=sha(out)),ensure_ascii=False),flush=True)

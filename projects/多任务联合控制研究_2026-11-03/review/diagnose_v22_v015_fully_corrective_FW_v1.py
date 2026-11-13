"""One fixed v015 problem, multi-step fully-corrective Frank-Wolfe diagnostic."""
from pathlib import Path
import sys,json,hashlib,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment
ensure_numeric_environment()
from scipy.optimize import linprog,minimize
from v21_original_matrix_kkt_verifier_v1 import canonical_problem,verify_stage
read=lambda p:json.loads(Path(p).read_text('utf8'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
src=ROOT/'review/v21_numerical_corpus_v1.json';corpus=read(src);sample=next(s for s in corpus['samples'] if s['sample_id']=='v21_numeric_015');p=sample['failed_stage_problem'];x=np.asarray(p['initial_x']);q=canonical_problem(p,x);n=len(x);G,h,E,f=q['G_unit'],q['h_unit'],q['E'],q['f'];A,b=q['A'],q['b']

def certify(x):
 v=verify_stage(p,x);grad=np.asarray(v['gradient']);tx=v['primal']['coordinate_tolerance'];res=G@x-h;active=np.flatnonzero((abs(res)<=tx)&np.any(G!=0,axis=1));D=np.column_stack((G[active].T,E.T));gs=max(1.,float(np.max(abs(grad),initial=0.)));colscale=np.max(abs(D),axis=0,initial=0.);good=colscale>0;Ds=D[:,good]/colscale[good][None,:];k=Ds.shape[1];ineqgood=good[:len(active)].sum();bound=[(0.,None) if i<len(active) else (None,None) for i in np.flatnonzero(good)]
 lp=linprog(np.r_[np.zeros(k),1.],A_ub=np.vstack((np.column_stack((Ds,-np.ones(n))),np.column_stack((-Ds,-np.ones(n))))),b_ub=np.r_[-grad/gs,grad/gs],bounds=bound+[(0.,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'maxiter':400})
 witness=np.zeros(D.shape[1]);lam=np.zeros(len(h));nu=np.zeros(len(f))
 if lp.x is not None:witness[good]=(lp.x[:k]*gs)/colscale[good];lam[active]=witness[:len(active)];nu=witness[len(active):]
 dual=dict(inequality_unit=lam,equality_raw=nu,provenance='independent_scaled_column_HiGHS_cone_dual_current_candidate_only');return verify_stage(p,x,duals=dual,comparison_x=p['initial_x']),dual

# Saved original certificate is evaluated as an AFTER-the-fact positive control,
# never used to choose atoms, directions, coefficients or stop the iterations.
known=p['original_certificate'];assert known is not None
K=known['KKT'];basis=np.asarray(K['equality_basis']);mu=np.asarray(K['original_loop_multipliers'])[:len(basis)];nu=np.linalg.lstsq(E.T,basis.T@mu,rcond=None)[0];lam=np.zeros(len(h));nonzero=np.any(q['G']!=0,axis=1);lam[nonzero]=np.asarray(K['full_normalized_inequality_dual']);knowncheck=verify_stage(p,known['x'],duals=dict(inequality_unit=lam,equality_raw=nu,provenance='original_saved_L4_certificate_positive_control_not_algorithm_input'))
initial,initialdual=certify(x);atoms=[x.copy()];weights=np.array([1.]);records=[];accepted=initial['kkt_passed'];budget=64;began=time.perf_counter()
for iteration in range(1,budget+1):
 if accepted:break
 basecheck=verify_stage(p,x);grad=np.asarray(basecheck['gradient']);gs=max(1.,float(np.max(abs(grad),initial=0.)));linear=linprog(grad/gs,A_ub=G,b_ub=h,A_eq=E if len(E) else None,b_eq=f if len(E) else None,bounds=[(None,None)]*n,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'maxiter':400})
 if linear.x is None:records.append(dict(iteration=iteration,status='linear_oracle_no_point',library_message=linear.message));break
 vertex=np.asarray(linear.x);vertexcheck=verify_stage(p,vertex)
 if not vertexcheck['primal']['passed']:records.append(dict(iteration=iteration,status='linear_oracle_point_original_primal_rejected',verification=vertexcheck));break
 direction=vertex-x;Ad=A@direction;rr=A@x-b;den=float(Ad@Ad);alpha=0. if den==0 else min(1.,max(0.,-float(rr@Ad)/den));line=x+alpha*direction;linecheck=verify_stage(p,line)
 atoms.append(vertex.copy());weights=np.r_[(1.-alpha)*weights,alpha]
 # Exact same original LS on conv(atoms). Remove only the constant objective
 # baseline in candidate arithmetic, not any relative cost or original row.
 V=np.column_stack(atoms);center=V[:,0];DV=V[:,1:]-center[:,None];R=A@DV;r0=A@center-b;start=weights[1:].copy()
 fun=lambda z:float((R@z)@(R@z)+2.*r0@(R@z));jac=lambda z:2.*R.T@(R@z+r0)
 sqp=minimize(fun,start,jac=jac,method='SLSQP',bounds=[(0.,1.)]*len(start),constraints=[dict(type='ineq',fun=lambda z:1.-float(np.sum(z)),jac=lambda z:-np.ones(len(z)))],options={'ftol':1e-12,'maxiter':400,'disp':False})
 rawweights=np.r_[1.-float(np.sum(sqp.x)),sqp.x];allowance=64*np.finfo(float).eps*max(1.,float(np.max(abs(rawweights),initial=0.)));repaired=False
 if np.min(rawweights,initial=0.) < -allowance:
  candidate=line;w=weights;candidate_source='exact_line_only_corrective_weights_rejected'
 else:
  w=np.maximum(rawweights,0.);w=w/float(np.sum(w));candidate=V@w;repaired=not np.array_equal(w,rawweights);candidate_source='fully_corrective_simplex_same_LS_constant_removed'
 value,dual=certify(candidate);accepted=value['kkt_passed'];objective_delta=value['objective']-basecheck['objective'];gap=float(grad@(x-vertex));rec=dict(iteration=iteration,status='original_KKT_verified' if accepted else 'continue_fixed_budget',candidate_source=candidate_source,linear_success=bool(linear.success),linear_status=int(linear.status),linear_vertex=vertex,linear_original_primal=vertexcheck['primal'],FrankWolfe_gap_original=gap,line_alpha=alpha,line_original_objective=linecheck['objective'],simplex_library_success=bool(sqp.success),simplex_library_status=int(sqp.status),simplex_iterations=int(sqp.nit),simplex_raw_weights=rawweights,simplex_weights=w,simplex_roundoff_repair=repaired,atom_count=len(atoms),candidate=candidate,objective_delta_this_iteration=objective_delta,verification=value,dual_witness=dual);records.append(rec);x=candidate;weights=w
 print(json.dumps(dict(iteration=iteration,objective=value['objective'],gap=gap,stationarity=value.get('stationarity_inf'),KKT=accepted,atoms=len(atoms)),ensure_ascii=False),flush=True)
final,finaldual=certify(x);out=ROOT/'review/v22_v015_fully_corrective_FW_diagnostic_v1.json';r=dict(schema='v22_v015_fully_corrective_FW_diagnostic_v1',complete=True,sample_id=sample['sample_id'],context_key=sample['context_key'],source=sample['source'],selection_kind=sample['selection_kind'],original_problem=p,known_saved_final_certificate_positive_control=knowncheck,known_saved_point_not_used_by_algorithm=True,initial_verification=initial,initial_dual_witness=initialdual,iterations=records,outer_iteration_budget=budget,final_x=x,final_verification=final,final_dual_witness=finaldual,accepted_original_KKT=final['kkt_passed'],final_minus_saved_cert_objective=final['objective']-knowncheck['objective'],same_original_problem_hash=initial['original_problem_sha256']==final['original_problem_sha256']==knowncheck['original_problem_sha256'],elapsed_s=time.perf_counter()-began,corpus_sha256=sha(src),independent_verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),checker_sha256=sha(__file__),production_changes=0,physics_steps=0,scope='One v015 historical case, fixed64-step fully-corrective conditional-gradient candidate. No regularizer, changed constraint/tolerance or physics. Saved old final point only offline positive-control score.')
with out.open('x',encoding='utf8') as f:json.dump(plain(r),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(accepted_original_KKT=r['accepted_original_KKT'],iterations=len(records),out=str(out),sha256=sha(out),objective_delta_from_initial=final['comparison']['objective_delta'],final_minus_saved_cert_objective=r['final_minus_saved_cert_objective']),ensure_ascii=False),flush=True)

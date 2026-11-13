"""Offline independent candidates for FROZEN V21 finite tasks, not a controller.

No production solver/physics import. All original rows and objectives survive.
SVD compresses equalities for SLSQP only; final checking uses every original row.
Dual recovery certifies the new point, never reconstructs an old certificate.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.optimize import linprog, minimize, lsq_linear
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v21_original_matrix_kkt_verifier_v1 import canonical_problem, verify_stage

ROOT = Path(__file__).resolve().parents[1]

def plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [plain(x) for x in v]
    return v

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def equality_basis(E, f, n):
    if not len(E): return np.empty((0,n)), np.empty(0), {'original_rows':0,'rank':0}
    u,s,vt = np.linalg.svd(E, full_matrices=False)
    cutoff = max(E.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank = int(np.sum(s>cutoff))
    Er = vt[:rank]; fr = (u[:,:rank].T@f)/s[:rank]
    drift = float(np.max(abs(E@(Er.T@fr)-f), initial=0.))
    # This is solver input compression only. No original-row acceptance changes.
    if drift > 1e-8*max(1.,float(np.max(abs(f), initial=0.))):
        raise ValueError('Inconsistent equality compression; original problem retained.')
    return Er,fr,{'original_rows':len(E),'rank':rank,'svd_cutoff':cutoff,'compression_residual_inf':drift}

def recover_dual(problem, x):
    p = canonical_problem(problem, x)
    gradient = p['c'] if p['kind']=='linear' else 2*p['A'].T@(p['A']@x-p['b'])
    tx = p['atol']+p['rtol']*max(1.,float(np.max(abs(x),initial=0.)))
    residual = p['G_unit']@x-p['h_unit']
    active = np.flatnonzero((abs(residual)<=tx) & (np.max(abs(p['G_unit']),axis=1,initial=0.)>0))
    W = np.column_stack((p['G_unit'][active].T, p['E'].T))
    dual = np.zeros(len(p['h'])); nu = np.zeros(len(p['f']))
    note = {'active_rows':active.tolist(),'method':'scipy.optimize.lsq_linear bvls; dual only; no primal change'}
    if W.shape[1]:
        lo = np.r_[np.zeros(len(active)),np.full(len(nu),-np.inf)]
        hi = np.full(W.shape[1],np.inf)
        r = lsq_linear(W, -gradient, bounds=(lo,hi), method='bvls', lsq_solver='exact', tol=1e-14, max_iter=400)
        dual[active] = r.x[:len(active)]; nu = r.x[len(active):]
        note.update(status=int(r.status),success=bool(r.success),iterations=int(r.nit),optimality=float(r.optimality))
    return {'inequality_unit':dual,'equality_raw':nu,
        'provenance':'recovered_by_lsq_linear_for_current_independent_candidate_not_historical'},note

def solve_problem(problem, *, scalar_withdrawal=False):
    x0 = np.asarray(problem['initial_x'],float)
    p = canonical_problem(problem,x0)
    original = verify_stage(problem,x0)
    if not original.get('available') or not original['primal']['passed']:
        return {'status':'ORIGINAL_ANCHOR_NOT_FEASIBLE','original_anchor':original,'candidate_available':False}
    original_dual,original_dual_note = recover_dual(problem,x0)
    original_with_dual = verify_stage(problem,x0,duals=original_dual)
    G,h,E,f = p['G_unit'],p['h_unit'],p['E'],p['f']
    started = time.perf_counter()
    proof = None
    if scalar_withdrawal:
        if p['kind']!='least_squares' or p['A'].shape[0]!=1:
            raise ValueError('Withdrawal LP needs the actual scalar LS objective.')
        lo=np.asarray(problem['lower'],float); hi=np.asarray(problem['upper'],float)
        row=p['A'][0]; boxupper=float(np.sum(np.where(row>=0,row*hi,row*lo)))
        allowance=32*np.finfo(float).eps*max(1.,abs(boxupper),abs(float(p['b'][0])))
        if p['b'][0]+allowance < boxupper:
            raise ValueError('LS target not an upper bound of original box; LP equivalence unproven.')
        proof={'target':float(p['b'][0]),'actual_box_upper':boxupper,'rounding_allowance':allowance,
            'equivalence':'min (a.x-b)^2 is max a.x when b bounds the entire actual box'}
        # Explicit unrestricted coordinates: every box row is already in G/E.
        r=linprog(-row,A_ub=G if len(G) else None,b_ub=h if len(G) else None,
            A_eq=E if len(E) else None,b_eq=f if len(E) else None,
            bounds=[(None,None)]*p['n'],method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,
                'ipm_optimality_tolerance':1e-10,'maxiter':400})
        method='HiGHS LP, original scalar LS checked afterwards'
        compression=None
    else:
        Er,fr,compression=equality_basis(E,f,p['n'])
        constraints=[]
        if len(G):constraints.append({'type':'ineq','fun':lambda x:h-G@x,'jac':lambda x:-G})
        if len(Er):constraints.append({'type':'eq','fun':lambda x:Er@x-fr,'jac':lambda x:Er})
        if p['kind']=='linear':fun=lambda x:float(p['c']@x); jac=lambda x:p['c']
        else:
            fun=lambda x:float((p['A']@x-p['b'])@(p['A']@x-p['b']))
            jac=lambda x:2*p['A'].T@(p['A']@x-p['b'])
        r=minimize(fun,x0,method='SLSQP',jac=jac,constraints=constraints,
            options={'ftol':1e-12,'maxiter':400,'disp':False})
        method='SLSQP original convex objective and analytic gradient'
    elapsed=time.perf_counter()-started
    result={'method':method,'library_success':bool(r.success),'library_status':int(r.status),
        'library_message':str(r.message),'iterations':int(r.nit),'candidate_available':r.x is not None,
        'primal_solver_elapsed_s':elapsed,'scalar_LP_equivalence':proof,'equality_solver_compression':compression,
        'original_anchor':original,'original_anchor_independent_KKT':original_with_dual,
        'original_anchor_dual_recovery':original_dual_note,
        'no_physics':True,'not_realtime_or_physical_superiority_evidence':True}
    if r.x is None:return dict(result,status='NO_INDEPENDENT_PRIMAL_CANDIDATE')
    x=np.asarray(r.x,float)
    primal=verify_stage(problem,x,comparison_x=x0)
    result.update(x=x,primal_before_dual=primal)
    if not primal.get('available') or not primal['primal']['passed']:
        return dict(result,status='INDEPENDENT_CANDIDATE_NOT_ORIGINAL_PRIMAL_FEASIBLE')
    witness,note=recover_dual(problem,x)
    verified=verify_stage(problem,x,duals=witness,comparison_x=x0)
    result.update(dual_recovery=note,dual_witness=witness,verification=verified,
        status='INDEPENDENT_ORIGINAL_PROBLEM_KKT_VERIFIED' if verified['kkt_passed'] else 'FEASIBLE_BUT_KKT_NOT_VERIFIED')
    return result

def run(corpus_name,out_name):
    corpus_path=ROOT/corpus_name
    assert not (ROOT/out_name).exists(), 'Use a fresh output name; preserve all previous attempts.'
    corpus=json.loads(corpus_path.read_text(encoding='utf-8'))
    rows=[]
    for sample in corpus['samples']:
        problem=sample.get('failed_stage_problem',sample.get('stage_problem'))
        label=sample.get('sample_id',sample.get('source',{}))
        try:
            row=solve_problem(problem,scalar_withdrawal=sample.get('layer_name',problem.get('layer_name'))=='scalar_pressure_share_normal_withdrawal')
        except Exception as exc:
            row={'status':'ORACLE_ERROR','candidate_available':False,'error':repr(exc),'traceback':traceback.format_exc()}
        row.update(sample_id=label,sample_role=sample.get('sample_role',sample.get('selection_kind')),context_key=sample.get('context_key'),
            layer_name=sample.get('layer_name',problem.get('layer_name')),source=sample.get('source'),
            original_is_failed_layer=problem.get('is_original_failed_layer'),
            original_anchor_already_KKT=bool(row.get('original_anchor_independent_KKT',{}).get('kkt_passed',False)))
        rows.append(row)
        print(json.dumps({'context':row['context_key'],'layer':row['layer_name'],'status':row['status'],
            'original_anchor_already_KKT':row['original_anchor_already_KKT']},ensure_ascii=False),flush=True)
    result={'schema':'v21_same_problem_offline_oracle_v1','corpus':corpus_name,'corpus_sha256':sha(corpus_path),
        'oracle_sha256':sha(__file__),'verifier_sha256':sha(Path(__file__).with_name('v21_original_matrix_kkt_verifier_v1.py')),
        'scipy_version':scipy.__version__,'rows':rows,'counts':dict(Counter(x['status'] for x in rows)),
        'assigned':len(corpus['samples']),'assessed':len(rows),'physics_steps':0,'production_changes':0,
        'original_anchor_already_KKT_count':sum(r['original_anchor_already_KKT'] for r in rows),
        'official_method_references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html',
            'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html'],
        'scope':'Selected historical finite tasks, not entire1377, not a new rollout, no solver-status-only acceptance.'}
    (ROOT/out_name).write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':out_name,'counts':result['counts'],'assessed':len(rows)},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();run(args.corpus,args.out)

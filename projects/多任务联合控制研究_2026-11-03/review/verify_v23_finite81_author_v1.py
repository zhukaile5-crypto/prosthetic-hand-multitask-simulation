"""All fixed65+historical joint16, original matrices; no model/physics."""
from pathlib import Path
from unittest.mock import patch
from dataclasses import is_dataclass
from collections import Counter
import argparse, hashlib, importlib.util, json, math, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v23_task_execution import finite_solver as fs
from v22_joint_task_fusion.numerics import numerical_identity

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(x):
    if is_dataclass(x):return plain(vars(x))
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def load(p):
    s=importlib.util.spec_from_file_location('independent_v23_original_KKT',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def adapt(p,lower,upper):
    """Preserve every original hard row; explicitly map box row permutation."""
    n=len(lower);G=np.asarray(p['G'],float).reshape(-1,n);h=np.asarray(p['h'],float)
    E=np.asarray(p['E'],float).reshape(-1,n);f=np.asarray(p['f'],float)
    fixed=np.flatnonzero(lower==upper);free=np.flatnonzero(lower!=upper)
    assert np.array_equal(E[:len(fixed)],np.eye(n)[fixed]) and np.array_equal(f[:len(fixed)],lower[fixed])
    expected=np.vstack((np.eye(n)[free],-np.eye(n)[free]));bounds=np.r_[upper[free],-lower[free]]
    used=set();box=[]
    for row,bound in zip(expected,bounds):
        matches=[i for i in range(len(G)) if i not in used and h[i]==bound and np.array_equal(G[i],row)]
        if not matches:raise ValueError('An original true box row is absent')
        box.append(matches[0]);used.add(matches[0])
    rest=[i for i in range(len(G)) if i not in used];order=np.asarray(rest+box,int)
    return dict(inequality_matrix=G[rest],inequality_upper=h[rest],equality_matrix=E[len(fixed):],equality_target=f[len(fixed):]),order

def main(out):
    if out.exists():raise FileExistsError(out)
    prod=('v23_task_execution/__init__.py','v23_task_execution/finite_solver.py','v22_joint_task_fusion/numerics.py')
    before={p:sha(ROOT/p) for p in prod};oldcore=ROOT/'v22_joint_task_fusion/finite_solver.py';oldhash=sha(oldcore)
    assert oldhash=='f30419b471c614b6ef81101ac52ce6679013fb2ed0caef58225b5699dc6c5c3b'
    corpuspath=ROOT/'review/v21_numerical_corpus_v1.json';jointpath=ROOT/'review/v22_joint_active_face_refinement_v1.json'
    assert sha(corpuspath)=='408168f1c54ace53d253d15ac35d007294779578ad0370a9d995147c5f127513'
    assert sha(jointpath)=='66073d44b3c6c04a9af849ae735e35d4df2b3239b55ff8a5e2a40383e32035b4'
    corpus=read(corpuspath);joints=read(jointpath);verifier=load(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    identity=numerical_identity();checks=[];rows=[]
    def ck(n,b,detail=None):checks.append(dict(name=n,passed=bool(b),detail=detail))
    # Small general interface/proof guards, not case-dependent tuning.
    def solve(lo,hi,A,b,x,**kw):return fs.solve_hierarchy(lower=lo,upper=hi,initial=x,levels=[fs.Level('analytic',np.asarray(A,float),b)],**kw)
    r=solve([-1.,-1.],[1.,1.],np.eye(2),[.2,-.3],[0.,0.]);ck('nonzero analytic LS',r.fully_converged and np.max(abs(r.x-[.2,-.3]))<1e-8)
    r=solve([.3,-1.],[.3,1.],np.eye(2),[.7,.4],[.3,0.]);ck('literal fixed axis',r.fully_converged and r.x[0]==.3)
    r=solve([0.],[1.],[[1.]],[-1.],[0.]);ck('one-sided original KKT anchor proof',r.fully_converged and r.x[0]==0. and r.diagnostics['FW_direction_count']==0)
    r=solve([-1.,-1.],[1.,1.],np.eye(2),[1.,1.],[0.,0.],inequality_matrix=[[1.,1.]],inequality_upper=[.1]);ck('coupled original hard row',r.fully_converged and np.max(abs(r.x-[.05,.05]))<1e-8)
    r=solve([-1.,-1.],[1.,1.],np.empty((0,2)),[],[.1,.2]);ck('known empty P0 flat objective',r.fully_converged and np.array_equal(r.x,[.1,.2]))
    r=fs.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],initial=[0.,0.],levels=[fs.Level('first',[[1.,0.]],[.2]),fs.Level('second',[[0.,1.]],[.4]),fs.Level('conflicting',np.eye(2),[.9,.9])]);ck('preceding attained remains frozen',r.fully_converged and np.max(abs(r.x-[.2,.4]))<1e-8)
    def unavailable(*a,**k):raise RuntimeError('author injected no candidate')
    with patch.object(fs,'_slsqp_candidate',unavailable),patch.object(fs,'_gradient_segment_candidate',unavailable),patch.object(fs,'_refine_once',unavailable):
        r=fs.solve_hierarchy(lower=[-1.],upper=[1.],initial=[0.],levels=[fs.Level('known flat',[[0.]],[0.]),fs.Level('unavailable',[[1.]],[.5])])
    ck('both candidate branches unavailable return only true prior prefix',r.available and not r.fully_converged and r.completed_prefix==1 and r.x[0]==0. and r.failed_layer['index']==1)
    with patch.object(fs,'_dual_witness',lambda *a,**k:(None,None,dict(author_injected_missing_dual=True))):r=solve([-1.],[1.],[[1.]],[.5],[0.])
    ck('candidate status never replaces original KKT witness',not r.available and r.x[0]==0.)
    k=fs._certify(np.array([[0.]]),np.array([0.]),np.array([-1e-12]),np.array([[1.],[-1.]]),np.array([1.,0.]),np.empty((0,1)),np.empty(0),(),1e-9,1e-8,400,lower=np.array([0.]),upper=np.array([1.]))
    ck('strict box independent of unchanged tx',k['checks']['original_hard_primal'] and not k['certified'])

    selected=[]
    for s in corpus['samples']:
        p=s['failed_stage_problem'];selected.append(dict(group='original65',id=s['sample_id'],context=s['context_key'],problem=p,lower=np.asarray(p['lower']),upper=np.asarray(p['upper']),initial=p['initial_x'],max_iterations=p['max_iterations']))
    sourceindex={tuple(s['context_key']):s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion'}
    for i,old in enumerate(joints['rows']):
        s=sourceindex[tuple(old['context_key'])];assert s['source']==old['source']
        selected.append(dict(group='joint16',id='joint16_'+str(i),context=old['context_key'],problem=old['original_joint_problem'],lower=np.asarray(s['stage_problems'][0]['lower']),upper=np.asarray(s['stage_problems'][0]['upper']),initial=s['failed_stage_problem']['initial_x'],max_iterations=400))
    assert len(selected)==81
    for item in selected:
        p=item['problem'];lo=item['lower'];hi=item['upper'];kw,order=adapt(p,lo,hi)
        start=time.perf_counter_ns()
        result=fs.solve_hierarchy(lower=lo,upper=hi,initial=item['initial'],levels=[fs.Level(p.get('layer_name','historical_joint_same_problem'),np.asarray(p['A']),p['b'])],max_iterations=item['max_iterations'],atol=p['atol'],rtol=p['rtol'],**kw)
        elapsed=(time.perf_counter_ns()-start)/1e6;independent=None;exact=False;strict=False;dual=None
        if result.available:
            c=result.certified_prefix[-1];domain=c['hard_domain'];kk=c['KKT']
            # JSON [] loses ONLY the already-known zero-row E shape. No
            # unknown matrix is padded or an equality row removed.
            original_E=np.asarray(p['E'],float).reshape(len(p['f']),len(lo))
            exact=bool(np.array_equal(domain['original_inequality_matrix_with_box'],np.asarray(p['G'])[order]) and np.array_equal(domain['original_inequality_upper_with_box'],np.asarray(p['h'])[order]) and np.array_equal(domain['fixed_equality_matrix'],original_E) and np.array_equal(domain['fixed_equality_target'],p['f']) and np.array_equal(c['matrix'],p['A']) and np.array_equal(c['target'],p['b']))
            lam=np.zeros(len(order));lam[order]=kk['full_normalized_inequality_dual']
            dual=dict(inequality_unit=lam,equality_raw=kk['equality_raw_dual'],provenance='current V23 dual mapped back to every original row order, no old witness')
            independent=verifier.verify_stage(p,result.x,duals=dual,comparison_x=item['initial'])
            strict=bool(np.all(result.x>=lo) and np.all(result.x<=hi))
        attempts=[]
        for a in result.levels[0]['attempts']:
            kk=a.get('KKT',{});gn=a.get('generation',{})
            attempts.append(dict(candidate=a['candidate'],certified=kk.get('certified',False),checks=kk.get('checks'),error=a.get('error'),
                library_status=gn.get('library_status'),library_iterations=gn.get('iterations'),FW_iteration=gn.get('iteration'),FW_gap=gn.get('linear_FW_gap'),alpha=gn.get('segment_alpha'),face_branch=gn.get('candidate_branch')))
        verified=bool(result.available and exact and strict and independent['kkt_passed'])
        rows.append(dict(group=item['group'],sample_id=item['id'],context_key=item['context'],available=result.available,fully_converged=result.fully_converged,
            original_problem_preserved=exact,strict_box=strict,independent_original_KKT=verified,
            original_problem_sha256=verifier.verify_stage(p,item['initial'])['original_problem_sha256'],
            original_inequality_row_permutation=order,original_row_order_is_unchanged=np.array_equal(order,np.arange(len(order))),
            x=result.x,dual=dual,independent={k:independent.get(k) for k in ('kkt_passed','objective','stationarity_inf','gradient_tolerance','complementarity_inf','primal')} if independent else None,
            candidate_source=None if not result.available else result.certified_prefix[-1]['candidate_source'],attempts=attempts,
            FW_direction_count=result.diagnostics['FW_direction_count'],face_refinement_count=result.diagnostics['face_refinement_count'],
            projection_count=result.diagnostics['box_candidate_projection_count'],elapsed_ms=elapsed,status=result.status))
    ck('all original65 retained and independently certified',len([r for r in rows if r['group']=='original65'])==65 and all(r['independent_original_KKT'] for r in rows if r['group']=='original65'))
    ck('all historical joint16 retained and independently certified',len([r for r in rows if r['group']=='joint16'])==16 and all(r['independent_original_KKT'] for r in rows if r['group']=='joint16'))
    ck('two predeclared branches at most one face each',all(r['face_refinement_count']<=2 and len({a['face_branch'] for a in r['attempts'] if a['face_branch']})==r['face_refinement_count'] for r in rows))
    after={p:sha(ROOT/p) for p in prod};ck('V23 and numerical dependency unchanged during run',before==after and numerical_identity()==identity)
    ck('frozen V22 f304 untouched',sha(oldcore)==oldhash)
    summaries={}
    for group in ('original65','joint16','all81'):
        rs=[r for r in rows if group=='all81' or r['group']==group];ts=[r['elapsed_ms'] for r in rs]
        summaries[group]=dict(assigned=len(rs),assessed=len(rs),available=sum(r['available'] for r in rs),independent_original_KKT=sum(r['independent_original_KKT'] for r in rs),strict_box=sum(r['strict_box'] for r in rs),FW_directions=sum(r['FW_direction_count'] for r in rs),FW_used_cases=sum(r['FW_direction_count']>0 for r in rs),face_refinements=sum(r['face_refinement_count'] for r in rs),projection_candidates=sum(r['projection_count'] for r in rs),accepted_sources=dict(Counter(r['candidate_source'] for r in rs)),timing_ms=dict(median=float(np.median(ts)),p95=float(np.percentile(ts,95)),maximum=max(ts)))
    report=dict(schema='v23_finite81_author_v1',passed=all(c['passed'] for c in checks),production_source_sha256=after,checker_sha256=sha(__file__),checks=checks,issues=[c['name'] for c in checks if not c['passed']],summaries=summaries,rows=rows,
        input_sha256={'review/v21_numerical_corpus_v1.json':sha(corpuspath),'review/v22_joint_active_face_refinement_v1.json':sha(jointpath)},
        frozen_parent_sha256={'v22_joint_task_fusion/finite_solver.py':oldhash},numerical_identity=identity,
        predeclared_route='All81 same anchor proof -> SLSQP/one exact-coordinate face -> only if still uncertified original max_iterations FW/one FW face; explicit new box candidate independently checked, no post-cert clipping',
        no_original_goals_bounds_rows_or_tolerances_changed=True,physics_steps=0,model_constructions=0,current_B_C_outcomes_read=False,runtime_connected=False,
        timing_scope='Single-host solver walltime, not controlled benchmark or5ms guarantee; no model/physics.81 historical mathematical problems, not81 physical draws.')
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(plain(dict(passed=report['passed'],checks=len(checks),issues=report['issues'],summary=summaries,out=str(out),sha256=sha(out))),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();raise SystemExit(main(a.out))

"""Independent read/array audit of stored V21 task-tradeoff witnesses.
No optimizer, production/model import, FK or physics. Does not recertify LP
duals absent from the persisted report, or call a segment point a QP optimum.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(A,x):return np.asarray([math.fsum(float(a)*float(b) for a,b in zip(row,x)) for row in A])
def loss(A,b,x):
    r=rows(A,x)-b;return math.fsum(float(v)*float(v) for v in r)
def canonical(p,*,linear_gradient=None):
    n=p['variable_count'];G=np.asarray(p['G'],float).reshape((-1,n));h=np.asarray(p['h'],float);E=np.asarray(p['E'],float).reshape((-1,n));f=np.asarray(p['f'],float)
    unit=np.empty_like(G);hu=np.empty_like(h);constant=[]
    for i,(g,v) in enumerate(zip(G,h)):
        scale=max((abs(float(q)) for q in g),default=0.)
        if scale==0:unit[i]=g;hu[i]=v;constant.append(i)
        else:
            a=g/scale;k=math.sqrt(math.fsum(float(q)*float(q) for q in a));unit[i]=a/k;hu[i]=(v/scale)/k
    return dict(n=n,G=G.tolist(),h=h.tolist(),E=E.tolist(),f=f.tolist(),
        A=None if linear_gradient is not None else np.asarray(p['A'],float).tolist(),
        b=None if linear_gradient is not None else np.asarray(p['b'],float).tolist(),
        c=None if linear_gradient is None else linear_gradient.tolist(),kind='least_squares' if linear_gradient is None else 'linear',
        atol=float(p['atol']),rtol=float(p['rtol']),G_unit=unit.tolist(),h_unit=hu.tolist(),constant_rows=constant,
        frozen=[dict(matrix=q['matrix'],attained=q['attained']) for q in p['previous_frozen']])
def phash(p):return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def primal(p,x):
    c=canonical(p);n=p['variable_count'];G=np.asarray(c['G_unit']).reshape((-1,n));h=np.asarray(c['h_unit']);E=np.asarray(c['E']).reshape((-1,n));f=np.asarray(c['f']);tol=p['atol']+p['rtol']*max(1.,float(max(abs(x))))
    gu=max(0.,float(max(rows(G,x)-h,default=0.)));eq=float(max(abs(rows(E,x)-f),default=0.));lo=np.asarray(p['lower']);hi=np.asarray(p['upper']);box=float(max(0.,max(lo-x),max(x-hi)))
    constant=all(c['h'][i]>=0 for i in c['constant_rows'])
    return dict(passed=gu<=tol and eq<=tol and box<=tol and constant,unit_inequality_inf=gu,equality_inf=eq,box_inf=box,tolerance=tol)

def main():
    corpus_path=ROOT/'review/v21_numerical_corpus_v1.json';corpus=read(corpus_path);srcsha=sha(corpus_path)
    sample=[s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion']
    assert len(sample)==16 and len({tuple(s['context_key']) for s in sample})==16
    outputs=[];inputs={str(corpus_path):srcsha};allpass=True
    for name in ('v21_motion_priority_counterfactual_v1','v21_normal_motion_tradeoff_counterfactual_v1'):
        path=ROOT/'review'/f'{name}.json';script=path.with_suffix('.py');d=read(path);inputs[str(path)]=sha(path);inputs[str(script)]=sha(script)
        assert d['corpus_sha256']==srcsha and d['source_sha256']==sha(script) and d['assigned']==d['assessed']==16
        idx={tuple(r['context_key']):r for r in d['rows']};assert len(idx)==16
        for s in sample:
            old=s['failed_stage_problem'];n=old['variable_count'];first=s['stage_problems'][0];r=idx[tuple(s['context_key'])];assert r['source']==s['source']
            if 'priority' in name:
                assert len(old['previous_frozen'])==1 and old['previous_frozen'][0]['name']=='scalar_pressure_share_normal_withdrawal'
                scalar=np.asarray(old['previous_frozen'][0]['matrix']);assert np.array_equal(np.asarray(old['E'])[-1:],scalar)
                assert np.array_equal(np.asarray(old['f'])[-1:],old['previous_frozen'][0]['attained'])
                p=dict(old,E=old['E'][:-1],f=old['f'][:-1],previous_frozen=[])
                boundary_ok=np.array_equal(p['lower'],old['lower']) and np.array_equal(p['upper'],old['upper'])
            else:
                assert first['layer_name']=='normal_inward_correction_slack'
                # No L1 equality is a frozen optimum-eta coordinate.
                assert np.all(np.asarray(first['E'],float).reshape((-1,n))[:,21:]==0.)
                G=np.vstack((np.asarray(old['G'],float).reshape((-1,n)),np.asarray(first['G'],float).reshape((-1,n))))
                p=dict(old,G=G.tolist(),h=[*old['h'],*first['h']],E=first['E'],f=first['f'],previous_frozen=[],lower=first['lower'],upper=first['upper'])
                boundary_ok=np.array_equal(p['lower'][:21],old['lower'][:21]) and np.array_equal(p['upper'][:21],old['upper'][:21])
                assert np.array_equal(np.asarray(p['G'])[:len(old['G'])],old['G']) and p['h'][:len(old['h'])]==old['h']
                assert np.array_equal(p['lower'][21:],first['lower'][21:]) and np.array_equal(p['upper'][21:],first['upper'][21:])
            assert boundary_ok
            x=np.asarray(r['segment_point'],float);x0=np.asarray(old['initial_x'],float);A=np.asarray(old['A'],float);b=np.asarray(old['b'],float)
            a=primal(p,x);a0=primal(p,x0);v=loss(A,b,x);v0=loss(A,b,x0);point=r['segment_original_hard_primal' if 'priority' in name else 'segment_primal']
            hashmatch=phash(canonical(p))==point['original_problem_sha256']
            # The persisted linear objective c is the LP problem identity.
            # Independently fsum-evaluate its LS gradient; do not require two
            # legal summation/layout orders to have identical low bits.
            g=2*rows(A.T,rows(A,x0)-b);stored_g=np.asarray(r['direction_LP_KKT']['gradient'],float)
            eps=np.finfo(float).eps;ops=4*n+4*len(b)+32;gamma=ops*eps/(1-ops*eps)
            # Conservative straight-line dot-product/subtraction operation
            # bound for this finite normal-range corpus (no sigma/tol floor).
            g_bound=np.nextafter(gamma*2*abs(A).T@(abs(A)@abs(x0)+abs(b)),np.inf)
            gradient_ok=bool(np.all(abs(g-stored_g)<=g_bound))
            linear_hashmatch=phash(canonical(p,linear_gradient=stored_g))==r['direction_LP_KKT']['original_problem_sha256']
            error=abs(v-r['counterfactual_motion_loss']);initial_error=abs(v0-r['original_motion_loss']);allowance=64*np.finfo(float).eps*max(1.,abs(v),abs(v0))
            normal=None
            if 'tradeoff' in name:
                A1=np.asarray(first['A'],float);b1=np.asarray(first['b'],float);na=loss(A1,b1,x0);nb=loss(A1,b1,x)
                normal=dict(old=na,new=nb,delta=nb-na,reported_error=max(abs(na-r['normal_correction_loss_original']),abs(nb-r['normal_correction_loss_counterfactual'])))
                assert normal['reported_error']<=64*np.finfo(float).eps*max(1.,na,nb)
            cert=r['direction_LP_KKT'];reported_kkt=cert['kkt_passed'] is True and cert['stationarity_inf']<=cert['gradient_tolerance'] and cert['complementarity_inf']<=cert['complementarity_tolerance'] and cert['primal']['passed']
            good=all((a['passed'],a0['passed'],hashmatch,linear_hashmatch,gradient_ok,error<=allowance,initial_error<=allowance,reported_kkt))
            allpass &= good
            outputs.append(dict(experiment=name,context_key=s['context_key'],source=s['source'],passed=good,
                original21_bounds_exact=True,all_old_G_h_retained=True,stored_problem_hash_matches_independent_reconstruction=hashmatch,
                stored_LP_problem_hash_matches=linear_hashmatch,independent_primal=a,independent_anchor_primal=a0,
                LP_gradient_within_independent_arithmetic_bound=gradient_ok,LP_gradient_recompute_inf=float(max(abs(g-stored_g))),
                LP_gradient_componentwise_bound=g_bound.tolist(),arithmetic_bound_operation_count=ops,
                original_loss=v0,counterfactual_loss=v,motion_loss_change=v-v0,objective_recompute_error=error,
                initial_loss_recompute_error=initial_error,normal_task_loss=normal,reported_LP_KKT_consistent=reported_kkt,
                LP_duals_independently_recertified=False,segment_is_global_motion_QP_optimum=False))
    summary={}
    for label in ('v21_motion_priority_counterfactual_v1','v21_normal_motion_tradeoff_counterfactual_v1'):
        rr=[r for r in outputs if r['experiment']==label]
        summary[label]=dict(n=len(rr),passed=sum(r['passed'] for r in rr),old_loss_range=[min(r['original_loss'] for r in rr),max(r['original_loss'] for r in rr)],
            new_loss_range=[min(r['counterfactual_loss'] for r in rr),max(r['counterfactual_loss'] for r in rr)],
            maximum_absolute_loss_change=max(abs(r['motion_loss_change']) for r in rr),
            normal_loss_increased=sum(r['normal_task_loss'] is not None and r['normal_task_loss']['delta']>0 for r in rr))
    result=dict(schema='v21_motion_tradeoff_cross_independent_v2',passed=allpass,segments_checked=32,summary=summary,rows=outputs,input_sha256=inputs,
        reader_sha256=sha(__file__),physics_steps=0,optimizer_calls=0,production_changes=0,
        scope='Stored16 first-failed-L3 states, not all1377 states; independent domain/problem-hash/primal/raw-LS recomputation, static KKT checker review.',
        previous_failed_attempt='review/v21_motion_tradeoff_cross_independent_v1.json',
        previous_failed_attempt_reason='Extra independent floating gradient byte-hash expectation; original segment problem hashes, primals and objectives passed. Original artifact/script retained.',
        limitations=['LP endpoint/dual multipliers are not persisted, so no independent LP dual recertification was done. Existing result KKT values and original verifier source were reviewed.',
          'LP KKT is for a linearized direction objective, not KKT/global optimality of the quadratic segment point.',
          'eta is a geometry proxy slack, not contact pressure. No true pressure hard bound existed in this hierarchy; unchanged offline force guard is not a pressure safety proof.'])
    out=ROOT/'review/v21_motion_tradeoff_cross_independent_v2.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=allpass,summary=summary),ensure_ascii=False))
    return 0 if allpass else 1
if __name__=='__main__':raise SystemExit(main())

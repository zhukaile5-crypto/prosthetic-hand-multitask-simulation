"""Fresh original-objective/KKT audit. No author checker, model or physics import."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
KERNEL=ROOT/'v14_coord/intent_qp.py'
FIXTURE=ROOT/'review/v14_archived_fusion225_queries_v1.json'
OUT=ROOT/'review/v14_qp_independent_v1.json'
PINS={'v14_coord/intent_qp.py':'1f56174b10af062dea161513874607e6971788d1a9589b2e2c414a5694eeeebc',
      'review/v14_archived_fusion225_queries_v1.json':'d00dc0bd81f79c694954d51d5e34e216f691a68caf820e6bdbd65cbd7ebd5609'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(x):return np.asarray(x,float)
def norm_inf(x):return float(np.max(abs(x))) if len(x) else 0.

def cone_witness(z,g,digits,caps,budget):
    """Independent dual feasibility intervals + bisection, not production helper."""
    groups=[np.flatnonzero(digits==k) for k in range(5)];sums=np.asarray([z[i].sum() for i in groups]);eps=np.finfo(float).eps
    active=np.abs(caps-sums)<=8.*eps*np.maximum(abs(caps),abs(sums))
    active_budget=abs(budget-z.sum())<=8.*eps*max(abs(budget),abs(float(z.sum())))
    def feasible(rho,return_witness=False):
        eta_lo=0.;eta_hi=np.inf;intervals=[]
        for k,ids in enumerate(groups):
            if not len(ids):intervals.append((0.,np.inf));continue
            low=float(np.max(-g[ids]))-rho;positive=ids[z[ids]>0.]
            high=float(np.min(-g[positive]))+rho if len(positive) else np.inf
            if low>high:return None
            intervals.append((low,high));eta_hi=min(eta_hi,high)
            if not active[k]:eta_lo=max(eta_lo,low)
        if not active_budget:
            if eta_lo>0. or eta_hi<0.:return None
            eta=0.
        else:
            if eta_lo>eta_hi:return None
            eta=eta_lo
        if not return_witness:return True
        lam=np.zeros(5)
        for k,ids in enumerate(groups):
            if len(ids) and active[k]:lam[k]=max(eta,intervals[k][0])-eta
        return eta,lam
    if feasible(0.) is not None:rho=0.
    else:
        low=0.;high=max(1.,norm_inf(g))
        while feasible(high) is None:high*=2.
        for _ in range(70):
            mid=.5*(low+high)
            if feasible(mid) is None:low=mid
            else:high=mid
        rho=high
    eta,lam=feasible(rho,True);adjusted=g+eta+lam[digits]
    stationarity=norm_inf(np.where(z>0.,adjusted,np.minimum(adjusted,0.)))
    complementarity=max(float(np.max(abs(lam*(caps-sums)))),abs(float(eta*(budget-z.sum()))))
    return dict(stationarity=stationarity,complementarity=complementarity,eta=float(eta),lambda_group=lam.tolist(),dual_violation=max(0.,-float(eta),-float(lam.min())),rho_search=float(rho))

def fresh(q,result):
    p0=arr(q['p0_21']);A=arr(q['A21x25']);J=arr(q['Jpx25']);y=arr(q['y_p']);target=arr(q['z_intent_p']);d=arr(result.d21);z=arr(result.zP)
    sr=arr(q['scales']['residual25']);sf=arr(q['scales']['intentP']);sd=arr(q['scales']['delta21']);sl=float(q['scales']['load']);c=arr(q['support_coeff_p']);load=float(q['load_proxy_N'])
    lo=arr(q['effort_lower21'])-p0;hi=arr(q['effort_upper21'])-p0;digits=np.asarray(q['patch_digit_p'],int);caps=arr(q['caps5']);budget=float(q['total_budget'])
    residual=A.T@d-J.T@(z-y);short=max(0.,load-float(c@z))
    objective=float((sr*residual)@(sr*residual)+(sf*(z-target))@(sf*(z-target))+(sd*d)@(sd*d)+(sl*short)**2)
    weighted=sr**2*residual;gd=2.*(A@weighted+sd**2*d);gz=-2.*J@weighted+2.*sf**2*(z-target)-2.*sl**2*short*c
    box=np.where(lo==hi,0.,np.where(d<=lo,np.minimum(gd,0.),np.where(d>=hi,np.maximum(gd,0.),gd)))
    cone=cone_witness(z,gz,digits,caps,budget)
    primal=max(0.,float(np.max(lo-d)),float(np.max(d-hi)),float(-z.min()) if len(z) else 0.,float(z.sum()-budget),*[float(z[digits==k].sum()-caps[k]) for k in range(5)])
    atol=float(q.get('atol',1e-8));rtol=float(q.get('rtol',1e-7));td=atol+rtol*max(1.,norm_inf(d));tz=atol+rtol*max(1.,norm_inf(z));tc=atol+rtol
    auxiliary=float(result.diagnostics['auxiliary_shortfall']);u=np.r_[sd*d,sf*z,sl*auxiliary];s=np.r_[sd,sf,sl]
    M=np.column_stack((A.T,-J.T,np.zeros(25)));R=np.vstack((sr[:,None]*M/s[None,:],np.eye(len(s))));rhs=np.r_[-sr*(J.T@y),np.zeros(21),sf*target,0.]
    lifted=float(np.square(R@u-rhs).sum());scale=max(1.,abs(objective),abs(lifted));roundoff=1024*np.finfo(float).eps*scale
    return dict(objective=objective,residual=residual,shortfall=short,box_KKT=norm_inf(box),normal_cone=cone,primal=primal,td=td,tz=tz,tc=tc,atol=atol,
                auxiliary_shortfall=auxiliary,lifted_objective=lifted,lifted_vs_original=abs(lifted-objective),roundoff=roundoff,
                p0_preserved=np.array_equal(arr(result.diagnostics['fixed_original_p0_21']),p0),target_preserved=np.array_equal(arr(result.diagnostics['original_z_intent_p']),target))

def main():
    assert all(sha(ROOT/n)==h for n,h in PINS.items())
    before={n:(ROOT/n).read_bytes() for n in PINS};old_helper=ROOT/'v13_joint/intent_qp.py';helper_sha=sha(old_helper)
    sys.path.insert(0,str(ROOT));from v14_coord.intent_qp import solve_intent_qp
    data=json.loads(FIXTURE.read_text(encoding='utf-8'));queries=data['queries'];assert len(queries)==225
    failures=[];check_count=0;rows=[];oracle_rows=[]
    def check(name,ok,detail=None):
        nonlocal check_count
        check_count+=1
        if not ok:failures.append(dict(name=name,detail=detail))
    def validate(q,r,label,require=True):
        check(label+'/available',r.available or not require,dict(status=r.status,reason=r.reason))
        if not r.available:
            if r.d21 is not None and r.zP is not None:
                # A failure candidate is not mislabelled converged; do not certify it.
                check(label+'/unavailable_not_converged',r.status!='converged')
            return dict(status=r.status,reason=r.reason,iterations=r.iterations)
        f=fresh(q,r);n=f['normal_cone']
        check(label+'/finite',np.isfinite(np.r_[r.d21,r.zP,r.residual25,r.objective]).all())
        check(label+'/original_primal',f['primal']<=f['atol'],f['primal'])
        check(label+'/original_box_KKT',f['box_KKT']<=f['td'],dict(value=f['box_KKT'],tol=f['td']))
        check(label+'/independent_laminar_KKT',n['stationarity']<=f['tz'],dict(value=n['stationarity'],tol=f['tz']))
        check(label+'/independent_complementarity',n['complementarity']<=f['tc'],n)
        check(label+'/nonnegative_duals',n['dual_violation']<=f['atol'],n)
        check(label+'/original_objective_bytes',np.float64(f['objective']).tobytes()==np.float64(r.objective).tobytes())
        check(label+'/original_residual_bytes',f['residual'].tobytes()==r.residual25.tobytes())
        check(label+'/original_shortfall',f['shortfall']==r.load_shortfall_N)
        check(label+'/unmodified_target_p0',f['p0_preserved'] and f['target_preserved'])
        check(label+'/same_tolerance_budget',r.diagnostics['numerical_tolerance_d21']==f['td'] and r.diagnostics['numerical_tolerance_zP']==f['tz'] and r.diagnostics['computational_budget']['max_iterations']==q.get('max_iterations',400) and r.iterations<=q.get('max_iterations',400))
        check(label+'/auxiliary_hinge_equivalence',f['auxiliary_shortfall']>=f['shortfall']-f['roundoff'] and abs(f['auxiliary_shortfall']-f['shortfall'])<=f['roundoff'] and f['lifted_vs_original']<=f['roundoff'],dict(aux=f['auxiliary_shortfall'],hinge=f['shortfall'],gap=f['lifted_vs_original'],roundoff=f['roundoff']))
        return dict(status=r.status,iterations=r.iterations,objective=f['objective'],primal=f['primal'],box_KKT=f['box_KKT'],z_KKT=n['stationarity'],complementarity=n['complementarity'],td=f['td'],tz=f['tz'],lifted_gap=f['lifted_vs_original'],auxiliary_gap=f['auxiliary_shortfall']-f['shortfall'])
    for query in queries:
        q=query['kwargs'];input_bytes=json.dumps(q,sort_keys=True,allow_nan=False);r=solve_intent_qp(**q);row=validate(q,r,f'query{query["decision_id"]}')
        check(f'query{query["decision_id"]}/inputs_unmodified',json.dumps(q,sort_keys=True,allow_nan=False)==input_bytes)
        row.update(decision_id=query['decision_id'],time_s=query['time_s']);rows.append(row)
    def fixture(P=1):
        return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),Jpx25=np.zeros((P,25)),y_p=np.zeros(P),z_intent_p=np.zeros(P),effort_lower21=np.full(21,-10.),effort_upper21=np.full(21,10.),patch_digit_p=np.zeros(P,int),caps5=np.full(5,10.),total_budget=10.,support_coeff_p=np.zeros(P),load_proxy_N=0.,scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.),max_iterations=400,atol=1e-8,rtol=1e-7)
    def oracle(name,q,condition):
        r=solve_intent_qp(**q);record=validate(q,r,name);check(name+'/closed_form',r.available and bool(condition(r)),dict(d=None if r.d21 is None else r.d21.tolist(),z=None if r.zP is None else r.zP.tolist(),status=r.status));record['name']=name;oracle_rows.append(record)
    q=fixture();q['A21x25'][0,0]=1.;q['Jpx25'][0,1]=1.;q['z_intent_p'][0]=1.;q['scales']['residual25'][0]=2e8
    oracle('hugeL_original_false_convergence_counterexample',q,lambda r:abs(r.zP[0]-.5)<1e-12 and abs(r.d21[0])<1e-15)
    q=fixture();q['z_intent_p'][0]=5e-13;q['support_coeff_p'][0]=1.;q['load_proxy_N']=1.;q['caps5'][0]=1e-12;q['total_budget']=1e-12
    oracle('tiny_strict_domain_no_atol_activity_swallow',q,lambda r:r.zP[0]==1e-12)
    q=fixture(0);q['load_proxy_N']=2.;q['scales']['load']=3.;q['effort_lower21'][3]=q['effort_upper21'][3]=.2
    oracle('no_patches_fixed_effort_positive_hinge',q,lambda r:len(r.zP)==0 and r.d21[3]==.2 and abs(r.objective-36.04)<1e-13)
    for name,target,load,expect in (('active_hinge',0.,1.,.5),('inactive_hinge',2.,1.,2.),('hinge_boundary',1.,1.,1.)):
        q=fixture();q['z_intent_p'][0]=target;q['support_coeff_p'][0]=1.;q['load_proxy_N']=load
        oracle(name,q,lambda r,e=expect:abs(r.zP[0]-e)<1e-12)
    q=fixture();q['support_coeff_p'][0]=-1.;q['load_proxy_N']=1.
    oracle('signed_support_lower_boundary',q,lambda r:r.zP[0]==0. and abs(r.objective-1.)<1e-14)
    q=fixture(2);q['z_intent_p'][:]=2.;q['caps5'][0]=q['total_budget']=1.;q['effort_lower21'][:]=q['effort_upper21'][:]=0.
    oracle('redundant_group_and_global_caps_fixed_d',q,lambda r:np.max(abs(r.zP-.5))<1e-14 and abs(r.objective-4.5)<1e-13)
    q=fixture(3);q['z_intent_p'][:]=[1.,2.,3.];q['caps5'][:]=0.;q['total_budget']=0.;q['support_coeff_p'][:]=[1.,-1.,2.];q['load_proxy_N']=2.
    oracle('zero_caps_and_budget_singleton',q,lambda r:np.array_equal(r.zP,np.zeros(3)) and abs(r.objective-18.)<1e-13)
    q=fixture();q['z_intent_p'][0]=3.;q['caps5'][0]=q['total_budget']=1.
    oracle('original_intent_not_preprojected',q,lambda r:r.zP[0]==1. and abs(r.objective-4.)<1e-13)
    # Pick an actually multi-iteration archived query solely to test the declared1-step failure boundary.
    longest=max(range(len(rows)),key=lambda k:rows[k]['iterations']);q=dict(queries[longest]['kwargs']);q['max_iterations']=1;r=solve_intent_qp(**q);record=validate(q,r,'one_step_budget',require=False)
    check('one_step_budget/does_not_claim_unverified_convergence',not r.available and r.status in ('nonconvergence','numerical_stagnation'));oracle_rows.append(dict(name='one_step_budget',archived_decision_id=queries[longest]['decision_id'],**record))
    q=fixture();q['load_proxy_N']=None;r=solve_intent_qp(**q);check('unknown_load_not_zero',not r.available and r.status=='unknown' and r.d21 is None)
    q=fixture();q['Jpx25']=None;r=solve_intent_qp(**q);check('unknown_J_not_zero',not r.available and r.status=='unknown' and r.d21 is None)
    check('no_mujoco_or_robot_module_import','mujoco' not in sys.modules and 'hand_plant' not in sys.modules)
    stable=all((ROOT/n).read_bytes()==v for n,v in before.items()) and sha(old_helper)==helper_sha;check('kernel_fixture_helpers_unchanged',stable)
    report=dict(schema='v14_qp_independent_v1',role='independent_original_objective_and_dual_witness_audit',passed=not failures,checks=check_count,failures=failures,archived_queries=len(rows),archived_converged=sum(r['status']=='converged' for r in rows),rows=rows,oracles=oracle_rows,
        source_sha256=PINS,old_validation_helper_sha256=helper_sha,checker_sha256=sha(Path(__file__)),source_stable=stable,physics_steps=0,models_constructed=0,
        independent_validation='Recomputes original residual/objective/gradient and strict box normal signs. z cone uses fresh group dual feasible intervals plus70-step bisection and actual stationarity/complementarity witness, never calls imported production KKT helpers.',
        auxiliary_validation='Rebuilds dimensionless augmented least-squares from original kwargs and exported auxiliary t; checks t=max(0,L-cz), constraints and lifted/original objective equality to local floating roundoff.',
        limitations=['225 cold solves are one seen nominal/fusion/617901 holding public trajectory, not225 independent physics contexts or new closed-loop trials.',
          'Fixture exact reconstruction/provenance is bound by supplied SHA; this reviewer does not rebuild robot observations, geometry, native scales or rerun the original trajectory.',
          'Original-unit atol/rtol and400 budget are unchanged; no global floating optimality, pressure/force safety or real-time certificate.',
          'No V14 runtime or native effort realizer integration is tested here.'])
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),passed=report['passed'],checks=check_count,archived_converged=report['archived_converged'],oracles=len(oracle_rows),failures=failures[:6]),ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()

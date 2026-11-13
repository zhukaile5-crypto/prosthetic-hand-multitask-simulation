"""Independent NumPy-only convex/projection/clock-free QP audit; no simulator.

Small active-set QPs and bisection projection use different algorithms from
the production event sweep/FISTA. High-scale and narrow-domain witnesses are
retained independently of nominal cases. Never writes the audited kernel.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SHA='b115f01828a3d195e9846337d6c3435d797db0efde96e0ef364f17ab869dba86'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def objective(x,k):
    d,z=x[:21],x[21:]
    residual=k['A21x25'].T@d-k['Jpx25'].T@(z-k['y_p'])
    short=max(0.,k['load_proxy_N']-k['support_coeff_p']@z)
    return float(np.dot(k['scales']['residual25']*residual,k['scales']['residual25']*residual)+
                 np.sum((k['scales']['intentP']*(z-k['z_intent_p']))**2)+
                 np.sum((k['scales']['delta21']*d)**2)+(k['scales']['load']*short)**2)

def gradient(x,k):
    d,z=x[:21],x[21:];r=k['A21x25'].T@d-k['Jpx25'].T@(z-k['y_p'])
    wr=k['scales']['residual25']**2*r
    short=max(0.,k['load_proxy_N']-k['support_coeff_p']@z)
    return np.r_[2*(k['A21x25']@wr+k['scales']['delta21']**2*d),
        -2*k['Jpx25']@wr+2*k['scales']['intentP']**2*(z-k['z_intent_p'])-2*k['scales']['load']**2*short*k['support_coeff_p']]

def projection_bisection(v,digits,caps,budget):
    """Independent dual root bisection, not production water-filling events."""
    v=np.asarray(v,float);digits=np.asarray(digits,int)
    if len(v)==0 or budget==0:return np.zeros_like(v)
    groups=[np.flatnonzero(digits==i) for i in range(5)];theta=np.zeros(5)
    for i,idx in enumerate(groups):
        if not len(idx):continue
        if caps[i]==0:theta[i]=max(0.,float(v[idx].max()));continue
        if np.maximum(v[idx],0).sum()>caps[i]:
            lo=0.;hi=max(0.,float(v[idx].max()))
            for _ in range(96):
                mid=(lo+hi)/2
                if np.maximum(v[idx]-mid,0).sum()>caps[i]:lo=mid
                else:hi=mid
            theta[i]=(lo+hi)/2
    def at(lam):
        z=np.zeros_like(v)
        for i,idx in enumerate(groups):
            if len(idx) and caps[i]>0:z[idx]=np.maximum(v[idx]-max(lam,theta[i]),0.)
        return z
    if at(0.).sum()<=budget:return at(0.)
    lo=0.;hi=max(0.,float(v.max()))
    for _ in range(96):
        mid=(lo+hi)/2
        if at(mid).sum()>budget:lo=mid
        else:hi=mid
    return at((lo+hi)/2)

def active_qp(H,g,G,h):
    """Exhaustive KKT subsets for small strictly convex quadratic programs."""
    n=len(g);best=None
    for count in range(min(n,len(h))+1):
        for subset in itertools.combinations(range(len(h)),count):
            Q=G[list(subset)] if count else np.empty((0,n));b=h[list(subset)] if count else np.empty(0)
            matrix=np.block([[H,Q.T],[Q,np.zeros((count,count))]])
            try:solution=np.linalg.solve(matrix,np.r_[-g,b])
            except np.linalg.LinAlgError:continue
            x=solution[:n];dual=solution[n:]
            if not np.isfinite(x).all() or np.max(G@x-h,initial=0)>2e-8 or np.min(dual,initial=0)<-2e-8:continue
            if np.max(abs(H@x+g+Q.T@dual),initial=0)>2e-8:continue
            value=float(.5*x@H@x+g@x)
            if best is None or value<best[0]:best=(value,x.copy())
    return None if best is None else best[1]

def small_oracle(k,active):
    P=len(k['y_p']);K=len(active);n=K+P
    M=np.column_stack((k['A21x25'][active].T,-k['Jpx25'].T));b=k['Jpx25'].T@k['y_p']
    sr=k['scales']['residual25'];sf=k['scales']['intentP'];sd=k['scales']['delta21'][active];sl=k['scales']['load']
    diagonal=np.r_[sd**2,sf**2];q=np.r_[np.zeros(K),k['support_coeff_p']]
    H0=2*(M.T@(sr[:,None]**2*M)+np.diag(diagonal))
    g0=2*(M.T@(sr**2*b)-np.r_[np.zeros(K),sf**2*k['z_intent_p']])
    G=[];h=[]
    for i,axis in enumerate(active):
        row=np.zeros(n);row[i]=1.;G.extend((row,-row));h.extend((k['effort_upper21'][axis]-k['p0_21'][axis],k['p0_21'][axis]-k['effort_lower21'][axis]))
    for j in range(P):
        row=np.zeros(n);row[K+j]=-1.;G.append(row);h.append(0.)
    for digit in range(5):
        ids=np.flatnonzero(np.asarray(k['patch_digit_p'])==digit)
        if len(ids):
            row=np.zeros(n);row[K+ids]=1.;G.append(row);h.append(k['caps5'][digit])
    row=np.r_[np.zeros(K),np.ones(P)];G.append(row);h.append(k['total_budget'])
    best=None
    for hinge in (False,True):
        H=H0+2*sl**2*np.outer(q,q) if hinge else H0
        g=g0-2*sl**2*k['load_proxy_N']*q if hinge else g0
        region=q if hinge else -q;bound=k['load_proxy_N'] if hinge else -k['load_proxy_N']
        if np.any(region):GG=np.vstack(G+[region]);hh=np.asarray(h+[bound])
        elif bound<0:continue
        else:GG=np.vstack(G);hh=np.asarray(h)
        compact=active_qp(H,g,GG,hh)
        if compact is None:continue
        x=np.r_[np.zeros(21),compact[K:]];x[np.asarray(active,int)]=compact[:K]
        value=objective(x,k)
        if best is None or value<best[0]:best=(value,x)
    return best

def fixture(P=1):
    return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),Jpx25=np.zeros((P,25)),
        y_p=np.zeros(P),z_intent_p=np.ones(P),effort_lower21=-np.ones(21)*10,effort_upper21=np.ones(21)*10,
        patch_digit_p=np.arange(P,dtype=int)%5,caps5=np.ones(5)*10,total_budget=50.,support_coeff_p=np.zeros(P),load_proxy_N=0.,
        scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',default=str(ROOT/'review/v13_joint_intent_qp_independent_b115_v1/kernel_b115.py'))
    parser.add_argument('--expected-sha',default=DEFAULT_SHA);parser.add_argument('--out',required=True);args=parser.parse_args()
    kernel=Path(args.kernel);out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh independent evidence required')
    if sha(kernel)!=args.expected_sha:raise ValueError('Exact kernel SHA differs from requested audit')
    spec=importlib.util.spec_from_file_location('independent_intent_'+args.expected_sha[:12],kernel)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    checks=[];details={};rng=np.random.default_rng(872341)
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=clean(detail)))
    check('no simulator import','mujoco' not in sys.modules)
    projection_errors=[]
    for index in range(96):
        P=(0,1,2,5,32,128)[index%6];v=rng.normal(1.,3.,P);digits=rng.integers(0,5,P)
        caps=rng.uniform(0.,10.,5);caps[index%5]=0. if index%4==0 else caps[index%5];budget=float(rng.uniform(0.,caps.sum()+3.))
        if index%9==0:budget=0.
        actual=module.project_normal_intents(v,digits,caps,budget);oracle=projection_bisection(v,digits,caps,budget)
        error=float(np.max(abs(actual-oracle),initial=0.));projection_errors.append(error)
        feasible=bool(np.all(actual>=0) and actual.sum()<=budget+2e-11 and all(actual[digits==i].sum()<=caps[i]+2e-11 for i in range(5)))
        check('independent_dual_projection_'+str(index),error<2e-10 and feasible,dict(P=P,max_abs_error=error))
        check('projection_idempotent_'+str(index),np.max(abs(module.project_normal_intents(actual,digits,caps,budget)-actual),initial=0)<2e-10)
    # Capture production nested evaluate returns; independent objective-only
    # central differences check its actual gradient rather than a copied test gradient.
    captured=[]
    def trace(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='evaluate' and Path(frame.f_code.co_filename)==kernel:
            captured.append((frame.f_locals['x'].copy(),float(arg[0]),np.asarray(arg[1]).copy()))
        return trace
    for index in range(6):
        k=fixture(3);k['A21x25']=rng.normal(0,.15,(21,25));k['Jpx25']=rng.normal(0,.2,(3,25));k['y_p']=rng.uniform(0,2,3)
        k['scales']=dict(residual25=rng.uniform(.5,2,25),intentP=rng.uniform(.5,2,3),delta21=rng.uniform(.5,2,21),load=1.2)
        k['support_coeff_p']=rng.uniform(-.4,1.,3);k['load_proxy_N']=4. if index%2==0 else 0.
        captured.clear();old_trace=sys.gettrace();sys.settrace(trace)
        try:r=module.solve_intent_qp(**k,max_iterations=1)
        finally:sys.settrace(old_trace)
        gradient_error=0.;objective_error=0.
        for x,value,g in captured:
            fd=np.empty_like(x)
            for j in range(len(x)):
                h=1e-6*max(1.,abs(x[j]));plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
                fd[j]=(objective(plus,k)-objective(minus,k))/(2*h)
            gradient_error=max(gradient_error,float(np.max(abs(fd-g))));objective_error=max(objective_error,abs(objective(x,k)-value))
        check('production_gradient_central_difference_'+str(index),len(captured)>0 and gradient_error<2e-6,gradient_error)
        check('production_objective_exact_definition_'+str(index),objective_error<1e-10,objective_error)
        M=np.c_[k['A21x25'].T,-k['Jpx25'].T];T=k['scales']['residual25'][:,None]*M;diag=np.r_[k['scales']['delta21']**2,k['scales']['intentP']**2]
        q=np.r_[np.zeros(21),k['support_coeff_p']];L=r.diagnostics['lipschitz_upper_bound']
        hess=[2*(T.T@T+np.diag(diag)+active*k['scales']['load']**2*np.outer(q,q)) for active in (0,1)]
        largest=max(np.linalg.eigvalsh(H)[-1] for H in hess);smallest=min(np.linalg.eigvalsh(H)[0] for H in hess)
        check('convex_Hessian_and_global_L_bound_'+str(index),smallest>0 and L+1e-10*max(1,L)>=largest,dict(L=L,max_eigenvalue=largest,min_eigenvalue=smallest))
    qp_cases=[]
    for index in range(16):
        k=fixture(2);active=[0,1];k['A21x25'][active]=rng.normal(0,.5,(2,25));k['Jpx25']=rng.normal(0,.3,(2,25))
        k['p0_21']=rng.uniform(-.2,.2,21);k['effort_lower21']=k['p0_21'].copy();k['effort_upper21']=k['p0_21'].copy()
        k['effort_lower21'][active]-=rng.uniform(.2,1.,2);k['effort_upper21'][active]+=rng.uniform(.2,1.,2)
        k['y_p']=rng.uniform(0,2,2);k['z_intent_p']=rng.uniform(.1,3,2);k['patch_digit_p']=np.array([0,index%2])
        k['caps5']=rng.uniform(.1,2.,5);k['total_budget']=float(rng.uniform(.1,2.5));k['support_coeff_p']=rng.uniform(-.5,1.,2);k['load_proxy_N']=float(rng.uniform(0,2.))
        k['scales']=dict(residual25=rng.uniform(.5,1.5,25),intentP=rng.uniform(.5,2,2),delta21=rng.uniform(.5,1.5,21),load=float(rng.uniform(.5,2)))
        oracle=small_oracle(k,active);r=module.solve_intent_qp(**k,max_iterations=4000,atol=1e-10,rtol=1e-9)
        x=np.r_[r.d21,r.zP] if r.d21 is not None else None
        error=None if oracle is None or x is None else float(np.max(abs(x-oracle[1])))
        gap=None if oracle is None or x is None else float(objective(x,k)-oracle[0])
        check('exhaustive_two_hinge_active_set_'+str(index),oracle is not None and r.available and error<2e-6 and abs(gap)<2e-8,dict(status=r.status,iterations=r.iterations,max_abs_solution_error=error,objective_gap=gap))
        check('unretargeted_result_residual_'+str(index),x is not None and np.max(abs(r.residual25-(k['A21x25'].T@r.d21-k['Jpx25'].T@(r.zP-k['y_p']))))<1e-12)
        qp_cases.append(dict(case=index,status=r.status,iterations=r.iterations,error=error,gap=gap))
    # Full declared dimensions, moderate scales; convergence is NOT required
    # beyond its declared budget. Any available candidate must meet independent conditions.
    full=[]
    for P in (0,1,5,32,128):
        k=fixture(P);k['A21x25']=rng.normal(0,.1,(21,25));k['Jpx25']=rng.normal(0,.1,(P,25));k['y_p']=rng.uniform(0,1,P)
        k['z_intent_p']=rng.uniform(0,2,P);k['caps5']=np.ones(5)*4;k['total_budget']=12.;k['support_coeff_p']=rng.uniform(-.2,1.,P);k['load_proxy_N']=2.
        r=module.solve_intent_qp(**k);x=np.r_[r.d21,r.zP] if r.d21 is not None else None
        feasible=x is not None and np.all(r.p21>=k['effort_lower21']-1e-8) and np.all(r.p21<=k['effort_upper21']+1e-8) and np.min(r.zP,initial=0)>=-1e-8 and r.zP.sum()<=12+1e-8 and all(r.zP[np.array(k['patch_digit_p'])==i].sum()<=4+1e-8 for i in range(5))
        check('full_dimension_feasible_and_honest_status_P'+str(P),feasible and (r.available or r.status=='nonconvergence'),dict(status=r.status,iterations=r.iterations))
        check('full_dimension_original_objective_P'+str(P),x is not None and abs(objective(x,k)-r.objective)<1e-10)
        full.append(dict(P=P,status=r.status,iterations=r.iterations,objective=r.objective))
    boundaries=[]
    mutations=[('missing_J','unknown',{'Jpx25':None}),('nan_y','unknown',{'y_p':np.array([np.nan])}),('missing_cap','unknown',{'caps5':None}),('missing_load','unknown',{'load_proxy_N':None}),
        ('negative_y','invalid_input',{'y_p':np.array([-1.])}),('float_digit','invalid_input',{'patch_digit_p':[0.]}),('bool_digit','invalid_input',{'patch_digit_p':[True]}),('outside_digit','invalid_input',{'patch_digit_p':[-1]}),
        ('missing_digit_has_no_fabrication','invalid_input',{'patch_digit_p':None}),('negative_cap','invalid_input',{'caps5':np.array([-1.,1,1,1,1])}),('negative_budget','invalid_input',{'total_budget':-1.}),('boolean_budget','invalid_input',{'total_budget':True}),('boolean_iteration','invalid_input',{'max_iterations':True})]
    for name,status,change in mutations:
        k=fixture();k.update(change);r=module.solve_intent_qp(**k)
        check('unknown_invalid_boundary_'+name,r.status==status and not r.available and r.d21 is None and r.zP is None,dict(status=r.status,reason=r.reason))
        boundaries.append(dict(name=name,status=r.status,reason=r.reason))
    k=fixture();k['A21x25'][0,0]=1.;k['Jpx25'][0,1]=1.;k['scales']['residual25'][0]=2e8
    bad=module.solve_intent_qp(**k)
    g_true=None if bad.zP is None else float(4*bad.zP[0]-2)
    scale_witness=dict(status=bad.status,available=bad.available,z=None if bad.zP is None else bad.zP.tolist(),
        reported_KKT=None if bad.zP is None else bad.diagnostics['projected_gradient_inf_zP'],true_interior_gradient=g_true,
        analytic_optimum_z=.5,analytic_optimum_objective=.5,objective=bad.objective,iterations=bad.iterations,
        L=bad.diagnostics.get('lipschitz_upper_bound'))
    check('no_false_convergence_under_allowed_finite_scale',not bad.available or (abs(bad.zP[0]-.5)<1e-6 and abs(g_true)<1e-6),scale_witness)
    # A separate stopping-scale limitation: narrow interval, all scales1.
    # PG can be small due to diameter even in exact arithmetic; do not confuse
    # it with an interior normal-cone stationarity certificate or a large cost gap.
    k=fixture();k['z_intent_p']=np.array([5e-13]);k['caps5'][0]=1e-12;k['total_budget']=1e-12;k['support_coeff_p']=np.ones(1);k['load_proxy_N']=1.
    narrow=module.solve_intent_qp(**k);z=float(narrow.zP[0]);gg=float(gradient(np.r_[narrow.d21,narrow.zP],k)[21]);unit=float(z-np.clip(z-gg,0.,1e-12))
    details['narrow_domain_stopping_limit']=dict(status=narrow.status,z=z,true_gradient=gg,unit_step_mapping=unit,
        declared_mapping=narrow.diagnostics['projected_gradient_inf_zP'],strict_interior=0<z<1e-12,
        objective_gap_to_upper=float(narrow.objective-objective(np.r_[narrow.d21,1e-12],k)),
        interpretation='Small PG mapping is not raw interior stationarity; objective/position gaps are tiny here. This is not the substantive0.5 cost-gap cancellation witness.')
    details.update(scale_cancellation_witness=scale_witness,projection_max_abs_error=max(projection_errors),small_active_set_cases=qp_cases,full_dimension_cases=full,unknown_invalid_cases=boundaries)
    check('no_simulator_import_after_all_math','mujoco' not in sys.modules)
    result=dict(schema='v13_joint_intent_qp_independent_v1',kernel_path=str(kernel),kernel_sha256=args.expected_sha,
        author_checker_sha256=sha(__file__),physics_steps=0,simulator_imported='mujoco' in sys.modules,
        passed=all(c['passed'] for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],details=details,
        scope='Pure convex/projection/numerical acceptance only; no native response, runtime, future-pressure, physical-safety or hard-real-time certificate')
    if sha(kernel)!=args.expected_sha:raise ValueError('Audited kernel snapshot changed')
    with out.open('x',encoding='utf-8') as stream:json.dump(clean(result),stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=len(result['failures']),out=str(out),sha256=sha(out),projection_max_abs_error=max(projection_errors),physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

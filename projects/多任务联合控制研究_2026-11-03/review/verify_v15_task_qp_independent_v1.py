"""Independent original-unit objective/gradient and small global QP oracle.

No author test/helper, robot module, SciPy or physics. The oracle enumerates
active halfspaces in ORIGINAL coordinates and BOTH hinge regions. It does not
reuse production projections, KKT checks, scaling or active-set solver.
"""
from pathlib import Path
from copy import deepcopy
from itertools import combinations
import hashlib,json,pickle,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
KERNEL=ROOT/'v15_support/task_qp.py';EXPECTED='6d48d44ad544af323a11a33a29d73ea5757c4cb35bd183a989b6a81433c817e4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(v):return np.asarray(v,float)
def norm(v):return float(np.max(abs(v),initial=0.))
def fixture(P=1,free=(0,)):
    p0=np.zeros(21);lo=p0.copy();hi=p0.copy()
    for i in free:lo[i]=-3.;hi[i]=3.
    return dict(p0_21=p0,A21x25=np.zeros((21,25)),Jpx25=np.zeros((P,25)),y_p=np.zeros(P),z_intent_p=np.zeros(P),
        effort_lower21=lo,effort_upper21=hi,patch_digit_p=np.arange(P,dtype=int)%5,caps5=np.full(5,3.),total_budget=4.,support_coeff_p=np.zeros(P),load_proxy_N=0.,
        scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.),max_iterations=400,atol=1e-8,rtol=1e-7)
def raw(q,x):
    P=len(q['y_p']);d=x[:21];z=x[21:];A=array(q['A21x25']);J=array(q['Jpx25']).reshape(P,25)
    sr=array(q['scales']['residual25']);sf=array(q['scales']['intentP']);sd=array(q['scales']['delta21']);sl=float(q['scales']['load']);c=array(q['support_coeff_p'])
    r=A.T@d-J.T@(z-array(q['y_p']));short=max(0.,float(q['load_proxy_N'])-float(c@z))
    f=float((sr*r)@(sr*r)+(sf*(z-array(q['z_intent_p'])))@(sf*(z-array(q['z_intent_p'])))+(sd*d)@(sd*d)+(sl*short)**2)
    gd=2*(A@(sr*sr*r)+sd*sd*d);gz=-2*J@(sr*sr*r)+2*sf*sf*(z-array(q['z_intent_p']))-2*sl*sl*short*c
    extras=q.get('extra_task_rows');E=np.empty((0,21+P)) if extras is None else array(extras['matrix']).reshape(-1,21+P)
    b=np.empty(0) if extras is None else array(extras['target']);e=E@x-b;g=np.r_[gd,gz]
    if len(E):f+=float(e@e);g+=2*E.T@e
    return f,g,r,short,e
def oracle(q):
    """Strictly-convex small reduction, brute force all<=n active faces."""
    P=len(q['y_p']);N=21+P;p0=array(q['p0_21']);lo=array(q['effort_lower21'])-p0;hi=array(q['effort_upper21'])-p0
    free=np.flatnonzero(lo!=hi);idx=np.r_[free,21+np.arange(P)].astype(int);fixed=np.zeros(N);fixed[:21]=lo;fixed[free]=0.;n=len(idx)
    A=array(q['A21x25']);J=array(q['Jpx25']).reshape(P,25);s=q['scales'];sr=array(s['residual25']);sd=array(s['delta21']);sf=array(s['intentP']);sl=float(s['load']);c=array(q['support_coeff_p']);load=float(q['load_proxy_N'])
    R=[sr[:,None]*np.c_[A.T,-J.T],np.c_[np.diag(sd),np.zeros((21,P))],np.c_[np.zeros((P,21)),np.diag(sf)]]
    rhs=[-sr*(J.T@array(q['y_p'])),np.zeros(21),sf*array(q['z_intent_p'])]
    e=q.get('extra_task_rows')
    if e is not None:
        E=array(e['matrix']).reshape(-1,N);b=array(e['target'])
        if len(E):R.append(E);rhs.append(b)
    baseR=np.vstack(R);baseb=np.r_[*rhs];G=[];h=[]
    for j,i in enumerate(free):
        row=np.zeros(n);row[j]=1.;G.extend((row.copy(),-row.copy()));h.extend((hi[i],-lo[i]))
    for j in range(P):row=np.zeros(n);row[len(free)+j]=-1.;G.append(row);h.append(0.)
    digits=np.asarray(q['patch_digit_p'],int)
    for digit in range(5):
        row=np.zeros(n);row[len(free):]=(digits==digit)
        if row.any():G.append(row);h.append(q['caps5'][digit])
    if P:row=np.zeros(n);row[len(free):]=1.;G.append(row);h.append(q['total_budget'])
    best=None;count=0
    for active_hinge in (False,True):
        RR=baseR.copy();bb=baseb.copy();GG=[v.copy() for v in G];hh=list(h)
        row=np.zeros(n);row[len(free):]=c if active_hinge else -c;bound=load if active_hinge else -load
        if not row.any():
            if bound<0:continue
        else:GG.append(row);hh.append(bound)
        if active_hinge:RR=np.vstack([RR,np.r_[np.zeros(21),sl*c]]);bb=np.r_[bb,sl*load]
        rhs_reduced=bb-RR@fixed;RR=RR[:,idx];H=RR.T@RR;bvec=RR.T@rhs_reduced
        GG=np.asarray(GG).reshape(-1,n);hh=np.asarray(hh)
        for k in range(min(n,len(GG))+1):
            for active in combinations(range(len(GG)),k):
                C=GG[list(active)] if k else np.empty((0,n));bound_active=hh[list(active)] if k else np.empty(0)
                K=np.block([[H,C.T],[C,np.zeros((k,k))]])
                try:answer=np.linalg.solve(K,np.r_[bvec,bound_active])
                except np.linalg.LinAlgError:continue
                count+=1;x=answer[:n];lam=answer[n:]
                if norm(np.maximum(GG@x-hh,0.))>1e-9 or np.min(lam,initial=0.)< -1e-9:continue
                full=fixed.copy();full[idx]=x;value=raw(q,full)[0]
                if best is None or value<best['objective']:
                    best=dict(objective=value,x=full,idx=idx,G=GG,h=hh,active=list(active),lambda_=lam,active_hinge=active_hinge)
    if best is None:raise AssertionError('Independent global oracle found no feasible active face')
    best['linear_solves']=count;return best
def expanded(q,r):
    P=len(q['y_p']);s=q['scales'];sr=array(s['residual25']);sd=array(s['delta21']);sf=array(s['intentP']);sl=float(s['load']);ss=np.r_[sd,sf,sl]
    A=array(q['A21x25']);J=array(q['Jpx25']).reshape(P,25);B=np.c_[A.T,-J.T,np.zeros(25)]
    R=np.vstack([sr[:,None]*B/ss,np.eye(22+P)]);rhs=np.r_[-sr*(J.T@array(q['y_p'])),np.zeros(21),sf*array(q['z_intent_p']),0.]
    e=q.get('extra_task_rows')
    if e is not None:
        E=array(e['matrix']).reshape(-1,21+P)
        if len(E):R=np.vstack([R,np.c_[E/ss[:-1],np.zeros(len(E))]]);rhs=np.r_[rhs,array(e['target'])]
    u=np.r_[sd*r.d21,sf*r.zP,sl*r.diagnostics['auxiliary_shortfall']]
    return float(np.square(R@u-rhs).sum())
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    assert sha(KERNEL)==EXPECTED
    from v15_support.task_qp import solve_intent_qp
    from v14_coord.intent_qp import solve_intent_qp as old
    cases=[];checks=[];failures=[];captured=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:failures.append(dict(name=name,detail=detail))
    def call(q):
        before=pickle.dumps(q,protocol=5);calls=[];previous=sys.gettrace()
        def trace(frame,event,arg):
            if event=='return' and frame.f_code.co_name=='evaluate' and Path(frame.f_code.co_filename).resolve()==KERNEL.resolve() and isinstance(arg,tuple):
                calls.append((array(frame.f_locals['x']).copy(),float(arg[0]),array(arg[1]).copy()))
            return trace
        sys.settrace(trace)
        try:r=solve_intent_qp(**q)
        finally:sys.settrace(previous)
        check('input_immutable/'+str(len(cases)),before==pickle.dumps(q,protocol=5))
        return r,calls
    def validate(name,q,closed=None):
        r,gradcalls=call(q);check(name+'/available',r.available,dict(status=r.status,reason=r.reason))
        if not r.available:return
        x=np.r_[r.d21,r.zP];f,g,res,short,e=raw(q,x);o=oracle(q);idx=o['idx'];C=o['G'][o['active']] if o['active'] else np.empty((0,len(idx)))
        adjusted=g[idx]+2*C.T@o['lambda_'];stationarity=norm(adjusted);slack=o['h']-o['G']@x[idx]
        effort_count=sum(i<21 for i in idx);sdirect=norm(adjusted[:effort_count]);zdirect=norm(adjusted[effort_count:])
        primal=norm(np.maximum(-slack,0));complementarity=norm(2*o['lambda_']*(o['h'][o['active']]-C@x[idx])) if o['active'] else 0.
        check(name+'/global_active_face_minimum',abs(f-o['objective'])<1e-10 and norm(x-o['x'])<1e-9,dict(actual=f,global_oracle=o['objective'],distance=norm(x-o['x'])))
        td=q['atol']+q['rtol']*max(1.,norm(x[:21]));tz=q['atol']+q['rtol']*max(1.,norm(x[21:]))
        check(name+'/independent_KKT',sdirect<=td and zdirect<=tz and primal<=q['atol'] and complementarity<=q['atol']+q['rtol'] and np.min(2*o['lambda_'],initial=0.)>=-q['atol'],dict(d_stationarity=sdirect,z_stationarity=zdirect,primal=primal,complementarity=complementarity,td=td,tz=tz))
        check(name+'/raw_objective_bytes',np.float64(f).tobytes()==np.float64(r.objective).tobytes())
        check(name+'/original_residual_bytes',res.tobytes()==r.residual25.tobytes())
        check(name+'/extra_residual_and_cost',np.array_equal(e,array(r.diagnostics['extra_task_residual'])) and float(e@e)==r.diagnostics['extra_task_objective'])
        lifted=expanded(q,r);check(name+'/scaled_LS_same_original_objective',abs(lifted-f)<=1024*np.finfo(float).eps*max(1.,abs(f),abs(lifted)),dict(raw=f,lifted=lifted))
        check(name+'/original_p0_and_intent_not_retargeted',np.array_equal(q['p0_21'],r.diagnostics['fixed_original_p0_21']) and np.array_equal(q['z_intent_p'],r.diagnostics['original_z_intent_p']))
        check(name+'/no_extra_hard_constraints_or_force_cert',not r.diagnostics['extra_hard_constraints_implemented'] and not r.diagnostics['physical_safety_certified'] and r.diagnostics['computational_budget']['max_iterations']==400)
        check(name+'/production_gradient_captured',bool(gradcalls))
        for j,(xx,actual_value,actual_g) in enumerate(gradcalls):
            fresh_value,fresh_g,*_=raw(q,xx)
            check(name+f'/actual_eval_gradient/{j}',np.array_equal(actual_g,fresh_g) and np.float64(actual_value).tobytes()==np.float64(fresh_value).tobytes())
        # Numerical derivative of independent full objective at a non-boundary
        # test point; compares actual nested evaluate gradient, not its diag.
        xx,_,gg=gradcalls[-1]
        for j in idx:
            eps=1e-6;up=xx.copy();down=xx.copy();up[j]+=eps;down[j]-=eps
            fd=(raw(q,up)[0]-raw(q,down)[0])/(2*eps)
            check(name+f'/central_difference/{j}',abs(fd-gg[j])<2e-7,dict(fd=fd,gradient=float(gg[j])))
        if closed is not None:check(name+'/closed_form',bool(closed(r)))
        cases.append(dict(name=name,status=r.status,iterations=r.iterations,objective=f,oracle_objective=o['objective'],oracle_linear_solves=o['linear_solves'],
            independent_stationarity=stationarity,primal=primal,complementarity=complementarity,d21=r.d21.tolist(),zP=r.zP.tolist(),extra_residual=e.tolist(),original_gradient=g.tolist()))
    try:
        q=fixture(0,(0,3));E=np.zeros((1,21));E[0,[0,3]]=[2.,-3.];q['extra_task_rows']=dict(matrix=E,target=np.array([4.]))
        validate('emptyP_signed_two_efforts_nonzero_goal',q,lambda r:abs(r.d21[0]-4/7)<1e-12 and abs(r.d21[3]+6/7)<1e-12)
        q=fixture(1,(0,));q['A21x25'][0,0]=1.;q['Jpx25'][0,0]=.7;q['y_p'][0]=.2;q['z_intent_p'][0]=.4
        E=np.zeros((2,22));E[:,[0,21]]=[[1.1,-2.],[-.5,1.]];q['extra_task_rows']=dict(matrix=E,target=np.array([.6,-.2]));validate('coupled_dz_signed_rows_and_targets',q)
        q=fixture(2,(0,2));q['A21x25'][0,0]=1.;q['A21x25'][2,1]=.5;q['Jpx25'][:,:2]=[[.6,-.2],[.3,.8]];q['y_p'][:]=[.3,.15];q['z_intent_p'][:]=[1.,.7]
        q['support_coeff_p'][:]=[1.1,-.25];q['load_proxy_N']=1.2;q['caps5'][:2]=[.65,.4];q['total_budget']=.75
        q['scales']['delta21'][[0,2]]=[.8,1.4];q['scales']['intentP'][:]=[1.2,.9];q['scales']['load']=1.7
        E=np.zeros((3,23));E[:,[0,2,21,22]]=[[1.2,-.4,.7,-.9],[-.5,.8,-1.,.3],[.2,-.7,.5,.5]]
        q['extra_task_rows']=dict(matrix=E,target=np.array([.4,-.2,.3]));validate('hinge_signed_support_digit_caps_budget',q)
        q=deepcopy(q);q['caps5'][:]=3.;q['total_budget']=4.;q['support_coeff_p'][:]=[1.,1.];q['load_proxy_N']=.1;validate('inactive_hinge_coupled_rows',q)
        q=fixture(2,());q['effort_lower21'][5]=q['effort_upper21'][5]=.2;q['z_intent_p'][:]=1.;q['patch_digit_p'][:]=0;q['caps5'][0]=q['total_budget']=1.
        E=np.zeros((1,23));E[0,[5,21,22]]=[1.,1.,-1.];q['extra_task_rows']=dict(matrix=E,target=np.array([.4]));validate('nonzero_fixed_effort_extra_offset_and_redundant_caps',q)
        q=fixture(0,(0,));q['p0_21'][0]=10.;q['effort_lower21'][0]=0.;q['effort_upper21'][0]=20.
        E=np.zeros((1,21));E[0,0]=2.;q['extra_task_rows']=dict(matrix=E,target=np.array([22.-2*q['p0_21'][0]]))
        validate('absolute_effort_task_requires_r_minus_Wp0',q,lambda r:abs(r.d21[0]-.8)<1e-12)
        q=fixture(0,(0,));q['effort_lower21'][0]=0.;q['effort_upper21'][0]=.2;E=np.zeros((1,21));E[0,0]=-1.;q['extra_task_rows']=dict(matrix=E,target=np.array([-2.]))
        validate('one_sided_saturation_retains_soft_task_gap',q,lambda r:r.d21[0]==.2 and abs(r.diagnostics['extra_task_residual'][0]-1.8)<1e-12)
        q=fixture(1,(0,));q['extra_task_rows']=dict(matrix=np.zeros((2,22)),target=np.array([3.,-4.]));validate('zero_coefficients_nonzero_target_constant_cost',q,lambda r:r.objective==25.)
        # No-row path must be the frozen V14 path, not a newly weighted task.
        q=fixture(2,(0,1));q['A21x25'][0,0]=.4;q['Jpx25'][:,:2]=[[.3,.1],[-.2,.6]];q['z_intent_p'][:]=[1.,.3]
        baseline=old(**q)
        for label,extra in (('none',None),('empty_matrix',dict(matrix=np.empty((0,23)),target=np.empty(0))),('json_empty',dict(matrix=[],target=[]))):
            qq=deepcopy(q);qq['extra_task_rows']=extra;r,captured=call(qq)
            check(label+'/all_original_numeric_fields_bit_exact',r.status==baseline.status and r.reason==baseline.reason and r.iterations==baseline.iterations and all(np.asarray(getattr(r,k)).tobytes()==np.asarray(getattr(baseline,k)).tobytes() for k in ('d21','zP','p21','residual25','objective','load_shortfall_N')))
        # Unknowns/invalid schemas must not become an extra all-zero model.
        base=fixture(1,(0,));E=np.zeros((1,22));E[0,[0,21]]=[1.,1.]
        unknowns=[('none_matrix',dict(matrix=None,target=[1.]),'unknown'),('none_target',dict(matrix=E,target=None),'unknown'),
            ('nan_matrix',dict(matrix=np.full((1,22),np.nan),target=[1.]),'unknown'),('257_rows',dict(matrix=np.zeros((257,22)),target=np.zeros(257)),'unknown'),
            ('bool_coeff',dict(matrix=np.zeros((1,22),bool),target=[0.]),'invalid_input'),('string_coeff',dict(matrix=[['0']*22],target=[0.]),'invalid_input'),
            ('wrong_columns',dict(matrix=np.zeros((1,21)),target=[0.]),'invalid_input'),('extra_truth_key',dict(matrix=E,target=[0.],true_mass=.12),'invalid_input')]
        for name,extra,status in unknowns:
            q=deepcopy(base);q['extra_task_rows']=extra;r,_=call(q);check(name+'/honest_unavailable',not r.available and r.status==status and r.d21 is None and r.zP is None,dict(status=r.status,reason=r.reason))
        # Large rows keep ORIGINAL-unit criteria; closeness of value is not KKT.
        large=[]
        for weight in (1e3,1e5,1e8):
            q=fixture(1,(0,));q['z_intent_p'][0]=1.;E=np.zeros((1,22));E[0,[0,21]]=weight;q['extra_task_rows']=dict(matrix=E,target=np.array([2.*weight]));r,calls=call(q)
            x=np.r_[r.d21,r.zP] if r.d21 is not None else None;g=raw(q,x)[1] if x is not None else None
            td=q['atol']+q['rtol']*max(1.,norm(r.d21)) if x is not None else None;tz=q['atol']+q['rtol']*max(1.,norm(r.zP)) if x is not None else None
            if r.available:check('large/'+str(weight)+'/raw_interor_gradient_passes_original_tolerance',norm(g[:21][[0]])<=td and norm(g[21:])<=tz)
            else:check('large/'+str(weight)+'/does_not_claim_convergence',r.status in ('nonconvergence','numerical_stagnation','numerical_failure'))
            check('large/'+str(weight)+'/tolerance_not_relaxed',r.diagnostics['computational_budget']['atol']==1e-8 and r.diagnostics['computational_budget']['rtol']==1e-7 and r.diagnostics['computational_budget']['max_iterations']==400)
            large.append(dict(weight=weight,status=r.status,available=r.available,iterations=r.iterations,raw_gradient_inf=None if g is None else norm(g),td=td,tz=tz))
        q=deepcopy(base);E=np.zeros((2,22));E[:,[0,21]]=[[1.,2.],[-3.,.5]];q['extra_task_rows']=dict(matrix=E,target=[3.,-1.]);q['max_iterations']=1;r,_=call(q)
        check('one_iteration_not_false_global_optimal',not r.available and r.status in ('nonconvergence','numerical_stagnation'),dict(status=r.status,reason=r.reason))
        check('source_hash_unchanged',sha(KERNEL)==EXPECTED)
        check('no_robot_mujoco_or_physics_import','mujoco' not in sys.modules and 'hand_plant' not in sys.modules)
        report=dict(schema='v15_generic_task_rows_independent_v1',passed=not failures,checks=checks,failures=failures,cases=cases,large_rows=large,
            independent_oracle='Original-coordinate global enumeration of active halfspaces and both hinge regions; independent raw objective/gradient, numeric derivative, original-unit stationarity/dual/complementarity; no production projection/KKT/active-set helper.',
            scope='Small synthetic mathematical fixtures only; no caller task-row design, observer, sensor, native realization, runtime, physics or clinical/material certificate.',
            source_sha256=EXPECTED,checker_sha256=sha(__file__),physics_steps=0,model_imports=0)
    except Exception as exc:report=dict(schema='v15_generic_task_rows_independent_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),checks=checks,failures=failures,cases=cases,source_sha256=EXPECTED,physics_steps=0)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failures=failures[:10],cases=len(cases),path=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

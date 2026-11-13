"""Pure independent objective/KKT/active-subset oracle for general task rows."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
def bits(v):
    if isinstance(v,np.ndarray):return b'A'+str(v.dtype).encode()+repr(v.shape).encode()+v.tobytes()
    if isinstance(v,(float,np.floating)):return b'F'+struct.pack('<d',float(v))
    if isinstance(v,(bool,np.bool_)):return b'B'+bytes([bool(v)])
    if isinstance(v,(int,np.integer)):return b'I'+str(int(v)).encode()+b';'
    if v is None:return b'N'
    if isinstance(v,str):return b'S'+v.encode()
    if isinstance(v,dict):return b'D'+b''.join(bits(k)+bits(x) for k,x in v.items())
    if isinstance(v,(list,tuple)):return b'L'+b''.join(bits(x) for x in v)
    raise TypeError(type(v))
def main():
    out=ROOT/'review/v15_task_rows_author_v1.json'
    if out.exists():raise FileExistsError('Fresh immutable math evidence required')
    from v15_support import task_qp as kernel
    from v14_coord.intent_qp import solve_intent_qp as baseline
    from review.verify_v13_joint_intent_qp_independent_v1 import fixture,objective as base_objective,gradient as base_gradient,active_qp
    source=Path(kernel.__file__).resolve();before=sha(source);checks=[];records=[];rng=np.random.default_rng(9153007)
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=clean(detail)))
    def objective(x,k,E,b):return base_objective(x,k)+float(np.square(E@x-b).sum())
    def gradient(x,k,E,b):return base_gradient(x,k)+2*E.T@(E@x-b)
    def result_case(name,k,E,b,expected=None):
        ecopy=E.copy();bcopy=b.copy();r=kernel.solve_intent_qp(**k,extra_task_rows={'matrix':E,'target':b})
        record=dict(name=name,status=r.status,reason=r.reason,iterations=r.iterations,d21=clean(r.d21),zP=clean(r.zP),objective=r.objective,diagnostics=clean(r.diagnostics));records.append(record)
        check(name+'_inputs_not_changed',bits(E)==bits(ecopy) and bits(b)==bits(bcopy))
        if r.d21 is not None:
            x=np.r_[r.d21,r.zP];g=gradient(x,k,E,b);record['independent_gradient']=g.tolist()
            check(name+'_raw_objective',abs(objective(x,k,E,b)-r.objective)<=1e-11*max(1.,abs(r.objective)))
            check(name+'_raw_extra_residual',np.array_equal(E@x-b,np.asarray(r.diagnostics['extra_task_residual'])))
            check(name+'_LS_sign_and_hinge_equivalence',abs(r.objective-r.diagnostics['expanded_least_squares_objective'])<=1e-10*max(1.,abs(r.objective)))
            if r.available:
                check(name+'_original_unit_dKKT',r.diagnostics['direct_KKT_inf_d21']<=r.diagnostics['numerical_tolerance_d21'])
                check(name+'_original_unit_zKKT',r.diagnostics['direct_KKT_inf_zP']<=r.diagnostics['numerical_tolerance_zP'])
        if expected is not None:check(name+'_independent_closed_form',r.available and np.max(abs(np.r_[r.d21,r.zP]-expected))<1e-9)
        return r
    k=fixture();E=np.zeros((1,22));E[0,[0,21]]=1.;b=np.array([4.]);expected=np.r_[1.,np.zeros(20),2.]
    result_case('coupled_effort_pressure',k,E,b,expected)
    k=fixture();k['caps5'][0]=1.;expected=np.r_[1.5,np.zeros(20),1.]
    result_case('coupled_group_cap',k,E,b,expected)
    k=fixture();k['effort_upper21'][0]=.1;expected=np.r_[.1,np.zeros(20),2.45]
    result_case('coupled_effort_upper',k,E,b,expected)
    k=fixture(0);E0=np.zeros((1,21));E0[0,0]=2.;result_case('emptyP_effort_only',k,E0,np.array([6.]),np.r_[2.4,np.zeros(20)])
    # Same variable optimum, nonzero constant residual must remain in the loss.
    k=fixture();r=result_case('zero_matrix_nonzero_target',k,np.zeros((2,22)),np.array([3.,-4.]),np.r_[np.zeros(21),1.]);check('constant_residual_not_dropped',r.objective==25.)
    for P in (0,1,2,5,32,128):
        k=fixture(P);k['A21x25']=rng.normal(0,.1,(21,25));k['Jpx25']=rng.normal(0,.1,(P,25));k['y_p']=rng.uniform(0,1,P)
        old=baseline(**k)
        for mode,extra in [('omitted',None),('explicit_empty',{'matrix':np.empty((0,21+P)),'target':np.empty(0)}),('json_empty',{'matrix':[],'target':[]})]:
            new=kernel.solve_intent_qp(**k,extra_task_rows=extra)
            for name in ('status','reason','d21','zP','p21','residual25','load_shortfall_N','objective','iterations'):
                check(f'default_P{P}_{mode}_{name}_exact',bits(getattr(old,name))==bits(getattr(new,name)))
            check(f'default_P{P}_{mode}_all_original_diagnostics_exact',all(k in new.diagnostics and bits(v)==bits(new.diagnostics[k]) for k,v in old.diagnostics.items()))
    # Independent enumeration on four variables; no production active-set code.
    for case in range(20):
        P=2;k=fixture(P);D=[0,1];k['A21x25'][D]=rng.normal(0,.3,(2,25));k['Jpx25']=rng.normal(0,.2,(P,25));k['y_p']=rng.uniform(0,1,P);k['z_intent_p']=rng.uniform(0,2,P)
        k['p0_21']=rng.normal(0,.2,21);k['effort_lower21']=k['p0_21'].copy();k['effort_upper21']=k['p0_21'].copy();k['effort_lower21'][D]-=rng.uniform(.05,1,2);k['effort_upper21'][D]+=rng.uniform(.05,1,2)
        k['caps5']=rng.uniform(.1,1.5,5);k['total_budget']=float(rng.uniform(.2,1.8));k['support_coeff_p']=rng.uniform(-.5,1.,P);k['load_proxy_N']=float(rng.uniform(0,2));k['scales']=dict(residual25=rng.uniform(.5,2,25),intentP=rng.uniform(.5,2,P),delta21=rng.uniform(.5,2,21),load=float(rng.uniform(.5,2)))
        E=np.zeros((1+case%5,23));compact_indices=np.array(D+[21,22]);E[:,compact_indices]=rng.normal(0,1,(len(E),4));b=rng.normal(0,2,len(E));r=result_case('enumerated_'+str(case),k,E,b)
        sr=k['scales']['residual25'];sf=k['scales']['intentP'];sd=k['scales']['delta21'][D];sl=k['scales']['load'];M=np.c_[k['A21x25'][D].T,-k['Jpx25'].T];ec=E[:,compact_indices];q=np.r_[np.zeros(2),k['support_coeff_p']]
        H0=2*((sr[:,None]*M).T@(sr[:,None]*M)+np.diag(np.r_[sd*sd,sf*sf])+ec.T@ec)
        g0=2*(M.T@(sr*sr*(k['Jpx25'].T@k['y_p']))-np.r_[np.zeros(2),sf*sf*k['z_intent_p']]-ec.T@b)
        G=[];h=[]
        for j,idx in enumerate(D):
            row=np.eye(4)[j];G.extend([row,-row]);h.extend([k['effort_upper21'][idx]-k['p0_21'][idx],-(k['effort_lower21'][idx]-k['p0_21'][idx])])
        for j in range(2):G.append(-np.eye(4)[2+j]);h.append(0.)
        for digit in range(2):row=np.zeros(4);row[2+digit]=1.;G.append(row);h.append(k['caps5'][digit])
        G.append(np.r_[0.,0.,1.,1.]);h.append(k['total_budget']);best=None
        for hinge in (False,True):
            H=H0+2*sl*sl*np.outer(q,q) if hinge else H0;g=g0-2*sl*sl*k['load_proxy_N']*q if hinge else g0
            gg=np.vstack([*G,q if hinge else -q]);hh=np.r_[h,k['load_proxy_N'] if hinge else -k['load_proxy_N']]
            xx=active_qp(H,g,gg,hh)
            if xx is None:continue
            full=np.zeros(23);full[compact_indices]=xx;value=objective(full,k,E,b)
            if best is None or value<best[0]:best=(value,full)
        check('enumerated_'+str(case)+'_independent_global_min',best is not None and r.available and abs(r.objective-best[0])<1e-8 and np.max(abs(np.r_[r.d21,r.zP]-best[1]))<1e-7)
        record=records[-1];record['independent_active_subset_objective']=None if best is None else best[0]
    # Trace actual production evaluate, independently differentiate objective.
    captured=[]
    def trace(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='evaluate' and Path(frame.f_code.co_filename).resolve()==source:captured.append((frame.f_locals['x'].copy(),float(arg[0]),np.asarray(arg[1]).copy()))
        return trace
    for case in range(6):
        k=fixture(3);k['A21x25']=rng.normal(0,.2,(21,25));k['Jpx25']=rng.normal(0,.3,(3,25));k['y_p']=rng.uniform(0,2,3);k['support_coeff_p']=rng.uniform(-.5,1,3);k['load_proxy_N']=3. if case%2 else 0.
        E=rng.normal(0,.3,(4,24));b=rng.normal(0,1,4);captured.clear();oldtrace=sys.gettrace();sys.settrace(trace)
        try:r=kernel.solve_intent_qp(**k,extra_task_rows={'matrix':E,'target':b},max_iterations=1)
        finally:sys.settrace(oldtrace)
        err=0.
        for x,value,g in captured:
            fd=np.empty_like(x)
            for j in range(len(x)):
                step=1e-6*max(1.,abs(x[j]));a=x.copy();z=x.copy();a[j]+=step;z[j]-=step;fd[j]=(objective(a,k,E,b)-objective(z,k,E,b))/(2*step)
            err=max(err,float(max(abs(fd-g))))
        check('actual_gradient_with_rows_FD_'+str(case),bool(captured) and err<2e-6,err)
        T=k['scales']['residual25'][:,None]*np.c_[k['A21x25'].T,-k['Jpx25'].T];q=np.r_[np.zeros(21),k['support_coeff_p']];D=np.r_[k['scales']['delta21']**2,k['scales']['intentP']**2]
        bound=max(np.linalg.eigvalsh(2*(T.T@T+np.diag(D)+active*k['scales']['load']**2*np.outer(q,q)+E.T@E))[-1] for active in (0,1))
        check('valid_L_including_task_rows_'+str(case),r.diagnostics['lipschitz_upper_bound']+1e-10>=bound and not r.diagnostics['global_L_used_for_steps'])
    k=fixture();bad=[('missing_target',{'matrix':np.zeros((1,22))},'invalid_input'),('matrix_unknown',{'matrix':None,'target':np.ones(1)},'unknown'),('target_unknown',{'matrix':np.zeros((1,22)),'target':None},'unknown'),('matrix_nan',{'matrix':np.full((1,22),np.nan),'target':np.ones(1)},'unknown'),('matrix_bool',{'matrix':np.ones((1,22),bool),'target':np.ones(1)},'invalid_input'),('target_bool',{'matrix':np.ones((1,22)),'target':np.ones(1,bool)},'invalid_input'),('wrong_columns',{'matrix':np.zeros((1,23)),'target':np.ones(1)},'invalid_input'),('row_limit',{'matrix':np.zeros((257,22)),'target':np.zeros(257)},'unknown'),('target_mismatch',{'matrix':np.zeros((1,22)),'target':np.zeros(2)},'invalid_input'),('hardconstraint_key',{'matrix':np.zeros((1,22)),'target':np.zeros(1),'hard':True},'invalid_input')]
    for name,extra,status in bad:
        r=kernel.solve_intent_qp(**k,extra_task_rows=extra);check(name+'_honest_unavailable',r.status==status and r.d21 is None and r.zP is None,dict(status=r.status,reason=r.reason))
    result_case('maximum256rows',fixture(2),rng.normal(0,.1,(256,23)),rng.normal(0,1,256))
    k=fixture();E=np.zeros((1,22));E[0,[0,21]]=1e8;r=result_case('large_task_row_scale',k,E,np.array([1e8]));check('large_rows_no_false_available',not r.available or max(r.diagnostics['direct_KKT_inf_d21'],r.diagnostics['direct_KKT_inf_zP'])<=max(r.diagnostics['numerical_tolerance_d21'],r.diagnostics['numerical_tolerance_zP']))
    r=kernel.solve_intent_qp(**fixture(),extra_task_rows={'matrix':np.ones((1,22))*1e200,'target':np.ones(1)})
    check('overflow_not_zero_gain_or_false_success',not r.available and r.status=='numerical_failure')
    check('no_simulator_import','mujoco' not in sys.modules);check('frozen_base_unchanged',sha(ROOT/'v14_coord/intent_qp.py')=='1f56174b10af062dea161513874607e6971788d1a9589b2e2c414a5694eeeebc');check('candidate_unchanged',sha(source)==before)
    output=dict(schema='v15_general_task_rows_math_author_v1',passed=all(x['passed'] for x in checks),check_count=len(checks),failures=[x for x in checks if not x['passed']],checks=checks,records=records,
        candidate_sha256=before,checker_sha256=sha(__file__),scope='pure mathematics only; defaults exactV14, independent exhaustive4variable KKT subsets and production-gradient finite differences; no model/physics or objective-utility tuning',physics_steps=0)
    out.write_text(json.dumps(clean(output),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=output['passed'],checks=len(checks),failures=output['failures'],source_sha256=before,out=str(out)),ensure_ascii=False))
    return 0 if output['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
